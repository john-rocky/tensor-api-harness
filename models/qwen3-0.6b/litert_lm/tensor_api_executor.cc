// SPDX-License-Identifier: Apache-2.0
// The Qwen3-0.6B Tensor API graph as a LiteRT-LM executor. See
// tensor_api_executor.h.

#include "tensor_api/qwen3-0.6b/tensor_api_executor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"  // from @litert
#include "tensor/datatypes.h"  // from @litert
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"
#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"  // from @litert
#include "tensor/tensor.h"  // from @litert
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::tensor::examples::qwen3 {

namespace {

namespace lm = ::litert::lm;
// Qualified because TensorApiExecutor::Create hides it inside the class.
namespace lt = ::litert::tensor;

constexpr float kMaskFloor = -30000.0f;  // fp16-safe
constexpr int32_t kPadId = 0;            // right padding of a prefill chunk
constexpr char kDecodeSignature[] = "decode";

std::string PrefillSignature(int length) {
  return absl::StrCat("prefill_", length);
}

// The executor's processed context holds the token ids only; the caches stay
// in the executor.
class TensorApiProcessedContext : public lm::ProcessedContext {
 public:
  explicit TensorApiProcessedContext(std::optional<uint32_t> lora_id,
                                     lm::ProcessedTokens tokens = {})
      : lora_id_(lora_id), tokens_(std::move(tokens)) {}

  std::optional<uint32_t> lora_id() const override { return lora_id_; }
  void set_lora_id(std::optional<uint32_t> lora_id) override {
    lora_id_ = lora_id;
  }
  lm::ProcessedTokens& processed_tokens() override { return tokens_; }

 private:
  std::optional<uint32_t> lora_id_;
  lm::ProcessedTokens tokens_;
};

// RoPE cos/sin rows for positions pos .. pos + seq - 1 (rotate-half layout).
void BuildRopeCosSin(int pos, int seq, int head_dim, float base,
                     std::vector<float>& cos_out, std::vector<float>& sin_out) {
  cos_out.resize(static_cast<size_t>(seq) * head_dim);
  sin_out.resize(static_cast<size_t>(seq) * head_dim);
  int half = head_dim / 2;
  for (int s = 0; s < seq; ++s) {
    for (int i = 0; i < half; ++i) {
      float inv_freq =
          1.0f / std::pow(base, static_cast<float>(2 * i) / head_dim);
      float angle = static_cast<float>(pos + s) * inv_freq;
      cos_out[s * head_dim + i] = cos_out[s * head_dim + half + i] =
          std::cos(angle);
      sin_out[s * head_dim + i] = sin_out[s * head_dim + half + i] =
          std::sin(angle);
    }
  }
}

// Row s (absolute position valid_len + s) sees positions 0..valid_len + s.
std::vector<float> BuildMask(int seq_q, int valid_len, int max_seq) {
  std::vector<float> mask(static_cast<size_t>(seq_q) * max_seq, kMaskFloor);
  for (int s = 0; s < seq_q; ++s) {
    int limit = std::min(valid_len + s + 1, max_seq);
    for (int j = 0; j < limit; ++j) mask[s * max_seq + j] = 0.0f;
  }
  return mask;
}

// Long id lists are logged by their first and last 32 ids.
std::string IdsStr(absl::Span<const int> ids) {
  constexpr size_t kEnds = 32;
  if (ids.size() <= 2 * kEnds) return absl::StrJoin(ids, " ");
  return absl::StrCat(absl::StrJoin(ids.first(kEnds), " "), " ... ",
                      absl::StrJoin(ids.last(kEnds), " "));
}

absl::StatusOr<::litert::Options> MakeOptions(
    const TensorApiExecutorOptions& o, int num_threads) {
  LITERT_ASSIGN_OR_RETURN(::litert::Options options,
                          ::litert::Options::Create());
  if (!o.use_gpu) {
    LITERT_RETURN_IF_ERROR(
        options.SetHardwareAccelerators(::litert::HwAccelerators::kCpu));
    auto cpu_options = options.GetCpuOptions();
    if (!cpu_options) return absl::InternalError("no CPU options");
    if (num_threads > 0) {
      LITERT_RETURN_IF_ERROR(cpu_options->SetNumThreads(num_threads));
    }
    if (o.litert_lm_xnnpack_flags) {
      // As SetCpuOptions in LiteRT-LM's
      // runtime/executor/litert_compiled_model_executor_utils.cc.
      LITERT_RETURN_IF_ERROR(cpu_options->SetXNNPackFlags(
          TfLiteXNNPackDelegateOptionsDefault().flags |
          TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
    }
    return std::move(options);
  }
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(::litert::HwAccelerators::kGpu));
  auto gpu_options = options.GetGpuOptions();
  if (!gpu_options) return absl::InternalError("no GPU options");
  if (o.gpu_precision == "fp32") {
    LITERT_RETURN_IF_ERROR(
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32));
  } else if (o.gpu_precision != "fp16") {
    return absl::InvalidArgumentError("GPU precision must be fp32|fp16");
  }
  if (o.gpu_buffer_storage == "buffer") {
    LITERT_RETURN_IF_ERROR(gpu_options->SetBufferStorageType(
        ::litert::GpuOptions::BufferStorageType::kBuffer));
  } else if (o.gpu_buffer_storage == "texture2d") {
    LITERT_RETURN_IF_ERROR(gpu_options->SetBufferStorageType(
        ::litert::GpuOptions::BufferStorageType::kTexture2D));
  } else if (o.gpu_buffer_storage != "default") {
    return absl::InvalidArgumentError(
        "GPU buffer storage must be default|buffer|texture2d");
  }
  return std::move(options);
}

std::string HwName(LiteRtHwAcceleratorSet hw) {
  std::vector<std::string> names;
  if (hw & kLiteRtHwAcceleratorCpu) names.push_back("cpu");
  if (hw & kLiteRtHwAcceleratorGpu) names.push_back("gpu");
  if (hw & kLiteRtHwAcceleratorNpu) names.push_back("npu");
  if (names.empty()) names.push_back("none");
  return absl::StrJoin(names, "+");
}

}  // namespace

TensorApiExecutor::TensorApiExecutor(
    const lm::LlmExecutorSettings& settings,
    const TensorApiExecutorOptions& options)
    : settings_(settings), options_(options) {}

absl::StatusOr<std::unique_ptr<TensorApiExecutor>> TensorApiExecutor::Create(
    const lm::LlmExecutorSettings& settings,
    const TensorApiExecutorOptions& options) {
  if (options.weights_path.empty()) {
    return absl::InvalidArgumentError(
        "no weights: pass the model.safetensors path with "
        "--tensor_api_weights");
  }
  auto executor = absl::WrapUnique(new TensorApiExecutor(settings, options));
  ABSL_RETURN_IF_ERROR(executor->Initialize());
  return executor;
}

absl::Status TensorApiExecutor::Initialize() {
  const int max_seq = static_cast<int>(settings_.GetMaxNumTokens());
  if (max_seq <= 0) {
    return absl::InvalidArgumentError("max_num_tokens is not set");
  }
  config_.max_seq = max_seq;
  config_.separate_lm_head = options_.lm_head_int8;
  prefill_lengths_ = options_.prefill_lengths;
  std::sort(prefill_lengths_.begin(), prefill_lengths_.end());
  prefill_lengths_.erase(
      std::unique(prefill_lengths_.begin(), prefill_lengths_.end()),
      prefill_lengths_.end());
  if (prefill_lengths_.empty() || prefill_lengths_.front() <= 0 ||
      prefill_lengths_.back() > max_seq) {
    return absl::InvalidArgumentError(
        absl::StrCat("prefill lengths must be in 1..", max_seq));
  }
  int num_threads = options_.num_threads;
  if (!options_.use_gpu && num_threads == 0) {
    auto cpu_config = settings_.GetBackendConfig<lm::CpuConfig>();
    if (cpu_config.ok()) num_threads = cpu_config->number_of_threads;
  }
  const char* quant = options_.weights_quant == WeightQuantMode::kInt8 ? "int8"
                      : options_.weights_quant == WeightQuantMode::kInt4
                          ? "int4"
                          : "none";
  ABSL_LOG(INFO) << "tensor_api: " << (options_.use_gpu ? "GPU" : "CPU")
                 << " executor, weights " << options_.weights_path
                 << ", KV cache length " << max_seq
                 << ", prefill signatures "
                 << absl::StrJoin(prefill_lengths_, ",") << ", weights_quant "
                 << quant << ", lm_head "
                 << (options_.lm_head_int8 ? "int8 copy" : "tied fp32 table")
                 << (options_.use_gpu
                         ? absl::StrCat(", GPU precision ",
                                        options_.gpu_precision, ", storage ",
                                        options_.gpu_buffer_storage)
                         : absl::StrCat(", CPU threads ", num_threads,
                                        ", XNNPACK flags ",
                                        options_.litert_lm_xnnpack_flags
                                            ? "of LiteRT-LM's CPU executor"
                                            : "LiteRT default"));

  QuantStats stats;
  ABSL_ASSIGN_OR_RETURN(
      WeightMap weights,
      LoadCheckpointWeights(config_, options_.weights_path,
                            options_.weights_quant, options_.lm_head_int8,
                            &stats));
  ABSL_LOG(INFO) << "tensor_api: " << weights.size()
                 << " weight tensors loaded (bf16 -> fp32)"
                 << (stats.quantized_tensors > 0
                         ? absl::StrCat(", ", stats.quantized_tensors,
                                        " quantized to ",
                                        stats.quantized_bytes, " bytes")
                         : std::string())
                 << (options_.lm_head_int8
                         ? absl::StrCat(", lm_head int8 copy ",
                                        stats.lm_head_bytes, " bytes")
                         : std::string());

  ModelFactory factory;
  auto add_signature = [&](const Qwen3Config& config, bool is_decode,
                           const std::string& name) -> absl::Status {
    Qwen3Inputs in = MakeQwen3Inputs(config, is_decode);
    Qwen3Outputs out = BuildQwen3(config, in, weights, is_decode);
    std::vector<TensorHandle> ins, outs;
    for (auto& t : in.AsList()) ins.push_back(t);
    for (auto& t : out.AsList()) outs.push_back(t);
    return factory.AddSignature(ins, outs, name);
  };
  for (int length : prefill_lengths_) {
    Qwen3Config config = config_;
    config.prefill_len = length;
    config.omit_logits = true;
    ABSL_RETURN_IF_ERROR(
        add_signature(config, /*is_decode=*/false, PrefillSignature(length)));
  }
  ABSL_RETURN_IF_ERROR(
      add_signature(config_, /*is_decode=*/true, kDecodeSignature));
  ABSL_ASSIGN_OR_RETURN(model_buffer_, factory.CreateFlatbuffer());
  weights.clear();
  ABSL_LOG(INFO) << "tensor_api: model built in memory, "
                 << model_buffer_.size() << " bytes";

  // The registry's creator gets no Environment: the executor owns one.
  const std::vector<::litert::EnvironmentOptions::Option> env_options;
  LITERT_ASSIGN_OR_RETURN(
      ::litert::Environment env,
      ::litert::Environment::Create(::litert::EnvironmentOptions(env_options)));
  env_ = std::make_unique<::litert::Environment>(std::move(env));
  const absl::Span<const uint8_t> model_span(
      reinterpret_cast<const uint8_t*>(model_buffer_.data()),
      model_buffer_.size());
  if (options_.delegation_report) {
    ABSL_ASSIGN_OR_RETURN(::litert::Options report_options,
                          MakeOptions(options_, num_threads));
    ::litert::BufferRef<uint8_t> buffer_ref(model_span.data(),
                                            model_span.size());
    LITERT_ASSIGN_OR_RETURN(
        ::litert::CompiledModel compiled,
        ::litert::CompiledModel::Create(*env_, buffer_ref, report_options));
    LITERT_ASSIGN_OR_RETURN(auto metrics, compiled.GetDelegationMetrics());
    ABSL_LOG(INFO) << "tensor_api delegation: total_nodes "
                   << metrics.total_node_count;
    for (const auto& acc : metrics.accelerators) {
      ABSL_LOG(INFO) << "tensor_api delegation: " << acc.accelerator_name
                     << " hw=" << HwName(acc.hardware_type)
                     << " delegated_nodes=" << acc.delegated_node_count
                     << " partitions=" << acc.partition_count;
    }
  }
  ABSL_ASSIGN_OR_RETURN(::litert::Options options,
                        MakeOptions(options_, num_threads));
  ABSL_ASSIGN_OR_RETURN(
      LitertDynamicRunner runner,
      LitertDynamicRunner::Create(*env_, model_span, options));
  runner_.emplace(std::move(runner));
  ABSL_LOG(INFO) << "tensor_api: compiled model and runner created";

  const size_t cache_n = static_cast<size_t>(config_.n_kv_groups) *
                         config_.max_seq * config_.head_dim;
  zero_cache_ = lt::Create(
      "zero_cache", Type::kFP32,
      {1, config_.n_kv_groups, config_.max_seq, config_.head_dim},
      std::vector<float>(cache_n, 0.0f));
  LITERT_ASSIGN_OR_RETURN(
      logits_host_, lm::CreateTensorBuffer<float>({1, 1, config_.vocab_size}));
  ABSL_ASSIGN_OR_RETURN(llm_context_,
                        CreateNewContext(std::nullopt, lm::RuntimeConfig()));
  return Reset();
}

lm::ProcessedTokens& TensorApiExecutor::Tokens() {
  return llm_context_->processed_context().processed_tokens();
}

absl::Status TensorApiExecutor::RollBackTokens() {
  const int step = llm_context_->runtime_state().current_step;
  lm::ProcessedTokens& tokens = Tokens();
  if (step == tokens.TokenCount()) return absl::OkStatus();
  if (step == 0) {
    ABSL_RETURN_IF_ERROR(tokens.RollBackToStep(0));
  } else {
    std::vector<int> token_at_step = tokens.GetTokenAtStep(step - 1);
    ABSL_RETURN_IF_ERROR(tokens.RollBackToStep(step - 1));
    if (!token_at_step.empty()) {
      ABSL_RETURN_IF_ERROR(tokens.AddPendingInputToken(
          {std::make_shared<lm::TokenData>(token_at_step[0])}));
    }
  }
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::SetStepInputs(const std::string& signature,
                                              const std::vector<int32_t>& ids,
                                              int position) {
  const int n = static_cast<int>(ids.size());
  const int hd = config_.head_dim;
  ABSL_RETURN_IF_ERROR(runner_->SetInput(
      signature, "input_ids",
      lt::Create("input_ids", Type::kI32, {n}, std::vector<int32_t>(ids))));
  ABSL_RETURN_IF_ERROR(runner_->SetInput(
      signature, "cache_position",
      lt::Create("cache_position", Type::kI32, {4},
                 std::vector<int32_t>{0, 0, position, 0})));
  std::vector<float> cos_v, sin_v;
  BuildRopeCosSin(position, n, hd, config_.rope_base, cos_v, sin_v);
  ABSL_RETURN_IF_ERROR(runner_->SetInput(
      signature, "rope_cos",
      lt::Create("rope_cos", Type::kFP32, {1, 1, n, hd}, std::move(cos_v))));
  ABSL_RETURN_IF_ERROR(runner_->SetInput(
      signature, "rope_sin",
      lt::Create("rope_sin", Type::kFP32, {1, 1, n, hd}, std::move(sin_v))));
  return runner_->SetInput(
      signature, "attention_mask",
      lt::Create("attention_mask", Type::kFP32, {1, 1, n, config_.max_seq},
                 BuildMask(n, position, config_.max_seq)));
}

absl::Status TensorApiExecutor::BindCaches(const std::string& signature) {
  for (int i = 0; i < config_.n_layers; ++i) {
    ABSL_RETURN_IF_ERROR(runner_->SetInput(
        signature, absl::StrCat("key_cache_", i), key_caches_[i]));
    ABSL_RETURN_IF_ERROR(runner_->SetInput(
        signature, absl::StrCat("value_cache_", i), value_caches_[i]));
  }
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::TakeCaches(const std::string& signature) {
  for (int i = 0; i < config_.n_layers; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        key_caches_[i],
        runner_->GetOutput(signature, absl::StrCat("key_cache_", i, "_new")));
    ABSL_ASSIGN_OR_RETURN(
        value_caches_[i],
        runner_->GetOutput(signature, absl::StrCat("value_cache_", i, "_new")));
  }
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::RunPrefill(absl::Span<const int> ids,
                                           int start) {
  size_t done = 0;
  int position = start;
  while (done < ids.size()) {
    const int remaining = static_cast<int>(ids.size() - done);
    // The shortest signature that holds the rest, the longest one otherwise.
    // DynamicUpdateSlice moves a start that would run past the cache, so a
    // chunk must fit at its position.
    int length = 0;
    for (int l : prefill_lengths_) {
      if (position + l > config_.max_seq) break;
      length = l;
      if (l >= remaining) break;
    }
    if (length == 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "no prefill signature fits at position ", position,
          " (KV cache length ", config_.max_seq, ")"));
    }
    const int n = std::min(remaining, length);
    std::vector<int32_t> chunk(ids.begin() + done, ids.begin() + done + n);
    chunk.resize(length, kPadId);
    const std::string signature = PrefillSignature(length);
    ABSL_RETURN_IF_ERROR(SetStepInputs(signature, chunk, position));
    ABSL_RETURN_IF_ERROR(BindCaches(signature));
    ABSL_RETURN_IF_ERROR(runner_->Run(signature));
    ABSL_RETURN_IF_ERROR(TakeCaches(signature));
    cached_ids_.resize(position);
    cached_ids_.insert(cached_ids_.end(), chunk.begin(), chunk.begin() + n);
    position += n;
    done += n;
  }
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::RunDecode(int id, int position) {
  if (position >= config_.max_seq) {
    return absl::FailedPreconditionError(absl::StrCat(
        "position ", position, " is past the KV cache length ",
        config_.max_seq));
  }
  ABSL_RETURN_IF_ERROR(SetStepInputs(kDecodeSignature, {id}, position));
  ABSL_RETURN_IF_ERROR(BindCaches(kDecodeSignature));
  ABSL_RETURN_IF_ERROR(runner_->Run(kDecodeSignature));
  ABSL_RETURN_IF_ERROR(TakeCaches(kDecodeSignature));
  cached_ids_.resize(position);
  cached_ids_.push_back(id);

  ABSL_ASSIGN_OR_RETURN(TensorHandle logits,
                        runner_->GetOutput(kDecodeSignature, "logits"));
  auto buffer = logits.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  logits_.assign(data, data + lock.size() / sizeof(float));
  if (logits_.size() != static_cast<size_t>(config_.vocab_size)) {
    return absl::InternalError(
        absl::StrCat("decode logits have ", logits_.size(), " values"));
  }
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::WaitForCaches() {
  auto buffer = value_caches_.back().GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  if (lock.size() == 0) return absl::InternalError("empty cache buffer");
  return absl::OkStatus();
}

absl::Status TensorApiExecutor::Prefill(const lm::ExecutorInputs& inputs) {
  return Prefill(inputs, lm::ExecutorPrefillParams());
}

absl::Status TensorApiExecutor::Prefill(
    const lm::ExecutorInputs& inputs,
    const lm::ExecutorPrefillParams& prefill_params) {
  if (inputs.GetVisionDataPtr().ok() || inputs.GetAudioDataPtr().ok()) {
    return absl::UnimplementedError("text input only");
  }
  if (prefill_params.GetCurrentStep() >= 0) {
    ABSL_RETURN_IF_ERROR(SetCurrentStep(prefill_params.GetCurrentStep()));
  }
  ABSL_ASSIGN_OR_RETURN(const ::litert::TensorBuffer* ids_buffer,
                        inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(std::vector<int> ids,
                          lm::CopyFromTensorBuffer<int>(*ids_buffer));
  if (ids.empty()) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(RollBackTokens());

  // Run the pending token (if any) and every id but the last, which becomes
  // the pending token.
  lm::ProcessedTokens& tokens = Tokens();
  const lm::ProcessedTokens::StepAndToken next =
      tokens.GetNextUnprocessedToken();
  std::vector<int> run;
  if (!next.token.empty()) run.push_back(next.token[0]->id());
  run.insert(run.end(), ids.begin(), ids.end() - 1);
  if (options_.log_ids) {
    ABSL_LOG(INFO) << "tensor_api prefill: " << ids.size() << " ids ["
                   << IdsStr(ids) << "], runs " << run.size()
                   << " from position " << next.step << ", pending "
                   << ids.back();
  }
  if (!run.empty()) {
    ABSL_RETURN_IF_ERROR(RunPrefill(run, next.step));
    if (options_.use_gpu && prefill_params.GetWaitForCompletion()) {
      ABSL_RETURN_IF_ERROR(WaitForCaches());
    }
    if (!next.token.empty()) {
      ABSL_RETURN_IF_ERROR(tokens.MarkPendingInputTokenAsProcessed());
    }
    tokens.AddProcessedTokens(std::vector<int>(ids.begin(), ids.end() - 1));
  }
  ABSL_RETURN_IF_ERROR(tokens.AddPendingInputToken(
      {std::make_shared<lm::TokenData>(ids.back())}));
  llm_context_->runtime_state().current_step = tokens.TokenCount();
  llm_context_->runtime_state().ran_decode = false;
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer> TensorApiExecutor::DecodeLogits(
    const lm::ExecutorInputs& inputs) {
  ABSL_RETURN_IF_ERROR(RollBackTokens());
  lm::ProcessedTokens& tokens = Tokens();
  // An id from the caller (the sampler's last pick) replaces the pending
  // token.
  if (inputs.GetTextDataPtr().ok()) {
    ABSL_ASSIGN_OR_RETURN(const ::litert::TensorBuffer* ids_buffer,
                          inputs.GetTextTokenIdsPtr());
    LITERT_ASSIGN_OR_RETURN(std::vector<int> ids,
                            lm::CopyFromTensorBuffer<int>(*ids_buffer));
    if (!ids.empty() && ids[0] >= 0) {
      if (ids.size() != 1) {
        return absl::InvalidArgumentError("one decode id per call");
      }
      tokens.InvalidatePendingInputToken();
      ABSL_RETURN_IF_ERROR(tokens.AddPendingInputToken(
          {std::make_shared<lm::TokenData>(ids[0])}));
    }
  }
  const lm::ProcessedTokens::StepAndToken next =
      tokens.GetNextUnprocessedToken();
  if (next.token.empty()) {
    return absl::FailedPreconditionError("no token to decode");
  }
  const int id = next.token[0]->id();
  if (options_.log_ids) {
    ABSL_LOG(INFO) << "tensor_api decode: id " << id << " at position "
                   << next.step;
  }
  ABSL_RETURN_IF_ERROR(RunDecode(id, next.step));
  ABSL_RETURN_IF_ERROR(tokens.MarkPendingInputTokenAsProcessed());
  ++llm_context_->runtime_state().current_step;
  llm_context_->runtime_state().ran_decode = true;

  LITERT_ASSIGN_OR_RETURN(auto host,
                          lm::ReferTensorBufferAsSpan<float>(logits_host_));
  std::copy(logits_.begin(), logits_.end(), host.begin());
  LITERT_ASSIGN_OR_RETURN(::litert::TensorBuffer output,
                          logits_host_.Duplicate());
  return output;
}

absl::Status TensorApiExecutor::Decode(const lm::ExecutorInputs& inputs,
                                       ::litert::TensorBuffer& output_logits) {
  ABSL_ASSIGN_OR_RETURN(output_logits, DecodeLogits(inputs));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>> TensorApiExecutor::Decode() {
  return Decode(lm::ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>> TensorApiExecutor::Decode(
    const lm::ExecutorDecodeParams& decode_params) {
  ABSL_RETURN_IF_ERROR(RollBackTokens());
  lm::ProcessedTokens& tokens = Tokens();
  const lm::ProcessedTokens::StepAndToken next =
      tokens.GetNextUnprocessedToken();
  if (next.token.empty()) {
    return absl::FailedPreconditionError("no token to decode");
  }
  const int id = next.token[0]->id();
  ABSL_RETURN_IF_ERROR(RunDecode(id, next.step));
  ABSL_RETURN_IF_ERROR(tokens.MarkPendingInputTokenAsProcessed());
  const int picked = static_cast<int>(
      std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
  if (options_.log_ids) {
    ABSL_LOG(INFO) << "tensor_api decode: id " << id << " at position "
                   << next.step << " -> " << picked;
  }
  ABSL_RETURN_IF_ERROR(tokens.AddPendingInputToken(
      {std::make_shared<lm::TokenData>(picked)}));
  llm_context_->runtime_state().current_step = tokens.TokenCount();
  llm_context_->runtime_state().ran_decode = true;
  return std::vector<std::vector<int>>{{picked}};
}

absl::string_view TensorApiExecutor::ExecutorBackendName() const {
  return options_.use_gpu ? "TensorApiExecutorGpu" : "TensorApiExecutorCpu";
}

absl::StatusOr<int> TensorApiExecutor::GetVocabSize() {
  return config_.vocab_size;
}

absl::StatusOr<int> TensorApiExecutor::GetCurrentStep() const {
  return llm_context_->runtime_state().current_step;
}

absl::Status TensorApiExecutor::SetCurrentStep(int current_step) {
  const int old_step = llm_context_->runtime_state().current_step;
  if (old_step == current_step) return absl::OkStatus();
  const int max_step = Tokens().TokenCount();
  if (current_step < 0 || current_step > max_step) {
    return absl::InvalidArgumentError(absl::StrCat(
        "step ", current_step, " is outside 0..", max_step));
  }
  llm_context_->runtime_state().current_step = current_step;
  return absl::OkStatus();
}

absl::StatusOr<lm::LlmExecutorSettings> TensorApiExecutor::GetExecutorSettings()
    const {
  return settings_;
}

absl::Status TensorApiExecutor::UpdateExecutorSettings(
    const lm::LlmExecutorSettings& executor_settings) {
  settings_ = executor_settings;
  if (static_cast<int>(settings_.GetMaxNumTokens()) != config_.max_seq) {
    ABSL_LOG(WARNING) << "tensor_api: max_num_tokens "
                      << settings_.GetMaxNumTokens()
                      << " after creation; the KV cache stays at "
                      << config_.max_seq;
    settings_.SetMaxNumTokens(config_.max_seq);
  }
  return absl::OkStatus();
}

absl::StatusOr<lm::RuntimeConfig> TensorApiExecutor::GetRuntimeConfig() const {
  return llm_context_->runtime_config();
}

absl::Status TensorApiExecutor::UpdateRuntimeConfig(
    const lm::RuntimeConfig& runtime_config) {
  llm_context_->runtime_config() = runtime_config;
  return absl::OkStatus();
}

absl::StatusOr<lm::RuntimeState> TensorApiExecutor::GetRuntimeState() const {
  return llm_context_->runtime_state();
}

absl::Status TensorApiExecutor::UpdateRuntimeState(
    const lm::RuntimeState& runtime_state) {
  llm_context_->runtime_state() = runtime_state;
  return absl::OkStatus();
}

absl::StatusOr<const lm::ProcessedTokens*>
TensorApiExecutor::GetProcessedTokens() const {
  return &llm_context_->processed_context().processed_tokens();
}

::litert::Environment* TensorApiExecutor::GetEnvironment() const {
  return env_.get();
}

absl::Status TensorApiExecutor::Reset() {
  key_caches_.assign(config_.n_layers, zero_cache_);
  value_caches_.assign(config_.n_layers, zero_cache_);
  cached_ids_.clear();
  llm_context_->runtime_state().current_step = 0;
  return Tokens().RollBackToStep(0);
}

absl::StatusOr<std::unique_ptr<lm::LlmContext>>
TensorApiExecutor::CreateNewContext(std::optional<uint32_t> lora_id,
                                    lm::RuntimeConfig runtime_config) const {
  auto runtime_state = std::make_unique<lm::RuntimeState>();
  runtime_state->rand_gen = std::make_shared<std::default_random_engine>(
      runtime_config.sampler_params.has_value()
          ? runtime_config.sampler_params->seed()
          : 0);
  return std::make_unique<lm::LlmContext>(
      std::make_unique<TensorApiProcessedContext>(lora_id),
      std::make_unique<lm::RuntimeConfig>(std::move(runtime_config)),
      std::move(runtime_state));
}

absl::StatusOr<std::unique_ptr<lm::LlmContext>>
TensorApiExecutor::CloneContext() const {
  return std::make_unique<lm::LlmContext>(
      std::make_unique<TensorApiProcessedContext>(
          llm_context_->processed_context().lora_id(),
          llm_context_->processed_context().processed_tokens()),
      std::make_unique<lm::RuntimeConfig>(llm_context_->runtime_config()),
      std::make_unique<lm::RuntimeState>(llm_context_->runtime_state()));
}

absl::Status TensorApiExecutor::RestoreContext(
    std::unique_ptr<lm::LlmContext> llm_context) {
  if (llm_context == nullptr) {
    return absl::InvalidArgumentError("null context");
  }
  if (llm_context->runtime_state().current_step == 0) {
    llm_context_ = std::move(llm_context);
    return Reset();
  }
  // Only a context whose processed tokens are the ones in the caches can be
  // restored: the caches are not part of a context.
  const std::vector<int> context_ids =
      llm_context->processed_context().processed_tokens().GetCopyOfTokens()[0];
  const int processed = llm_context->processed_context()
                            .processed_tokens()
                            .GetNextUnprocessedToken()
                            .step;
  if (processed > static_cast<int>(cached_ids_.size()) ||
      !std::equal(context_ids.begin(), context_ids.begin() + processed,
                  cached_ids_.begin())) {
    return absl::UnimplementedError(
        "restoring a context whose keys and values are not in the caches");
  }
  llm_context_ = std::move(llm_context);
  return absl::OkStatus();
}

absl::StatusOr<std::string> TensorApiExecutor::GetProfileSummary() {
  return absl::FailedPreconditionError("Profiling is not enabled.");
}

}  // namespace litert::tensor::examples::qwen3
