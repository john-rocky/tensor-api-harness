// SPDX-License-Identifier: Apache-2.0
// Registers the Tensor API executor (tensor_api_executor.h) with LiteRT-LM's
// ExecutorBackendRegistry under one backend name per accelerator:
//
//   --backend=tensor_api_cpu   LiteRT CPU (XNNPACK)
//   --backend=tensor_api_gpu   the LiteRT GPU accelerator
//
// LiteRT-LM's Backend value names the hardware and the executor at once, so
// the accelerator is part of the name. The executor returns logits and the
// session samples them (BackendTraits defaults: external CPU sampler). Link
// this file with alwayslink = 1; the flags below are parsed by the main's
// absl::ParseCommandLine together with LiteRT-LM's flags.

#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_backend_registry.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"
#include "tensor_api/qwen3-0.6b/tensor_api_executor.h"

ABSL_FLAG(std::string, tensor_api_weights, "",
          "model.safetensors of Qwen/Qwen3-0.6B, read when the engine creates "
          "a tensor_api_cpu or tensor_api_gpu executor");
ABSL_FLAG(std::vector<std::string>, tensor_api_prefill_lengths,
          std::vector<std::string>({"128", "1024"}),
          "Comma-separated lengths of the prefill signatures");
ABSL_FLAG(std::string, tensor_api_weights_quant, "none",
          "none|int8|int4: nearest-integer quantization of the decoder matmul "
          "weights while loading (int8 per channel, int4 blockwise-32)");
ABSL_FLAG(bool, tensor_api_lm_head_int8, false,
          "Add an int8 per-channel copy of the embedding table for the "
          "lm_head");
ABSL_FLAG(std::string, tensor_api_gpu_precision, "fp32",
          "fp32|fp16 GPU calculation precision");
ABSL_FLAG(std::string, tensor_api_gpu_buffer_storage, "buffer",
          "default|buffer|texture2d GPU tensor storage");
ABSL_FLAG(int, tensor_api_num_threads, 0,
          "CPU threads (0 = number_of_threads of the executor settings' CPU "
          "config)");
ABSL_FLAG(bool, tensor_api_litert_lm_xnnpack_flags, false,
          "CPU: use the XNNPACK flags of LiteRT-LM's CPU executor (the default "
          "flags plus dynamic fully-connected and the latest operators)");
ABSL_FLAG(bool, tensor_api_log_ids, true,
          "Log the token ids the executor receives");
ABSL_FLAG(bool, tensor_api_delegation_report, false,
          "Compile the model once more and log its delegation metrics");
ABSL_FLAG(std::string, tensor_api_attention, "raw",
          "raw|raw_vt|sdpa|sdpa_transposed|sdpa_transposed_param: the "
          "attention core as raw ops (raw_vt: with the value caches stored "
          "transposed), as the odml.scaled_dot_product_attention composite, "
          "or as the odml.sdpa_transposed composite (value caches transposed; "
          "_param adds the param tensor and the bool mask)");
ABSL_FLAG(std::string, tensor_api_norms, "raw",
          "raw|composite: every RMSNorm as raw ops or as the odml.rms_norm "
          "composite");
ABSL_FLAG(bool, tensor_api_gpu_share_constants, false,
          "GPU: share constant tensors (weights) between the signatures");
ABSL_FLAG(bool, tensor_api_gpu_program_cache, false,
          "GPU: serialize the compiled programs (and, with shared constants, "
          "the converted weights) to --cache_dir");
ABSL_FLAG(std::string, tensor_api_gpu_external_caches, "off",
          "off|stock|runner: the KV caches as external tensor patterns, with "
          "the external tensors mode off (as LiteRT-LM's GPU executor) or on "
          "(as LitertDynamicRunner's feedback loops)");
ABSL_FLAG(bool, tensor_api_gpu_litert_lm_options, false,
          "GPU: the other compile options of LiteRT-LM's GPU executor (without "
          "the KV cache patterns and the command buffer preparation)");
ABSL_FLAG(int, tensor_api_gpu_command_buffer_steps, -1,
          "GPU: steps of command buffer preparation in advance (-1 = not set)");
ABSL_FLAG(bool, tensor_api_xnnpack_weight_cache, false,
          "CPU: XNNPACK weight cache file in --cache_dir");
ABSL_FLAG(std::string, tensor_api_model_file, "off",
          "off|fresh|reuse: keep the serialized model in memory, or write it "
          "to --cache_dir and compile the mapped file (reuse: an existing "
          "file of the same key skips loading and building)");
ABSL_FLAG(bool, tensor_api_report, false,
          "Log the init phases, the memory footprint at each stage and the "
          "time of each part of a decode step");
ABSL_FLAG(int, tensor_api_report_hold_s, 0,
          "With --tensor_api_report: sleep this many seconds after logging "
          "the decode steps");

namespace {

namespace lm = ::litert::lm;
namespace qwen3 = ::litert::tensor::examples::qwen3;

absl::StatusOr<qwen3::TensorApiExecutorOptions> OptionsFromFlags(
    bool use_gpu) {
  qwen3::TensorApiExecutorOptions options;
  options.weights_path = absl::GetFlag(FLAGS_tensor_api_weights);
  options.use_gpu = use_gpu;
  options.gpu_precision = absl::GetFlag(FLAGS_tensor_api_gpu_precision);
  options.gpu_buffer_storage =
      absl::GetFlag(FLAGS_tensor_api_gpu_buffer_storage);
  options.prefill_lengths.clear();
  for (const std::string& s : absl::GetFlag(FLAGS_tensor_api_prefill_lengths)) {
    int length;
    if (!absl::SimpleAtoi(s, &length)) {
      return absl::InvalidArgumentError(
          absl::StrCat("--tensor_api_prefill_lengths: not a number: ", s));
    }
    options.prefill_lengths.push_back(length);
  }
  const std::string quant = absl::GetFlag(FLAGS_tensor_api_weights_quant);
  if (quant == "none") {
    options.weights_quant = qwen3::WeightQuantMode::kNone;
  } else if (quant == "int8") {
    options.weights_quant = qwen3::WeightQuantMode::kInt8;
  } else if (quant == "int4") {
    options.weights_quant = qwen3::WeightQuantMode::kInt4;
  } else {
    return absl::InvalidArgumentError(
        "--tensor_api_weights_quant must be none|int8|int4");
  }
  options.lm_head_int8 = absl::GetFlag(FLAGS_tensor_api_lm_head_int8);
  options.num_threads = absl::GetFlag(FLAGS_tensor_api_num_threads);
  options.litert_lm_xnnpack_flags =
      absl::GetFlag(FLAGS_tensor_api_litert_lm_xnnpack_flags);
  options.log_ids = absl::GetFlag(FLAGS_tensor_api_log_ids);
  options.delegation_report = absl::GetFlag(FLAGS_tensor_api_delegation_report);
  const std::string attention = absl::GetFlag(FLAGS_tensor_api_attention);
  if (attention != "raw" && attention != "raw_vt" && attention != "sdpa" &&
      attention != "sdpa_transposed" && attention != "sdpa_transposed_param") {
    return absl::InvalidArgumentError(
        "--tensor_api_attention must be "
        "raw|raw_vt|sdpa|sdpa_transposed|sdpa_transposed_param");
  }
  options.transposed_value_cache = attention == "raw_vt";
  options.sdpa_attention = attention == "sdpa";
  options.sdpa_transposed = attention == "sdpa_transposed" ||
                            attention == "sdpa_transposed_param";
  options.sdpa_param_tensor = attention == "sdpa_transposed_param";
  const std::string norms = absl::GetFlag(FLAGS_tensor_api_norms);
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError(
        "--tensor_api_norms must be raw|composite");
  }
  options.rms_norm_composite = norms == "composite";
  options.gpu_share_constants =
      absl::GetFlag(FLAGS_tensor_api_gpu_share_constants);
  options.gpu_program_cache = absl::GetFlag(FLAGS_tensor_api_gpu_program_cache);
  const std::string external =
      absl::GetFlag(FLAGS_tensor_api_gpu_external_caches);
  if (external == "off") {
    options.gpu_external_caches = qwen3::GpuExternalCaches::kOff;
  } else if (external == "stock") {
    options.gpu_external_caches = qwen3::GpuExternalCaches::kStock;
  } else if (external == "runner") {
    options.gpu_external_caches = qwen3::GpuExternalCaches::kRunner;
  } else {
    return absl::InvalidArgumentError(
        "--tensor_api_gpu_external_caches must be off|stock|runner");
  }
  options.gpu_litert_lm_options =
      absl::GetFlag(FLAGS_tensor_api_gpu_litert_lm_options);
  options.gpu_command_buffer_steps =
      absl::GetFlag(FLAGS_tensor_api_gpu_command_buffer_steps);
  options.xnnpack_weight_cache =
      absl::GetFlag(FLAGS_tensor_api_xnnpack_weight_cache);
  const std::string model_file = absl::GetFlag(FLAGS_tensor_api_model_file);
  if (model_file == "off") {
    options.model_file = qwen3::ModelFileMode::kMemory;
  } else if (model_file == "fresh") {
    options.model_file = qwen3::ModelFileMode::kFresh;
  } else if (model_file == "reuse") {
    options.model_file = qwen3::ModelFileMode::kReuse;
  } else {
    return absl::InvalidArgumentError(
        "--tensor_api_model_file must be off|fresh|reuse");
  }
  options.report = absl::GetFlag(FLAGS_tensor_api_report);
  options.report_hold_s = absl::GetFlag(FLAGS_tensor_api_report_hold_s);
  return options;
}

absl::StatusOr<std::unique_ptr<lm::LlmExecutor>> CreateExecutor(
    bool use_gpu, const lm::LlmExecutorSettings& settings,
    lm::ModelResources& model_resources) {
  ABSL_LOG(INFO) << "tensor_api: ExecutorBackendRegistry creator called for "
                 << lm::GetBackendString(settings.GetBackend())
                 << "; tokenizer, prompt handling, stop tokens and sampler "
                    "come from the engine's model resources";
  ABSL_ASSIGN_OR_RETURN(qwen3::TensorApiExecutorOptions options,
                        OptionsFromFlags(use_gpu));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<qwen3::TensorApiExecutor> executor,
                        qwen3::TensorApiExecutor::Create(settings, options));
  return std::unique_ptr<lm::LlmExecutor>(std::move(executor));
}

}  // namespace

LITERT_LM_REGISTER_EXECUTOR_BACKEND(
    "tensor_api_cpu",
    [](const lm::LlmExecutorSettings& settings,
       lm::ModelResources& model_resources) {
      return CreateExecutor(/*use_gpu=*/false, settings, model_resources);
    });

LITERT_LM_REGISTER_EXECUTOR_BACKEND(
    "tensor_api_gpu",
    [](const lm::LlmExecutorSettings& settings,
       lm::ModelResources& model_resources) {
      return CreateExecutor(/*use_gpu=*/true, settings, model_resources);
    });
