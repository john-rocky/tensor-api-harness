// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B host loop: builds the prefill and decode signatures for each
// prompt (prefill length = the prompt length), runs prefill, then greedy
// decode with the KV caches rebound from the previous call's outputs by
// handle. Token ids come from a file; the tokenizer stays outside.
//
//   bazel build //tensor/examples/qwen3-0.6b:qwen3_main
//   qwen3_main --weights=<model.safetensors> --ids_file=<ids.txt> \
//     --accelerator=cpu [--dump_blocks --dump_dir=<dir>] \
//     [--greedy=16 --out_ids=<ids.json>]
//
// ids file: one prompt per line, "<id> <token id> <token id> ...".
// --dump_blocks adds emb, block_00..block_27 and final_norm to the prefill
// signature's outputs and writes them, with logits_last.f32 (the last
// position's logits), to <dump_dir>/<id>/ as raw fp32.
// --greedy=N writes <out_ids> as {"prompts": [{"id", "ids"}]}: the prompt
// ids followed by N generated ids (1 from prefill, N - 1 decode steps).
// --dump_chain (with --greedy and --dump_dir) writes chain_00.f32 (the
// prefill logits at the last prompt position) and chain_01.f32 ..
// chain_<N-1>.f32 (each decode step's logits), each [vocab] fp32, to
// <dump_dir>/<id>/.
// --delegation_report compiles the model once more before the runner is
// created and prints CompiledModel::GetDelegationMetrics() (the runner keeps
// its CompiledModel private): total nodes, then per accelerator the delegated
// node count and the partition count.
// --rebind=copy reads each cache output to the host and writes it into the
// next call's input (a diagnostic); the default, handle, passes the output
// handle to SetInput, which binds a duplicate of the output TensorBuffer
// without copying its contents.
// Graph forms (defaults = the raw-op fp32 graph): --norms=composite emits
// every RMSNorm as odml.rms_norm; --attention=sdpa emits the attention core
// as odml.scaled_dot_product_attention over the GQA pre-fold;
// --weights_quant=int8|int4 quantizes the decoder matmul weights while
// loading (see qwen3_weights.h); --weights_quant_lm_head adds an int8 copy of
// the embedding table for the lm_head.
// --dump_final_norm_form=identity (with --dump_blocks) feeds the lm_head from
// Mul(final_norm, 1.0) so the exported final_norm is not its direct input.
// --time measures each runner.Run call with steady_clock (prefill, and each
// decode step). With --time_readback (the default) the interval ends after the
// logits output buffer is locked and one value is read, which waits for the
// accelerator; with --time_readback=false it ends when runner.Run returns.
// Setting inputs, the rebind and copying the logits are outside the timed
// interval. It prints each prompt's decode median and the median of those
// medians, and --timing_out writes every value as JSON.
// --prefill_len=N (N > 0) builds one model for all prompts with a prefill
// signature of length N: each prompt is padded on the right with id 0, the
// logits at the last prompt position are used and decode starts right after
// the prompt. The pad rows written into the caches stay behind the causal
// mask until decode overwrites them. --serialize_only (with --prefill_len)
// builds and saves the model and exits; --load_tflite=<path> (with the
// --prefill_len the file was saved with) runs a saved model without building
// a graph or reading weights.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"
#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights, "",
          "model.safetensors path (empty = synthetic weights under the real "
          "tensor names)");
ABSL_FLAG(std::string, ids_file, "",
          "Prompt ids, one prompt per line: <id> <token id> ...");
ABSL_FLAG(int, max_seq, 64, "Fixed KV cache length");
ABSL_FLAG(int, greedy, 0,
          "Generated tokens per prompt (0 = prefill only)");
ABSL_FLAG(std::string, out_ids, "", "Where to write the ids JSON (--greedy)");
ABSL_FLAG(bool, dump_blocks, false,
          "Add the per-block outputs to the prefill signature and dump them");
ABSL_FLAG(std::string, dump_dir, "", "Dump directory (--dump_blocks)");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "fp32",
          "fp16|fp32 GPU calculation precision (the delegate default is fp16)");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d GPU tensor storage");
ABSL_FLAG(std::string, tflite_path, "/tmp/qwen3_0_6b.tflite",
          "Where to write the serialized model");
ABSL_FLAG(bool, dump_chain, false,
          "Dump the prefill last-position logits and every decode step's "
          "logits as chain_NN.f32 (needs --greedy and --dump_dir)");
ABSL_FLAG(bool, delegation_report, true,
          "Print the delegation metrics of a separately compiled model");
ABSL_FLAG(std::string, log_level, "",
          "info|debug|verbose: the LiteRT logger's minimum severity (empty = "
          "the runtime default)");
ABSL_FLAG(std::string, rebind, "handle",
          "handle|copy: how cache outputs become the next call's inputs");
ABSL_FLAG(std::string, norms, "raw",
          "raw|composite: RMSNorm as raw ops or as the odml.rms_norm composite");
ABSL_FLAG(std::string, attention, "raw",
          "raw|sdpa: the attention core as raw ops or as the "
          "odml.scaled_dot_product_attention composite (GQA pre-folded)");
ABSL_FLAG(std::string, weights_quant, "none",
          "none|int8|int4: nearest-integer quantization of the 196 decoder "
          "matmul weights while loading (int8 per channel, int4 blockwise-32)");
ABSL_FLAG(bool, weights_quant_lm_head, false,
          "Add an int8 per-channel copy of the embedding table for the lm_head "
          "(the lookup keeps the fp32 table)");
ABSL_FLAG(bool, gpu_allow_src_quantized_fc, false,
          "GPU: quantize the input of large FullyConnected ops to 8 bit "
          "(EnableAllowSrcQuantizedFcConvOps, with constant tensor sharing)");
ABSL_FLAG(std::string, dump_final_norm_form, "direct",
          "direct|identity (--dump_blocks only): identity feeds the lm_head "
          "from Mul(final_norm, 1.0), so the exported final_norm is not the "
          "FullyConnected input");
ABSL_FLAG(bool, time, false,
          "Time each runner.Run call (prefill and every decode step)");
ABSL_FLAG(std::string, timing_out, "",
          "Where to write the timing JSON (--time)");
ABSL_FLAG(bool, time_readback, true,
          "End each timed interval after locking the logits output buffer and "
          "reading one value, so the time includes the accelerator finishing "
          "(false: runner.Run only, which on some GPU backends returns once "
          "the work is enqueued)");
ABSL_FLAG(bool, gpu_wait_hint, false,
          "Set the GPU option HintWaitingForCompletion (wait for the enqueued "
          "commands after each invoke)");
ABSL_FLAG(std::string, form, "", "Form name written into the timing JSON");
ABSL_FLAG(int, prefill_len, 0,
          "Fixed prefill length for every prompt, prompts padded on the right "
          "with id 0 (0 = one model per prompt at the prompt length)");
ABSL_FLAG(bool, serialize_only, false,
          "Build and save the model to --tflite_path, then exit (needs "
          "--prefill_len)");
ABSL_FLAG(std::string, load_tflite, "",
          "Run this saved model instead of building one (needs the "
          "--prefill_len it was saved with; no weights are read)");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;
namespace qwen3 = ::litert::tensor::examples::qwen3;

constexpr float kMaskFloor = -30000.0f;  // fp16-safe
constexpr int32_t kPadId = 0;            // --prefill_len padding

// The GPU accelerator this binary loads, for the form label.
#if defined(__ANDROID__)
constexpr char kGpuBackend[] = "opencl";
#else
constexpr char kGpuBackend[] = "metal";
#endif

struct Prompt {
  std::string id;
  std::vector<int32_t> ids;
};

absl::StatusOr<std::vector<Prompt>> ReadIdsFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) return absl::NotFoundError("ids_file: " + path);
  std::vector<Prompt> prompts;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    Prompt p;
    if (!(ss >> p.id)) continue;
    int32_t v;
    while (ss >> v) p.ids.push_back(v);
    if (p.ids.empty()) {
      return absl::InvalidArgumentError("no ids for prompt " + p.id);
    }
    prompts.push_back(std::move(p));
  }
  if (prompts.empty()) return absl::InvalidArgumentError("empty ids_file");
  return prompts;
}

std::string ShapeStr(const std::vector<int>& s) {
  return absl::StrCat("[", absl::StrJoin(s, ", "), "]");
}

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

absl::Status ReadFloats(LitertDynamicRunner& runner, const std::string& sig,
                        const std::string& name, std::vector<float>& out) {
  auto t = runner.GetOutput(sig, name);
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  out.assign(data, data + lock.size() / sizeof(float));
  return absl::OkStatus();
}

// Locks the output buffer and reads its first value: the lock waits until the
// accelerator has written the buffer (the end of a --time_readback interval).
absl::Status SyncOutput(LitertDynamicRunner& runner, const std::string& sig,
                        const std::string& name) {
  auto t = runner.GetOutput(sig, name);
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  if (lock.size() < sizeof(float)) {
    return absl::InternalError("empty output buffer: " + name);
  }
  volatile float first = *reinterpret_cast<const float*>(lock.data());
  (void)first;
  return absl::OkStatus();
}

absl::Status WriteRaw(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return absl::InternalError("cannot write " + path);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
  return absl::OkStatus();
}

int32_t ArgMaxRow(const float* row, int n) {
  return static_cast<int32_t>(std::max_element(row, row + n) - row);
}

absl::StatusOr<::litert::Options> MakeOptions() {
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const std::string accel = absl::GetFlag(FLAGS_accelerator);
  if (accel != "cpu" && accel != "gpu") {
    return absl::InvalidArgumentError("--accelerator must be cpu|gpu");
  }
  const bool use_gpu = accel == "gpu";
  options->SetHardwareAccelerators(use_gpu ? ::litert::HwAccelerators::kGpu
                                           : ::litert::HwAccelerators::kCpu);
  if (use_gpu) {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      const std::string precision = absl::GetFlag(FLAGS_gpu_precision);
      if (precision == "fp32") {
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      } else if (precision != "fp16") {
        return absl::InvalidArgumentError("--gpu_precision must be fp16|fp32");
      }
      const std::string storage = absl::GetFlag(FLAGS_gpu_buffer_storage);
      if (storage == "buffer") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kBuffer);
      } else if (storage == "texture2d") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kTexture2D);
      } else if (storage != "default") {
        return absl::InvalidArgumentError(
            "--gpu_buffer_storage must be default|buffer|texture2d");
      }
      const bool src_quantized =
          absl::GetFlag(FLAGS_gpu_allow_src_quantized_fc);
      if (src_quantized) {
        gpu_options->EnableConstantTensorSharing(true);
        gpu_options->EnableAllowSrcQuantizedFcConvOps(true);
      }
      const bool wait_hint = absl::GetFlag(FLAGS_gpu_wait_hint);
      if (wait_hint) gpu_options->HintWaitingForCompletion(true);
      std::cout << "gpu precision: " << precision << ", storage: " << storage
                << ", allow_src_quantized_fc: "
                << (src_quantized ? "true" : "false")
                << ", wait hint: " << (wait_hint ? "true" : "false")
                << std::endl;
    }
  }
  return std::move(*options);
}

std::string HwName(LiteRtHwAcceleratorSet hw) {
  std::vector<std::string> names;
  if (hw & kLiteRtHwAcceleratorCpu) names.push_back("cpu");
  if (hw & kLiteRtHwAcceleratorGpu) names.push_back("gpu");
  if (hw & kLiteRtHwAcceleratorNpu) names.push_back("npu");
  if (names.empty()) names.push_back("none");
  return absl::StrCat(absl::StrJoin(names, "+"), "(", hw, ")");
}

// Compiles the serialized model with the run's options and prints the
// delegation metrics. A compile failure is printed and returned.
absl::Status PrintDelegationReport(::litert::Environment& env,
                                   const std::string& tflite_path,
                                   const std::string& id) {
  auto options_or = MakeOptions();
  if (!options_or.ok()) return options_or.status();
  auto compiled = ::litert::CompiledModel::Create(env, tflite_path, *options_or);
  if (!compiled) {
    std::cout << id << " delegation: CompiledModel::Create failed: "
              << compiled.Error().Message() << std::endl;
    return absl::InternalError(absl::StrCat(
        "delegation report compile failed: ", compiled.Error().Message()));
  }
  auto metrics = compiled->GetDelegationMetrics();
  if (!metrics) {
    std::cout << id << " delegation: metrics unavailable: "
              << metrics.Error().Message() << std::endl;
    return absl::OkStatus();
  }
  std::cout << id << " delegation: total_nodes " << metrics->total_node_count
            << std::endl;
  for (const auto& acc : metrics->accelerators) {
    std::cout << id << " delegation: " << acc.accelerator_name
              << " hw=" << HwName(acc.hardware_type)
              << " delegated_nodes=" << acc.delegated_node_count
              << " partitions=" << acc.partition_count << std::endl;
  }
  auto full = compiled->IsFullyAccelerated();
  auto non_cpu_full = compiled->IsNonCpuFullyAccelerated();
  std::cout << id << " delegation: fully_accelerated "
            << (full ? (*full ? "true" : "false") : "unknown")
            << ", non_cpu_fully_accelerated "
            << (non_cpu_full ? (*non_cpu_full ? "true" : "false") : "unknown")
            << std::endl;
  return absl::OkStatus();
}

// Wall time of each runner.Run call (steady_clock). Setting inputs, the cache
// rebind and reading the logits happen outside the timed interval.
struct PromptTiming {
  double prefill_ms = 0.0;
  std::vector<double> decode_ms;
};

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Builds the prefill (length config.prefill_len) and decode signatures and
// saves them to `tflite_path`.
absl::Status BuildAndSave(const qwen3::Qwen3Config& config,
                          const qwen3::WeightMap& weights,
                          const std::string& tflite_path,
                          const std::string& label) {
  qwen3::Qwen3Inputs prefill_in = qwen3::MakeQwen3Inputs(config, false);
  qwen3::Qwen3Outputs prefill_out =
      qwen3::BuildQwen3(config, prefill_in, weights, false);
  qwen3::Qwen3Config decode_config = config;
  decode_config.dump_blocks = false;
  qwen3::Qwen3Inputs decode_in = qwen3::MakeQwen3Inputs(decode_config, true);
  qwen3::Qwen3Outputs decode_out =
      qwen3::BuildQwen3(decode_config, decode_in, weights, true);

  ModelFactory factory;
  {
    std::vector<TensorHandle> ins, outs;
    for (auto& t : prefill_in.AsList()) ins.push_back(t);
    for (auto& t : prefill_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "prefill");
    if (!status.ok()) return status;
  }
  {
    std::vector<TensorHandle> ins, outs;
    for (auto& t : decode_in.AsList()) ins.push_back(t);
    for (auto& t : decode_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "decode");
    if (!status.ok()) return status;
  }
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::error_code ec;
  auto tflite_bytes = std::filesystem::file_size(tflite_path, ec);
  std::cout << label << ": serialized " << tflite_path << " ("
            << (ec ? 0 : tflite_bytes) << " bytes), prefill_len "
            << config.prefill_len << ", max_seq " << config.max_seq
            << std::endl;
  return absl::OkStatus();
}

// Prints the delegation report (when asked) and creates the runner.
absl::StatusOr<LitertDynamicRunner> MakeRunner(::litert::Environment& env,
                                               const std::string& tflite_path,
                                               const std::string& label) {
  if (absl::GetFlag(FLAGS_delegation_report)) {
    auto report_status = PrintDelegationReport(env, tflite_path, label);
    if (!report_status.ok()) return report_status;
  }
  auto options_or = MakeOptions();
  if (!options_or.ok()) return options_or.status();
  auto runner_or = LitertDynamicRunner::Create(env, tflite_path, *options_or);
  if (!runner_or.ok()) return runner_or.status();
  std::cout << label << ": runner created" << std::endl;
  return runner_or;
}

struct OutputSpec {
  std::string name;
  std::vector<int> shape;
};

// The dumped tensors of the prefill signature (--dump_blocks), in output
// order; the names BuildQwen3 gives them.
std::vector<std::string> DumpNames(const qwen3::Qwen3Config& config) {
  std::vector<std::string> names = {"emb"};
  for (int i = 0; i < config.n_layers; ++i) {
    names.push_back(absl::StrCat("block_", i < 10 ? "0" : "", i));
  }
  names.push_back("final_norm");
  return names;
}

// The outputs whose element counts are checked after a signature's first
// call: the logits, the first layer's caches and (prefill with
// --dump_blocks) the dumps, with the shapes the graph gives them.
std::vector<OutputSpec> CheckedOutputs(const qwen3::Qwen3Config& config,
                                       int seq, bool is_decode) {
  const std::vector<int> cache = {1, config.n_kv_groups, config.max_seq,
                                  config.head_dim};
  std::vector<OutputSpec> specs = {
      {"logits", {1, seq, config.vocab_size}},
      {"key_cache_0_new", cache},
      {"value_cache_0_new", cache}};
  if (config.dump_blocks && !is_decode) {
    for (const std::string& name : DumpNames(config)) {
      specs.push_back({name, {1, seq, config.emb_dim}});
    }
  }
  return specs;
}

// Runs prefill (+ dumps) and the greedy loop for one prompt on a runner whose
// prefill signature has length config.prefill_len (at least the prompt
// length; the prompt is padded on the right with kPadId). Appends the
// generated ids to `generated`.
absl::Status RunPrompt(LitertDynamicRunner& runner,
                       const qwen3::Qwen3Config& config, const Prompt& prompt,
                       int greedy, std::vector<int32_t>& generated,
                       PromptTiming& timing) {
  std::error_code ec;
  const int len = static_cast<int>(prompt.ids.size());
  const int seq = config.prefill_len;
  std::vector<int32_t> prefill_ids = prompt.ids;
  prefill_ids.resize(seq, kPadId);

  const int hd = config.head_dim;
  const int vocab = config.vocab_size;
  std::vector<float> cos_v, sin_v;
  auto set_step = [&](const std::string& sig, const std::vector<int32_t>& ids,
                      int pos) -> absl::Status {
    const int n = static_cast<int>(ids.size());
    auto st = runner.SetInput(
        sig, "input_ids",
        Create("input_ids", Type::kI32, {n}, std::vector<int32_t>(ids)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "cache_position",
                         Create("cache_position", Type::kI32, {4},
                                std::vector<int32_t>{0, 0, pos, 0}));
    if (!st.ok()) return st;
    BuildRopeCosSin(pos, n, hd, config.rope_base, cos_v, sin_v);
    st = runner.SetInput(sig, "rope_cos",
                         Create("rope_cos", Type::kFP32, {1, 1, n, hd},
                                std::vector<float>(cos_v)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "rope_sin",
                         Create("rope_sin", Type::kFP32, {1, 1, n, hd},
                                std::vector<float>(sin_v)));
    if (!st.ok()) return st;
    return runner.SetInput(sig, "attention_mask",
                           Create("attention_mask", Type::kFP32,
                                  {1, 1, n, config.max_seq},
                                  BuildMask(n, pos, config.max_seq)));
  };
  auto print_outputs = [&](const std::string& sig,
                           const std::vector<OutputSpec>& outs)
      -> absl::Status {
    // The graph shape next to the element count the runtime returned.
    for (const OutputSpec& out : outs) {
      std::vector<float> v;
      auto st = ReadFloats(runner, sig, out.name, v);
      if (!st.ok()) return st;
      size_t expect = 1;
      for (int d : out.shape) expect *= static_cast<size_t>(d);
      std::cout << prompt.id << " " << sig << " output " << out.name << " "
                << ShapeStr(out.shape) << " elements " << v.size()
                << (v.size() == expect ? "" : " SHAPE MISMATCH") << std::endl;
      if (v.size() != expect) {
        return absl::InternalError("output size mismatch: " + out.name);
      }
    }
    return absl::OkStatus();
  };

  // --- Prefill ---
  auto st = set_step("prefill", prefill_ids, 0);
  if (!st.ok()) return st;
  const size_t cache_n =
      static_cast<size_t>(config.n_kv_groups) * config.max_seq * hd;
  for (int i = 0; i < config.n_layers; ++i) {
    for (const char* kind : {"key_cache_", "value_cache_"}) {
      std::string name = absl::StrCat(kind, i);
      st = runner.SetInput(
          "prefill", name,
          Create(name.c_str(), Type::kFP32,
                 {1, config.n_kv_groups, config.max_seq, hd},
                 std::vector<float>(cache_n, 0.0f)));
      if (!st.ok()) return st;
    }
  }
  const bool readback = absl::GetFlag(FLAGS_time_readback);
  auto t0 = std::chrono::steady_clock::now();
  st = runner.Run("prefill");
  if (st.ok() && readback) st = SyncOutput(runner, "prefill", "logits");
  auto t1 = std::chrono::steady_clock::now();
  if (!st.ok()) return st;
  timing.prefill_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  st = print_outputs("prefill", CheckedOutputs(config, seq, false));
  if (!st.ok()) return st;

  std::vector<float> logits;
  st = ReadFloats(runner, "prefill", "logits", logits);
  if (!st.ok()) return st;
  const float* last = logits.data() + static_cast<size_t>(len - 1) * vocab;

  if (config.dump_blocks) {
    const std::string dir =
        absl::StrCat(absl::GetFlag(FLAGS_dump_dir), "/", prompt.id);
    std::filesystem::create_directories(dir, ec);
    const std::vector<std::string> names = DumpNames(config);
    for (const std::string& name : names) {
      std::vector<float> v;
      st = ReadFloats(runner, "prefill", name, v);
      if (!st.ok()) return st;
      // [1, seq, emb_dim]: keep the prompt rows, drop the pad rows.
      v.resize(static_cast<size_t>(len) * config.emb_dim);
      st = WriteRaw(absl::StrCat(dir, "/", name, ".f32"), v);
      if (!st.ok()) return st;
    }
    st = WriteRaw(dir + "/logits_last.f32", std::vector<float>(last, last + vocab));
    if (!st.ok()) return st;
    std::ofstream am(dir + "/argmax.json");
    am << "{\"argmax\": [";
    for (int s = 0; s < len; ++s) {
      am << (s ? ", " : "")
         << ArgMaxRow(logits.data() + static_cast<size_t>(s) * vocab, vocab);
    }
    am << "]}\n";
    std::cout << prompt.id << ": dumped " << names.size() + 1
              << " tensors to " << dir << std::endl;
  }

  if (greedy <= 0) return absl::OkStatus();

  // --- Greedy decode: token 1 from prefill, then greedy - 1 decode steps ---
  const bool dump_chain = absl::GetFlag(FLAGS_dump_chain);
  const std::string chain_dir =
      absl::StrCat(absl::GetFlag(FLAGS_dump_dir), "/", prompt.id);
  if (dump_chain) {
    std::filesystem::create_directories(chain_dir, ec);
    st = WriteRaw(chain_dir + "/chain_00.f32",
                  std::vector<float>(last, last + vocab));
    if (!st.ok()) return st;
  }
  const bool rebind_copy = absl::GetFlag(FLAGS_rebind) == "copy";
  // Handle form: GetOutput returns a TensorHandle over a duplicate of the
  // output TensorBuffer; SetInput sees a LitertBuffer and binds a duplicate
  // of it with SetInputBuffer, so no cache bytes are copied.
  auto rebind = [&](const std::string& from_sig) -> absl::Status {
    for (int i = 0; i < config.n_layers; ++i) {
      for (const char* kind : {"key_cache_", "value_cache_"}) {
        const std::string in_name = absl::StrCat(kind, i);
        const std::string out_name = absl::StrCat(in_name, "_new");
        if (rebind_copy) {
          std::vector<float> v;
          auto s = ReadFloats(runner, from_sig, out_name, v);
          if (!s.ok()) return s;
          s = runner.SetInput(
              "decode", in_name,
              Create(in_name.c_str(), Type::kFP32,
                     {1, config.n_kv_groups, config.max_seq, hd},
                     std::move(v)));
          if (!s.ok()) return s;
          continue;
        }
        auto t = runner.GetOutput(from_sig, out_name);
        if (!t.ok()) return t.status();
        auto s = runner.SetInput("decode", in_name, *t);
        if (!s.ok()) return s;
      }
    }
    return absl::OkStatus();
  };
  int32_t tok = ArgMaxRow(last, vocab);
  generated.push_back(tok);
  st = rebind("prefill");
  if (!st.ok()) return st;
  for (int step = 1; step < greedy; ++step) {
    const int pos = len + step - 1;
    st = set_step("decode", {tok}, pos);
    if (!st.ok()) return st;
    t0 = std::chrono::steady_clock::now();
    st = runner.Run("decode");
    if (st.ok() && readback) st = SyncOutput(runner, "decode", "logits");
    t1 = std::chrono::steady_clock::now();
    if (!st.ok()) return st;
    timing.decode_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (step == 1) {
      st = print_outputs("decode", CheckedOutputs(config, 1, true));
      if (!st.ok()) return st;
    }
    st = ReadFloats(runner, "decode", "logits", logits);
    if (!st.ok()) return st;
    if (dump_chain) {
      char name[32];
      std::snprintf(name, sizeof(name), "/chain_%02d.f32", step);
      st = WriteRaw(chain_dir + name, logits);
      if (!st.ok()) return st;
    }
    tok = ArgMaxRow(logits.data(), vocab);
    generated.push_back(tok);
    st = rebind("decode");
    if (!st.ok()) return st;
  }
  std::cout << prompt.id << ": generated " << absl::StrJoin(generated, " ")
            << std::endl;
  return absl::OkStatus();
}

absl::Status Run() {
  qwen3::Qwen3Config config;
  config.max_seq = absl::GetFlag(FLAGS_max_seq);
  config.dump_blocks = absl::GetFlag(FLAGS_dump_blocks);
  const int greedy = absl::GetFlag(FLAGS_greedy);
  if (config.dump_blocks && absl::GetFlag(FLAGS_dump_dir).empty()) {
    return absl::InvalidArgumentError("--dump_blocks needs --dump_dir");
  }
  if (greedy > 0 && absl::GetFlag(FLAGS_out_ids).empty()) {
    return absl::InvalidArgumentError("--greedy needs --out_ids");
  }
  if (absl::GetFlag(FLAGS_dump_chain) &&
      (greedy <= 0 || absl::GetFlag(FLAGS_dump_dir).empty())) {
    return absl::InvalidArgumentError(
        "--dump_chain needs --greedy and --dump_dir");
  }
  const std::string rebind = absl::GetFlag(FLAGS_rebind);
  if (rebind != "handle" && rebind != "copy") {
    return absl::InvalidArgumentError("--rebind must be handle|copy");
  }
  const std::string norms = absl::GetFlag(FLAGS_norms);
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  config.rms_norm_composite = norms == "composite";
  const std::string attention = absl::GetFlag(FLAGS_attention);
  if (attention != "raw" && attention != "sdpa") {
    return absl::InvalidArgumentError("--attention must be raw|sdpa");
  }
  config.sdpa_attention = attention == "sdpa";
  const std::string quant = absl::GetFlag(FLAGS_weights_quant);
  qwen3::WeightQuantMode quant_mode;
  if (quant == "none") {
    quant_mode = qwen3::WeightQuantMode::kNone;
  } else if (quant == "int8") {
    quant_mode = qwen3::WeightQuantMode::kInt8;
  } else if (quant == "int4") {
    quant_mode = qwen3::WeightQuantMode::kInt4;
  } else {
    return absl::InvalidArgumentError("--weights_quant must be none|int8|int4");
  }
  config.separate_lm_head = absl::GetFlag(FLAGS_weights_quant_lm_head);
  const std::string final_norm_form = absl::GetFlag(FLAGS_dump_final_norm_form);
  if (final_norm_form != "direct" && final_norm_form != "identity") {
    return absl::InvalidArgumentError(
        "--dump_final_norm_form must be direct|identity");
  }
  config.dump_final_norm_identity = final_norm_form == "identity";
  const bool time_runs = absl::GetFlag(FLAGS_time);
  if (!absl::GetFlag(FLAGS_timing_out).empty() && !time_runs) {
    return absl::InvalidArgumentError("--timing_out needs --time");
  }
  const int fixed_len = absl::GetFlag(FLAGS_prefill_len);
  const bool serialize_only = absl::GetFlag(FLAGS_serialize_only);
  const std::string load_path = absl::GetFlag(FLAGS_load_tflite);
  if (fixed_len < 0 || fixed_len > config.max_seq) {
    return absl::InvalidArgumentError("--prefill_len must be 0..max_seq");
  }
  if (serialize_only && (fixed_len == 0 || !load_path.empty())) {
    return absl::InvalidArgumentError(
        "--serialize_only needs --prefill_len and no --load_tflite");
  }
  if (!load_path.empty() &&
      (fixed_len == 0 || !absl::GetFlag(FLAGS_weights).empty())) {
    return absl::InvalidArgumentError(
        "--load_tflite needs the --prefill_len the file was saved with and "
        "no --weights");
  }
  const bool use_gpu = absl::GetFlag(FLAGS_accelerator) == "gpu";
  const std::string backend = use_gpu ? kGpuBackend : "cpu";
  const std::string precision =
      quant != "none" ? quant
                      : (use_gpu ? absl::GetFlag(FLAGS_gpu_precision) : "fp32");
  std::string form = absl::GetFlag(FLAGS_form);
  if (form.empty()) {
    form = absl::StrCat(backend, " ", precision, " attention=", attention,
                        " norms=", norms);
  }
  std::cout << "form: " << form << " (attention " << attention << ", norms "
            << norms << ", weights_quant " << quant << ", lm_head "
            << (config.separate_lm_head ? "int8 copy" : "tied fp32 table")
            << (config.dump_blocks
                    ? absl::StrCat(", final_norm dump ", final_norm_form)
                    : std::string())
            << ")" << std::endl;
  std::vector<::litert::EnvironmentOptions::Option> env_options;
  const std::string log_level = absl::GetFlag(FLAGS_log_level);
  if (!log_level.empty()) {
    int64_t severity;
    if (log_level == "info") {
      severity = kLiteRtLogSeverityInfo;
    } else if (log_level == "debug") {
      severity = kLiteRtLogSeverityDebug;
    } else if (log_level == "verbose") {
      severity = kLiteRtLogSeverityVerbose;
    } else {
      return absl::InvalidArgumentError(
          "--log_level must be info|debug|verbose");
    }
    env_options.emplace_back(
        ::litert::EnvironmentOptions::Tag::kMinLoggerSeverity,
        ::litert::LiteRtVariant(severity));
    std::cout << "log level: " << log_level << " (" << severity << ")"
              << std::endl;
  }
  std::cout << "rebind: " << rebind << std::endl;
  std::vector<Prompt> prompts;
  if (!serialize_only) {
    auto prompts_or = ReadIdsFile(absl::GetFlag(FLAGS_ids_file));
    if (!prompts_or.ok()) return prompts_or.status();
    prompts = std::move(*prompts_or);
  }
  for (const Prompt& prompt : prompts) {
    const int len = static_cast<int>(prompt.ids.size());
    if (len + greedy > config.max_seq) {
      return absl::InvalidArgumentError(absl::StrCat(
          prompt.id, ": prompt ", len, " + greedy ", greedy, " > max_seq ",
          config.max_seq));
    }
    if (fixed_len > 0 && len > fixed_len) {
      return absl::InvalidArgumentError(absl::StrCat(
          prompt.id, ": prompt ", len, " > --prefill_len ", fixed_len));
    }
  }

  qwen3::WeightMap weights;
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (!load_path.empty()) {
    std::cout << "model: " << load_path << " (prefill_len " << fixed_len
              << "; no graph built, no weights read)" << std::endl;
  } else if (weights_path.empty()) {
    if (quant_mode != qwen3::WeightQuantMode::kNone ||
        config.separate_lm_head) {
      return absl::InvalidArgumentError(
          "--weights_quant and --weights_quant_lm_head need --weights");
    }
    weights = qwen3::MakeSyntheticWeights(config, /*seed=*/42);
    std::cout << "weights: synthetic (seed 42), " << weights.size()
              << " tensors" << std::endl;
  } else {
    qwen3::QuantStats stats;
    auto weights_or = qwen3::LoadCheckpointWeights(
        config, weights_path, quant_mode, config.separate_lm_head, &stats);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    std::cout << "weights: " << weights_path << " (" << weights.size()
              << " tensors, bf16->fp32, lm_head "
              << (config.separate_lm_head
                      ? "an int8 copy of model.embed_tokens.weight"
                      : "tied to model.embed_tokens.weight")
              << ")" << std::endl;
    if (quant_mode != qwen3::WeightQuantMode::kNone) {
      std::cout << "weights_quant " << quant << ": " << stats.quantized_tensors
                << " tensors, fp32 " << stats.fp32_bytes << " bytes -> "
                << stats.quantized_bytes << " bytes (codes + scales)"
                << std::endl;
    }
    if (config.separate_lm_head) {
      std::cout << "lm_head int8 copy: " << stats.lm_head_bytes
                << " bytes (codes + scales)" << std::endl;
    }
  }

  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);
  std::string model_path = load_path;
  if (fixed_len > 0) {
    config.prefill_len = fixed_len;
    if (model_path.empty()) {
      auto st = BuildAndSave(config, weights, tflite_path, "all");
      if (!st.ok()) return st;
      weights.clear();
      model_path = tflite_path;
    }
    if (serialize_only) return absl::OkStatus();
  }

  auto env = ::litert::Environment::Create(
      ::litert::EnvironmentOptions(env_options));
  if (!env) return absl::InternalError("Environment::Create failed");
  // --prefill_len: one runner for every prompt.
  std::optional<LitertDynamicRunner> shared_runner;
  if (fixed_len > 0) {
    auto runner_or = MakeRunner(*env, model_path, "all");
    if (!runner_or.ok()) return runner_or.status();
    shared_runner.emplace(std::move(*runner_or));
  }

  std::ostringstream json;
  json << "{\"prompts\": [";
  bool first = true;
  std::ostringstream timing_json;
  timing_json << "{\"form\": \"" << form << "\", \"backend\": \"" << backend
              << "\", \"precision\": \"" << precision
              << "\", \"attention\": \"" << attention << "\", \"norms\": \""
              << norms << "\", \"weights_quant\": \"" << quant
              << "\", \"lm_head\": \""
              << (config.separate_lm_head ? "int8" : "fp32")
              << "\", \"timing\": \""
              << (absl::GetFlag(FLAGS_time_readback) ? "run+readback" : "run")
              << "\", \"gpu_wait_hint\": "
              << (absl::GetFlag(FLAGS_gpu_wait_hint) ? "true" : "false")
              << ", \"prompts\": [";
  std::vector<double> prompt_medians;
  for (const Prompt& prompt : prompts) {
    std::vector<int32_t> generated;
    PromptTiming timing;
    absl::Status st;
    if (shared_runner.has_value()) {
      st = RunPrompt(*shared_runner, config, prompt, greedy, generated,
                     timing);
    } else {
      // One model per prompt, prefill length = the prompt length.
      qwen3::Qwen3Config prompt_config = config;
      prompt_config.prefill_len = static_cast<int>(prompt.ids.size());
      st = BuildAndSave(prompt_config, weights, tflite_path, prompt.id);
      if (!st.ok()) return st;
      auto runner_or = MakeRunner(*env, tflite_path, prompt.id);
      if (!runner_or.ok()) return runner_or.status();
      st = RunPrompt(*runner_or, prompt_config, prompt, greedy, generated,
                     timing);
    }
    if (!st.ok()) return st;
    std::vector<int32_t> all = prompt.ids;
    all.insert(all.end(), generated.begin(), generated.end());
    json << (first ? "" : ", ") << "{\"id\": \"" << prompt.id
         << "\", \"ids\": [" << absl::StrJoin(all, ", ") << "]}";
    if (time_runs) {
      const double median = Median(timing.decode_ms);
      prompt_medians.push_back(median);
      std::vector<std::string> decode;
      for (double ms : timing.decode_ms) decode.push_back(absl::StrCat(ms));
      timing_json << (first ? "" : ", ") << "{\"id\": \"" << prompt.id
                  << "\", \"S\": " << prompt.ids.size()
                  << ", \"prefill_len\": "
                  << (fixed_len > 0 ? fixed_len
                                    : static_cast<int>(prompt.ids.size()))
                  << ", \"prefill_ms\": " << timing.prefill_ms
                  << ", \"decode_ms\": [" << absl::StrJoin(decode, ", ")
                  << "]}";
      std::cout << prompt.id << " time: prefill (S " << prompt.ids.size()
                << ") " << timing.prefill_ms << " ms, decode median "
                << median << " ms/token over " << timing.decode_ms.size()
                << " steps" << std::endl;
    }
    first = false;
  }
  json << "]}\n";
  timing_json << "]}\n";
  if (greedy > 0) {
    std::ofstream out(absl::GetFlag(FLAGS_out_ids));
    if (!out) return absl::InternalError("cannot write --out_ids");
    out << json.str();
    std::cout << "wrote " << absl::GetFlag(FLAGS_out_ids) << std::endl;
  }
  if (time_runs) {
    std::cout << "time: " << form << " decode " << Median(prompt_medians)
              << " ms/token (median of the per-prompt medians)" << std::endl;
    const std::string timing_out = absl::GetFlag(FLAGS_timing_out);
    if (!timing_out.empty()) {
      std::ofstream out(timing_out);
      if (!out) return absl::InternalError("cannot write --timing_out");
      out << timing_json.str();
      std::cout << "wrote " << timing_out << std::endl;
    }
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = Run();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  std::cout << "DONE qwen3_main" << std::endl;
  return 0;
}
