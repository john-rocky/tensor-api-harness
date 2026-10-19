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
