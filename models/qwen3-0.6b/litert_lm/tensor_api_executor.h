// SPDX-License-Identifier: Apache-2.0
// The Qwen3-0.6B Tensor API graph (../graph) as a LiteRT-LM executor.
//
// When the engine creates it, the executor reads model.safetensors, builds
// the prefill and decode signatures on the Tensor API in memory (one prefill
// signature per prefill length, without the lm_head; one decode signature
// with the logits) and compiles them with LiteRT on the CPU (XNNPACK) or the
// GPU. LiteRT-LM's session keeps the tokenizer, the prompt handling, the stop
// tokens and the sampler: the executor takes token ids and returns logits.
//
// Token bookkeeping follows LiteRT-LM's compiled-model executor: Prefill runs
// every given token but the last one, which stays pending; DecodeLogits runs
// one token at its position (the pending one, replaced by the id the caller
// passes) and returns the logits for the next position. The KV caches are
// signature inputs and outputs of size max_num_tokens; every call binds the
// previous call's cache outputs as its cache inputs without copying them.

#ifndef TENSOR_API_QWEN3_0_6B_TENSOR_API_EXECUTOR_H_
#define TENSOR_API_QWEN3_0_6B_TENSOR_API_EXECUTOR_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"  // from @litert
#include "tensor/tensor.h"  // from @litert

namespace litert::tensor::examples::qwen3 {

// Where the serialized model lives while it is compiled and run.
enum class ModelFileMode {
  // The executor's own buffer (the compiled model reads it in place).
  kMemory,
  // A file in the cache directory, written at every start; the graph, its
  // weight tensors and the buffer are released before the compile, which
  // maps the file.
  kFresh,
  // As kFresh, but an existing file of the same key is compiled without
  // loading the checkpoint or building the graph.
  kReuse,
};

// GPU: how the KV cache tensors are presented to the GPU accelerator.
enum class GpuExternalCaches {
  kOff,
  // As LiteRT-LM's GPU executor: no external tensors mode, with the caches
  // as external tensor patterns.
  kStock,
  // As LitertDynamicRunner's feedback loops: external tensors mode, with the
  // caches as external tensor patterns.
  kRunner,
};

struct TensorApiExecutorOptions {
  // model.safetensors of Qwen/Qwen3-0.6B.
  std::string weights_path;
  bool use_gpu = false;
  // GPU only: fp32 | fp16 calculation precision, and default | buffer |
  // texture2d tensor storage.
  std::string gpu_precision = "fp32";
  std::string gpu_buffer_storage = "buffer";
  // Lengths of the prefill signatures. A prefill runs in chunks; each chunk
  // uses the shortest signature that holds it (the longest one otherwise) and
  // is padded on the right with id 0.
  std::vector<int> prefill_lengths = {128, 1024};
  WeightQuantMode weights_quant = WeightQuantMode::kNone;
  // Adds an int8 per-channel copy of the embedding table for the lm_head.
  bool lm_head_int8 = false;
  // CPU only: XNNPACK threads (0 = the runtime default).
  int num_threads = 0;
  // CPU only: the XNNPACK flags of LiteRT-LM's own CPU executor (the default
  // flags plus dynamic fully-connected and the latest operators) instead of
  // LiteRT's defaults.
  bool litert_lm_xnnpack_flags = false;
  // Logs every token id the executor receives (prefill ids, decode inputs).
  bool log_ids = true;
  // Compiles the model once more and logs the delegation metrics.
  bool delegation_report = false;
  // Graph forms (Qwen3Config): the attention core as the
  // odml.scaled_dot_product_attention composite, every RMSNorm as the
  // odml.rms_norm composite. Off = the raw-op graph.
  bool sdpa_attention = false;
  bool rms_norm_composite = false;
  // The attention core as the odml.sdpa_transposed composite with the value
  // caches transposed, and with the param tensor and the bool mask
  // (Qwen3Config::sdpa_transposed and sdpa_param_tensor).
  bool sdpa_transposed = false;
  bool sdpa_param_tensor = false;
  // The raw attention ops with the value caches stored transposed
  // (Qwen3Config::transposed_value_cache).
  bool transposed_value_cache = false;
  // GPU only: constant tensors (weights) shared between the signatures.
  bool gpu_share_constants = false;
  // GPU only: compiled programs (and, with shared constants, the converted
  // weights) serialized to the cache directory under this form's key.
  bool gpu_program_cache = false;
  GpuExternalCaches gpu_external_caches = GpuExternalCaches::kOff;
  // GPU only: the other compile options of LiteRT-LM's GPU executor
  // (CreateCompilationOptions and SetCommonGpuOptions), without the KV cache
  // patterns and the command buffer preparation.
  bool gpu_litert_lm_options = false;
  // GPU only: steps of command buffer preparation in advance (-1 = not set).
  int gpu_command_buffer_steps = -1;
  // CPU only: XNNPACK weight cache file in the cache directory.
  bool xnnpack_weight_cache = false;
  ModelFileMode model_file = ModelFileMode::kMemory;
  // Logs the init phases, the memory footprint at each stage and the time of
  // each part of a decode step.
  bool report = false;
  // With report: seconds to sleep after the decode breakdown is logged (for
  // an outside footprint of the process).
  int report_hold_s = 0;
};

class TensorApiExecutor : public ::litert::lm::LlmExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<TensorApiExecutor>> Create(
      const ::litert::lm::LlmExecutorSettings& settings,
      const TensorApiExecutorOptions& options);
  ~TensorApiExecutor() override;

  absl::Status Prefill(const ::litert::lm::ExecutorInputs& inputs) override;
  absl::Status Prefill(
      const ::litert::lm::ExecutorInputs& inputs,
      const ::litert::lm::ExecutorPrefillParams& prefill_params) override;

  // Greedy pick inside the executor (used when the session has no sampler).
  absl::StatusOr<std::vector<std::vector<int>>> Decode() override;
  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ::litert::lm::ExecutorDecodeParams& decode_params) override;

  absl::Status Decode(const ::litert::lm::ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;
  // Returns [1, 1, vocab] fp32 logits in host memory.
  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ::litert::lm::ExecutorInputs& inputs) override;

  absl::string_view ExecutorBackendName() const override;
  absl::StatusOr<int> GetVocabSize() override;
  absl::StatusOr<int> GetCurrentStep() const override;
  absl::Status SetCurrentStep(int current_step) override;
  absl::StatusOr<::litert::lm::LlmExecutorSettings> GetExecutorSettings()
      const override;
  absl::Status UpdateExecutorSettings(
      const ::litert::lm::LlmExecutorSettings& executor_settings) override;
  absl::StatusOr<::litert::lm::RuntimeConfig> GetRuntimeConfig()
      const override;
  absl::Status UpdateRuntimeConfig(
      const ::litert::lm::RuntimeConfig& runtime_config) override;
  absl::StatusOr<::litert::lm::RuntimeState> GetRuntimeState() const override;
  absl::Status UpdateRuntimeState(
      const ::litert::lm::RuntimeState& runtime_state) override;
  absl::StatusOr<const ::litert::lm::ProcessedTokens*> GetProcessedTokens()
      const override;
  ::litert::Environment* GetEnvironment() const override;
  absl::Status Reset() override;
  absl::StatusOr<std::unique_ptr<::litert::lm::LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      ::litert::lm::RuntimeConfig runtime_config) const override;
  absl::StatusOr<std::unique_ptr<::litert::lm::LlmContext>> CloneContext()
      const override;
  absl::Status RestoreContext(
      std::unique_ptr<::litert::lm::LlmContext> llm_context) override;
  absl::StatusOr<std::string> GetProfileSummary() override;

 private:
  // Microseconds of the parts of one DecodeLogits call (report only).
  struct StepTimes {
    double inputs = 0;    // step inputs written, caches bound
    double run = 0;       // Run
    double caches = 0;    // cache outputs taken as the next inputs
    double readback = 0;  // logits GetOutput + Lock + copy to the host
    double copy_out = 0;  // logits copied to the returned buffer
    double total = 0;     // the whole call
    double outside = -1;  // from the previous call's return (-1: none)
  };

  TensorApiExecutor(const ::litert::lm::LlmExecutorSettings& settings,
                    const TensorApiExecutorOptions& options);

  absl::Status Initialize();
  // The key of this form's serialized model and caches.
  absl::StatusOr<std::string> ModelKey() const;
  // Logs the decode step times collected since the last summary.
  void ReportDecodeSteps();
  ::litert::lm::ProcessedTokens& Tokens();
  // Drops processed tokens past the current step (after a rewind), keeping
  // the token at the current step pending.
  absl::Status RollBackTokens();
  // Runs `ids` at positions start .. start + ids.size() - 1 through the
  // prefill signatures.
  absl::Status RunPrefill(absl::Span<const int> ids, int start);
  // Runs `id` at `position` through the decode signature and copies the
  // logits to logits_. Fills `times` (when not null).
  absl::Status RunDecode(int id, int position, StepTimes* times = nullptr);
  // Waits until the last call's caches are written: on the GPU, Run can
  // return once the work is enqueued, and locking an output buffer waits.
  absl::Status WaitForCaches();
  // Sets the token, position, RoPE and mask inputs of `signature` for `ids`
  // (the signature's length) starting at `position`.
  absl::Status SetStepInputs(const std::string& signature,
                             const std::vector<int32_t>& ids, int position);
  // Binds the current caches as the cache inputs of `signature`, and after a
  // run takes its cache outputs as the current caches.
  absl::Status BindCaches(const std::string& signature);
  absl::Status TakeCaches(const std::string& signature);

  ::litert::lm::LlmExecutorSettings settings_;
  TensorApiExecutorOptions options_;
  Qwen3Config config_;
  std::vector<int> prefill_lengths_;
  std::unique_ptr<::litert::Environment> env_;
  // The serialized model; the compiled model reads it in place.
  std::vector<char> model_buffer_;
  std::optional<LitertDynamicRunner> runner_;
  // The current cache of every layer, as handles over runner buffers (or the
  // zero cache before the first run of a context).
  std::vector<TensorHandle> key_caches_;
  std::vector<TensorHandle> value_caches_;
  TensorHandle zero_cache_;
  // The value caches' zeros: zero_cache_ unless they are transposed.
  TensorHandle zero_value_cache_;
  // Token ids whose keys and values are in the caches, by position.
  std::vector<int> cached_ids_;
  std::unique_ptr<::litert::lm::LlmContext> llm_context_;
  std::vector<float> logits_;
  ::litert::TensorBuffer logits_host_;
  // Report only.
  std::vector<StepTimes> step_times_;
  std::optional<std::chrono::steady_clock::time_point> last_decode_end_;
  bool prefill_reported_ = false;
  bool decode_reported_ = false;
};

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_API_QWEN3_0_6B_TENSOR_API_EXECUTOR_H_
