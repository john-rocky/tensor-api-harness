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
};

class TensorApiExecutor : public ::litert::lm::LlmExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<TensorApiExecutor>> Create(
      const ::litert::lm::LlmExecutorSettings& settings,
      const TensorApiExecutorOptions& options);

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
  TensorApiExecutor(const ::litert::lm::LlmExecutorSettings& settings,
                    const TensorApiExecutorOptions& options);

  absl::Status Initialize();
  ::litert::lm::ProcessedTokens& Tokens();
  // Drops processed tokens past the current step (after a rewind), keeping
  // the token at the current step pending.
  absl::Status RollBackTokens();
  // Runs `ids` at positions start .. start + ids.size() - 1 through the
  // prefill signatures.
  absl::Status RunPrefill(absl::Span<const int> ids, int start);
  // Runs `id` at `position` through the decode signature and copies the
  // logits to logits_.
  absl::Status RunDecode(int id, int position);
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
  // Token ids whose keys and values are in the caches, by position.
  std::vector<int> cached_ids_;
  std::unique_ptr<::litert::lm::LlmContext> llm_context_;
  std::vector<float> logits_;
  ::litert::TensorBuffer logits_host_;
};

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_API_QWEN3_0_6B_TENSOR_API_EXECUTOR_H_
