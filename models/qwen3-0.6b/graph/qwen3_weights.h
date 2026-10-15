// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B checkpoint loading (model.safetensors, bf16 -> fp32).
//
// Reuses the gemma3 example's SafetensorLoader and undoes its Gemma-only
// norm convention: that loader adds +1.0 to every tensor named *layernorm* or
// *norm.weight (Gemma stores w - 1); Qwen3 norms use the raw weight.
// The lm_head is tied: the graph reuses model.embed_tokens.weight, so the
// checkpoint's lm_head.weight (the same bytes) is not loaded.

#ifndef TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_
#define TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"

namespace litert::tensor::examples::qwen3 {

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Qwen3Config& config,
                                                const std::string& path);

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_
