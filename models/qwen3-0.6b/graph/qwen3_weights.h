// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B checkpoint loading (model.safetensors, bf16 -> fp32).
//
// Reuses the gemma3 example's SafetensorLoader and undoes its Gemma-only
// norm convention: that loader adds +1.0 to every tensor named *layernorm* or
// *norm.weight (Gemma stores w - 1); Qwen3 norms use the raw weight.
// The lm_head is tied: the graph reuses model.embed_tokens.weight, so the
// checkpoint's lm_head.weight (the same bytes) is not loaded.
//
// Optional nearest-integer weight quantization while loading, on the 196
// decoder matmul weights (q/k/v/o and gate/up/down of every layer):
//   int8: symmetric per output channel, scale = max|w| / 127, zero point 0
//         (PerChannelAffineQuantization, dimension 0).
//   int4: symmetric per block of 32 input elements, scale = max|w| / 7
//         rounded to fp16 before the weights are rounded against it, codes in
//         [-7, 7] bit-packed two per byte (kI4, BlockwiseQuantization).
// Norm weights and the embedding table (the lookup source and the tied
// lm_head) stay fp32. With quantized_lm_head, an extra int8 per-channel copy
// of the table is added as "lm_head.weight" for the lm_head FullyConnected
// only (the lookup keeps the fp32 table).
// With dequantize_to_fp32, every quantized weight (the lm_head copy too) is
// stored as its quantized values, code * scale, in an fp32 tensor: the graph
// sees the same rounded weights as the quantized form, with fp32 types.

#ifndef TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_
#define TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"

namespace litert::tensor::examples::qwen3 {

enum class WeightQuantMode { kNone, kInt8, kInt4 };

// What the quantization changed, for the record.
struct QuantStats {
  int quantized_tensors = 0;
  // Bytes of the quantized weights as fp32, and as stored after quantization
  // (codes plus scales: fp32 per-channel scales for int8, fp16 block scales
  // for int4; with dequantize_to_fp32, the fp32 values).
  int64_t fp32_bytes = 0;
  int64_t quantized_bytes = 0;
  // The extra int8 lm_head copy, codes plus scales, or its fp32 values with
  // dequantize_to_fp32 (0 when not added).
  int64_t lm_head_bytes = 0;
};

absl::StatusOr<WeightMap> LoadCheckpointWeights(
    const Qwen3Config& config, const std::string& path,
    WeightQuantMode quant_mode = WeightQuantMode::kNone,
    bool quantized_lm_head = false, QuantStats* stats = nullptr,
    bool dequantize_to_fp32 = false);

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_WEIGHTS_H_
