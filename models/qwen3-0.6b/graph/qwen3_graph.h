// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B prefill and decode graphs on the Tensor API.
//
// Both signatures take token ids, look up the embedding rows in-graph, run
// the 28 decoder layers with a fixed-size KV cache written at a runtime
// position (DynamicUpdateSlice), and end in the final RMSNorm and the tied
// lm_head (the embedding table reused as the FullyConnected weight).
// RoPE cos/sin rows and the additive attention mask are host inputs; the
// token loop and the greedy pick stay on the host.

#ifndef TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_GRAPH_H_
#define TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_GRAPH_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::qwen3 {

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;
using WeightMap = absl::flat_hash_map<std::string, TfTensor>;

struct Qwen3Inputs {
  // [seq] kI32 token ids (prefill: prefill_len, decode: 1).
  TfTensor input_ids;
  // [4] kI32 {0, 0, position, 0} — runtime KV write position.
  TfTensor cache_position;
  // [1, 1, seq, head_dim] kFP32 each.
  TfTensor rope_cos;
  TfTensor rope_sin;
  // [1, 1, seq, max_seq] kFP32 additive mask.
  TfTensor attention_mask;
  // sdpa_transposed only: [4] kI32 {0, 0, 0, position} — the write position
  // in the transposed value caches.
  bool has_value_cache_position = false;
  TfTensor value_cache_position;
  // sdpa_param_tensor only: [1, 1, 1, 7] kI32 {start, end, end, 0, 0, 0, 0}
  // with end = start + seq (the values LiteRT-LM writes for its GPU models).
  bool has_param_tensor = false;
  TfTensor param_tensor;
  // Per layer: [1, n_kv_groups, max_seq, head_dim] kFP32; with
  // sdpa_transposed the value caches are [1, n_kv_groups, head_dim, max_seq].
  std::vector<TfTensor> key_caches;
  std::vector<TfTensor> value_caches;

  std::vector<TfTensor> AsList() const;
};

struct Qwen3Outputs {
  // [1, seq, vocab_size] kFP32. Not part of the outputs when omit_logits.
  TfTensor logits;
  bool omit_logits = false;
  // Updated caches, same shapes as the inputs.
  std::vector<TfTensor> key_caches;
  std::vector<TfTensor> value_caches;
  // dump_blocks only (prefill): emb, block_00..block_<n-1>, final_norm, each
  // [1, seq, emb_dim].
  std::vector<TfTensor> dumps;

  std::vector<TfTensor> AsList() const;
};

Qwen3Inputs MakeQwen3Inputs(const Qwen3Config& config, bool is_decode);

// Every weight tensor the graphs consume, under the checkpoint names, with
// the expected shape. Shared by the synthetic generator and the loader.
struct WeightSpec {
  std::string name;
  std::vector<int> shape;
  float init_scale;  // synthetic init only: 1.0 for norms, 0.02 elsewhere
};
std::vector<WeightSpec> GetWeightSpecs(const Qwen3Config& config);

WeightMap MakeSyntheticWeights(const Qwen3Config& config, unsigned seed);

Qwen3Outputs BuildQwen3(const Qwen3Config& config, const Qwen3Inputs& inputs,
                        const WeightMap& weights, bool is_decode);

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_GRAPH_H_
