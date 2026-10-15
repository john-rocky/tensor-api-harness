// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B text decoder config (Qwen/Qwen3-0.6B config.json).

#ifndef TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_CONFIG_H_
#define TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_CONFIG_H_

namespace litert::tensor::examples::qwen3 {

struct Qwen3Config {
  int emb_dim = 1024;
  int n_layers = 28;
  int n_heads = 16;
  int n_kv_groups = 8;
  int head_dim = 128;
  int hidden_dim = 3072;
  int vocab_size = 151936;
  float rms_norm_eps = 1e-6f;
  float rope_base = 1000000.0f;

  // Fixed KV cache length and the static prefill length (the prompt length).
  int max_seq = 64;
  int prefill_len = 8;

  // Graph-construction switch: add the embedding output, every decoder
  // block's output and the final norm output to the prefill signature's
  // outputs, for the per-block output match.
  bool dump_blocks = false;
  // dump_blocks only: feed the lm_head FullyConnected from Mul(final_norm,
  // 1.0) instead of from final_norm itself, so the exported final_norm is not
  // the FullyConnected's direct input.
  bool dump_final_norm_identity = false;

  // Graph-form switches (row 8). The defaults give the raw-op graph.
  // Emit every RMSNorm as the odml.rms_norm composite (decomposition = the
  // raw math).
  bool rms_norm_composite = false;
  // Emit the attention core as the odml.scaled_dot_product_attention
  // composite over GQA pre-folded into batched multi-head shapes, with the
  // additive mask as the fourth operand (decomposition = the raw fold).
  bool sdpa_attention = false;
  // Use the separately quantized "lm_head.weight" from the weight map for the
  // lm_head instead of the shared embedding table (row 12 option).
  bool separate_lm_head = false;

  int qkv_out_dim() const { return n_heads * head_dim; }
  int kv_out_dim() const { return n_kv_groups * head_dim; }
};

}  // namespace litert::tensor::examples::qwen3

#endif  // TENSOR_EXAMPLES_QWEN3_0_6B_QWEN3_CONFIG_H_
