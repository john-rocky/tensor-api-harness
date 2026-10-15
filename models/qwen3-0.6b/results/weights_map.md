# Qwen3-0.6B — weights map

Source: `model.safetensors` of `Qwen/Qwen3-0.6B` (snapshot `c1899de289a04d12100db370d81485cdf75e47ca`), read from its header. 311 tensors in the file, 310 bound in the graph. Every tensor is bf16 in the file and widened to fp32 by the loader (`graph/qwen3_weights.cc`); the loader checks each shape against the config.

| name | shape | dtype in file | graph dtype | use |
|---|---|---|---|---|
| `model.embed_tokens.weight` | [151936, 1024] | BF16 | fp32 | `EmbeddingLookup` table and the lm_head `FullyConnected` weight (one tensor) |
| `model.layers.0.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.0.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.0.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.0.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.0.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.0.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.1.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.1.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.1.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.1.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.1.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.2.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.2.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.2.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.2.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.2.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.3.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.3.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.3.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.3.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.3.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.4.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.4.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.4.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.4.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.4.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.5.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.5.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.5.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.5.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.5.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.6.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.6.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.6.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.6.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.6.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.7.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.7.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.7.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.7.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.7.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.8.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.8.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.8.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.8.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.8.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.9.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.9.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.9.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.9.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.9.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.10.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.10.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.10.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.10.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.10.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.11.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.11.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.11.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.11.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.11.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.12.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.12.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.12.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.12.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.12.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.13.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.13.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.13.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.13.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.13.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.14.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.14.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.14.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.14.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.14.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.15.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.15.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.15.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.15.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.15.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.16.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.16.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.16.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.16.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.16.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.17.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.17.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.17.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.17.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.17.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.18.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.18.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.18.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.18.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.18.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.19.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.19.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.19.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.19.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.19.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.20.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.20.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.20.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.20.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.20.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.21.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.21.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.21.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.21.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.21.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.22.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.22.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.22.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.22.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.22.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.23.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.23.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.23.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.23.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.23.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.24.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.24.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.24.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.24.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.24.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.25.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.25.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.25.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.25.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.25.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.26.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.26.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.26.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.26.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.26.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.input_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.27.mlp.down_proj.weight` | [1024, 3072] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.mlp.gate_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.mlp.up_proj.weight` | [3072, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.post_attention_layernorm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.27.self_attn.k_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.27.self_attn.k_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.self_attn.o_proj.weight` | [1024, 2048] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.self_attn.q_norm.weight` | [128] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
| `model.layers.27.self_attn.q_proj.weight` | [2048, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `model.layers.27.self_attn.v_proj.weight` | [1024, 1024] | BF16 | fp32 | `FullyConnected` weight [out, in] |
| `lm_head.weight` | [151936, 1024] | BF16 | — | not read: the lm_head uses `model.embed_tokens.weight` (tied; same bytes) |
| `model.norm.weight` | [1024] | BF16 | fp32 | RMSNorm weight, raw (the loader's +1.0 is subtracted back) |
