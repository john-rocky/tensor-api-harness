# Qwen3-0.6B — architecture and op inventory

Checkpoint: `Qwen/Qwen3-0.6B` (`Qwen3ForCausalLM`, bf16 safetensors, 311 tensors). Config: 28 decoder
layers, hidden 1024, 16 query heads and 8 key/value heads (grouped-query attention, 2 query heads per
group), head dim 128 (so the query width is 2048, not the hidden size), MLP width 3072, RMSNorm eps
1e-6, RoPE theta 1e6 with no scaling, no attention bias, vocab 151936, tied word embeddings.
The checkpoint still stores `lm_head.weight`; its bytes equal `model.embed_tokens.weight`.

The graph code is in `graph/` (`qwen3_graph.cc`). Two signatures, `prefill` (S tokens) and `decode`
(1 token), share every weight.

## Blocks and ops

| block | ops in PyTorch | Tensor API op(s) used | note |
|---|---|---|---|
| token embedding | `nn.Embedding` lookup | `EmbeddingLookup(ids [S] int32, table [151936, 1024])` → `Reshape` to [1, S, 1024] | in-graph; 1-D int32 ids, emits the TFLite `EMBEDDING_LOOKUP` builtin |
| input RMSNorm (×28) | `x * rsqrt(mean(x²) + eps) * w` | `Mul`, `Mean` (last axis, keep dims), `Add`, `Rsqrt`, `Mul` | raw weight `w` (no +1 offset); the `odml.rms_norm` composite is a later option (attention-form row) |
| q / k / v projection | `nn.Linear` (no bias) | `FullyConnected` | q [2048, 1024], k and v [1024, 1024] |
| head split | `view` + `transpose(1, 2)` | `Reshape` + `Transpose` | [1, S, heads, 128] → [1, heads, S, 128] |
| q_norm / k_norm | RMSNorm over the head dim (128), before RoPE | same RMSNorm ops on the last axis of [1, heads, S, 128] | weights [128] |
| RoPE | `q * cos + rotate_half(q) * sin` | `Slice` ×2, `Mul` (by −1), `Concatenation`, `Mul`, `Add` | cos/sin rows [1, 1, S, 128] are signature inputs computed on the host |
| KV cache write | (dynamic cache) | `DynamicUpdateSlice(cache [1, 8, max_seq, 128], k or v, cache_position)` | runtime start index `{0, 0, pos, 0}` as an int32 input |
| GQA attention | `repeat_kv`, `q @ kᵀ * scale`, `+ mask`, `softmax`, `@ v` | `Reshape` (fold kv groups into batch, query heads into rows), `BatchMatMul` (adj_y), `Mul`, `Add`, `Softmax`, `BatchMatMul` | no repeated K/V tensor; head h reads group h / 2 (repeat_kv order). Additive mask [1, 1, S, max_seq], floor −30000 |
| o projection | `nn.Linear` | `Transpose` + `Reshape` + `FullyConnected` | o [1024, 2048] |
| residual adds | `+` | `Add` | |
| post-attention RMSNorm | as above | as above | |
| SwiGLU MLP | `down(silu(gate(x)) * up(x))` | `FullyConnected` ×3, `Logistic`, `Mul` ×2 | SiLU as `x * Logistic(x)` |
| final RMSNorm | as above | as above | |
| lm_head (tied) | `nn.Linear(1024, 151936)` | `FullyConnected(hidden, table)` with the same table tensor as the embedding lookup | logits [1, S, 151936] in fp32 |
| greedy pick | `argmax` | host argmax over the logits row | `ArgMax` exists (`tensor/arithmetic.h`) for an in-graph pick later |

## Ops with no direct equivalent and the rewrite

- RoPE: the Tensor API `Rope` op does not emit to the TFLite flatbuffer, so the rotation is written
  as rotate-half with host-computed cos/sin rows (`inv_freq = 1 / theta^(2i/128)`, the two halves
  repeated, as in the PyTorch `Qwen3RotaryEmbedding`).
- The token loop: a graph has no control flow, so prefill, the decode steps and the KV cache rebind
  (the previous call's cache outputs bound to the next call's inputs by handle) run on the host.
- Sampling: no random-number op; this iteration decodes greedily (argmax), with the argmax on the host.
- The causal mask and the cache position are host inputs, so one decode graph serves every position.
- Tokenizer: outside the graph. Token ids come from the Hugging Face tokenizer in a file.

No op in a required block lacks a mapping.
