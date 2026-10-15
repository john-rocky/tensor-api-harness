# Delegation on the Metal delegate (rows 9 and 11), with the CPU for comparison

## How the numbers were taken

`qwen3_main --delegation_report` (on by default) compiles the serialized model once with the run's
options before the runner is created and prints `CompiledModel::GetDelegationMetrics()`: the total
node count, then per accelerator the delegated node count and the partition count, then
`IsFullyAccelerated()` and `IsNonCpuFullyAccelerated()`. The runner compiles the model a second
time for the run itself (it keeps its `CompiledModel` private). Each prompt has its own model
(the prefill length is the prompt length: S = 5, 4 and 18), so every line below was printed three
times, once per prompt, with the same numbers. Both signatures (prefill and decode) are in one
model; the KV cache length is 64.

The Metal accelerator is the prebuilt `libLiteRtMetalAccelerator.dylib`
(`Dynamically loaded GPU accelerator(libLiteRtMetalAccelerator.dylib) registered.`). The log does
not print the GPU's name.

## Per form

| form | total nodes | accelerator | delegated nodes | partitions | fully accelerated | output check |
|---|---|---|---|---|---|---|
| CPU fp32 | 3994 | CpuAccelerator (XNNPACK) | 3880 | 58 | false | row 7: `kv_chain_p*.txt` 16/16 each |
| Metal fp32, buffer storage | 3994 | GPU Metal | 3994 | 2 | true (and non-CPU fully accelerated) | `text_match_metal_fp32.txt` 3/3; `kv_chain_metal_fp32_p*.txt` 16/16 each; `block_match_metal_fp32_p*.txt` 30/31 each (see below) |
| Metal fp32, default storage | 3994 | GPU Metal | 3994 | 2 | true | `text_match_metal_fp32_default.txt` 3/3 |
| Metal fp32, texture2d storage (p0 only) | 3994 | GPU Metal | 3994 | 2 | true | greedy ids equal the reference ids (21 of 21) |
| Metal fp16, buffer storage | 3994 | GPU Metal | 3994 | 2 | true | `text_match_metal_fp16.txt` 0/3; `block_match_metal_fp16_p*.txt` fail (see below) |

On Metal, no op fell back to the CPU in any form: every one of the 3994 nodes is delegated, in
two partitions, one per signature (the log shows `Initializing Metal-based API from graph.` twice
per compile). That includes the embedding lookup (`EMBEDDING_LOOKUP` with rank-1 int32 ids), the
`DYNAMIC_UPDATE_SLICE` cache writes at the runtime `cache_position`, and the tied lm_head
`FULLY_CONNECTED` over the [151936, 1024] table. No compile failure text appeared in any form.

On the CPU, 3880 of the 3994 nodes go to XNNPACK in 58 partitions and 114 run on the built-in
kernels. Which ops those 114 are is not printed by the runtime: unverified.

With the logger at its lowest severity (`--log_level=verbose`, Metal fp32 buffer storage, p0), the
log has no line naming an op, a fallback or an unsupported op. Since every node is delegated, there
was no fallback to name; whether the runtime would name one at this level is unverified.

## Is the GPU computing? (silent-fallback check)

- Every Metal fp32 output differs in bytes from the CPU fp32 output of the same graph: the 48 chain
  rows (16 per prompt) and the three prefill last-position logits. The largest difference between
  Metal fp32 and CPU fp32 at the last prompt position is 4.5e-5 (p0), 3.5e-5 (p1), 5.1e-5 (p2).
- The delegation metrics report 3994 of 3994 nodes on GPU Metal for every form.
- Together: the Metal runs computed on the GPU; there is no silent CPU fallback at these shapes.

## Storage: default and texture2d against buffer

At these shapes (KV cache length 64, prefill lengths 4 to 18, the 2.4 GB fp32 graph) the three
storage settings give the same delegation (3994 / 3994, 2 partitions) and byte-identical outputs:
all 48 chain rows of the default-storage run equal the buffer-storage run's, and all 16 p0 chain
rows of the texture2d run equal them too. So at the shapes this model uses here, the default
storage does not push anything to the CPU. Larger KV cache lengths were not run.

## Metal fp32: the one block-match failure

With `--dump_blocks` (the prefill signature also outputs the embedding, the 28 block outputs and
the final norm), Metal fp32 passes the embedding, all 28 blocks and the last-position logits, and
fails only `final_norm` on every prompt (`block_match_metal_fp32_p*.txt`):

| prompt | block_27 max abs difference | final_norm | logits_last max abs difference |
|---|---|---|---|
| p0 | 1.6e-2 | max abs `nan`, cosine `nan` | 5.6e-5 |
| p1 | 5.9e-3 | max abs 4.9e1, cosine -0.042 | 3.5e-5 |
| p2 | 1.0e-2 | max abs 3.2e38, cosine 0.0051 | 4.2e-5 |

The `final_norm` values read back from Metal are mostly zeros (526 of 5120 for p0, 3875 of 4096
for p1, 14237 of 18432 for p2) mixed with values such as 1.99e14 and 4.7e-32, while the logits
that the lm_head computes from that same tensor match the reference. So the GPU computes the final
norm correctly for its consumer, but the values exported as a signature output are not the computed
ones. The cause is not established. `final_norm` is the only dumped tensor whose consumer is the
lm_head `FULLY_CONNECTED`; the block outputs, which pass, feed the next block. The production
signatures (no dumps) do not export `final_norm`.

## Metal fp16

Same graph, `--gpu_precision=fp16`, buffer storage, all 3994 nodes delegated:

- Per block (`block_match_metal_fp16_p*.txt`): the difference grows block by block. The cosine at
  block_26 is 0.976 (p0), 0.976 (p1), 0.894 (p2). block_27 is `nan` for p0 and p2 and has a cosine
  of 0.57 for p1. The last-position logits have cosine 0.598 (p0), 0.426 (p1), 0.000 (p2). The
  first tensor over the bar is block_05 (p0), block_02 (p1) and block_00 (p2, max abs 2.1e-2
  against a bar of 2e-2).
- The prefill argmax differs from the reference at every position: 5 of 5 (p0), 4 of 4 (p1),
  18 of 18 (p2).
- Greedy decode (`text_match_metal_fp16.txt`): every prompt mismatches at its first generated
  token, and each prompt repeats one id 16 times (624 for p0, 5183 for p1, 0 for p2).
- Chained decode against the stateless reference: 0 of 16 rows have the reference argmax, for each
  prompt; the largest row max abs difference is 33.5.
- Some residual-stream values of this model reach about 6,500 in magnitude from block_02 on
  (`max_abs_ref` in `reference/manifest_p*.json`). Whether an fp16 overflow inside the delegate's
  kernels is the cause of the `nan` at block_27 is not established.
