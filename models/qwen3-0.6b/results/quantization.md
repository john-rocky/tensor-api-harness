# Row 12 — weight quantization: output match and bytes

Round-to-nearest quantization of the 196 decoder matmul weights (q/k/v/o and gate/up/down in each of
the 28 layers) while loading; norms and the embedding table stay fp32 (see `ENVIRONMENT.md`,
"Iteration 3"). The embedding table is both the lookup source and the tied lm_head, so in the first
forms the lm_head is fp32. The `+ lm_head int8` forms add an int8 per-channel copy of the table that
only the lm_head `FULLY_CONNECTED` reads.

Match: greedy decode of 16 tokens for the three prompts against `reference/ref_ids.json`, counted as
the generated ids equal to the reference before each prompt's first mismatch (3 × 16 = 48). The bar,
written before iteration 2: at least 44 of 48 passes. Metal forms run at fp32 delegate precision with
buffer storage.

Bytes: the quantized weights as fp32 are 1,761,607,680 bytes (196 tensors). "Weights stored" is codes
plus scales as the loader counts them (int8: one fp32 scale per output channel; int4: one fp16 scale
per block of 32). "Serialized" is the `.tflite` file both signatures share (fp32 graph:
2,384,789,652 bytes).

| form | weights stored (bytes) | serialized (bytes) | ids equal /48 (p0, p1, p2) | bar ≥ 44 | text match file | per-block match (fp32 bars, for reference) | delegation |
|---|---|---|---|---|---|---|---|
| CPU fp32 (iteration 2) | 1,761,607,680 | 2,384,789,652 | 48 (16, 16, 16) | passes | `text_match.txt` | 31/31 per prompt | XNNPACK 3880 / 3994 nodes, 58 partitions |
| CPU int8 | 441,778,176 | 1,071,852,948 | 41 (16, 16, 9) | **fails** | `text_match_cpu_int8.txt` | first failing tensor `block_00` (max abs 4.7e-2 / 4.2e-2 / 4.5e-2, cosine 0.99937 / 0.99935 / 0.99932); last-position logits cosine 0.99716 / 0.99563 / 0.99595 (`block_match_cpu_int8_p*.txt`) | XNNPACK 3880 / 3994, 58 partitions |
| CPU int4 | 247,726,080 | 898,490,644 | 6 (2, 2, 2) | **fails** | `text_match_cpu_int4.txt` | first failing tensor `block_00` (max abs 0.14 / 0.21 / 0.16, cosine 0.99097 / 0.99120 / 0.99104); last-position logits cosine 0.96037 / 0.96603 / 0.97680 (`block_match_cpu_int4_p*.txt`) | XNNPACK 3880 / 3994, 58 partitions |
| Metal fp32, int8 weights | 441,778,176 | 1,071,852,948 | 48 (16, 16, 16) | passes | `text_match_metal_fp32_int8.txt` | `block_00` passes; first failing tensor `block_01` (max abs 3.5e-2 / 8.7e-2 / 3.4e-2, cosine 0.99991 / 0.99994 / 0.99990); last-position logits cosine 0.99945 / 0.99925 / 0.99951 (`block_match_metal_fp32_int8_p*.txt`) | Metal 3994 / 3994, 2 partitions, fully accelerated |
| Metal fp32, int4 weights | 247,726,080 | 898,490,644 | — | not measured | — | — | the process ended with a segmentation fault (exit 139) about one second after `Initializing Metal-based API from graph.`, in the first compile of the first prompt; not retried |
| CPU int8 + lm_head int8 | 441,778,176 + 156,190,208 | 1,231,082,068 | 48 (16, 16, 16) | passes | `text_match_cpu_int8_lm_head_int8.txt` | not dumped | XNNPACK 3880 / 3994, 58 partitions |
| CPU int4 + lm_head int8 | 247,726,080 + 156,190,208 | 1,057,719,764 | 20 (2, 16, 2) | **fails** | `text_match_cpu_int4_lm_head_int8.txt` | not dumped | XNNPACK 3880 / 3994, 58 partitions |
| Metal fp32, int8 weights + lm_head int8 | 441,778,176 + 156,190,208 | 1,231,082,068 | 48 (16, 16, 16) | passes | `text_match_metal_fp32_int8_lm_head_int8.txt` | not dumped | Metal 3994 / 3994, 2 partitions |
| Metal fp32, int4 weights + lm_head int8 | 247,726,080 + 156,190,208 | 1,057,719,764 | — | not measured | — | — | segmentation fault (exit 139) at the same point as Metal int4 |

Notes:

- CPU int8 fails the bar (41 of 48) while Metal with the same int8 weights passes 48 of 48, and the CPU
  int8 per-block difference at `block_00` is larger than Metal's (cosine 0.99937 against 0.99993). How
  the CPU kernel treats the activations of an int8-weight `FULLY_CONNECTED` was not inspected in this
  iteration: unverified.
- CPU int8 with the int8 lm_head copy passes 48 of 48 where CPU int8 with the fp32 lm_head reaches 41: the
  p2 trajectory, which left the reference at its 10th generated token, stays on it with the other
  lm_head. One run per form; this is the measured outcome, not a claim that an int8 lm_head is more
  accurate.
- int4 blockwise-32 with nearest-integer codes fails the bar on the CPU in both lm_head forms (6 and 20 of 48).
- The Metal int4 crash was not investigated further in this iteration. It happens before any
  inference, in the delegate's initialization of the graph that holds blockwise int4 weights.
- The per-block files use the fp32 manifests unchanged; for quantized weights their FAIL lines are a
  record of where the difference starts, not a bar.
