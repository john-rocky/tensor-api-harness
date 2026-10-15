# Row 8 — attention and norm forms per backend

Forms (flags of `graph/qwen3_main`): raw = the raw-op graph of iterations 1 and 2; rms =
`--norms=composite`, every RMSNorm (input, post-attention, q_norm, k_norm and the final norm) as the
`odml.rms_norm` composite with the `epsilon` attribute and the raw math as decomposition; sdpa =
`--attention=sdpa`, the attention core as the `odml.scaled_dot_product_attention` composite with the
`scale` attribute. For sdpa, grouped-query attention is pre-folded into batched multi-head shapes
before the composite: q `[8, 2·S, 1, 128]` against k/v `[8, 64, 1, 128]` (one kv group per batch
entry, its two q heads along the sequence axis). The additive mask is tiled to `[1, 1, 2·S, 64]` and
passed as the fourth operand. The decomposition is the raw fold. All forms are fp32 weights. Metal
runs at fp32 delegate precision with buffer storage.

Match: greedy 16 tokens for three prompts against `reference/ref_ids.json` (ids equal before the first
mismatch, out of 48), and for the CPU the per-block match with the iteration 1 manifests. Chain: the 16
logits rows of the greedy run against the stateless reference (`reference/manifest_chain_p*.json`).
Time: decode ms per token from `speed.md` (median of the three per-prompt medians, measured in one quiet
window).

| form | backend | nodes delegated / total, partitions | text match | ids /48 | other match | decode ms/token | prefill ms (S 5 / 4 / 18) |
|---|---|---|---|---|---|---|---|
| raw | CPU | XNNPACK 3880 / 3994, 58 | PASS 3/3 (`text_match.txt`) | 48 | per-block 31/31 ×3 (`block_match_p*.txt`); this build's raw run is byte-identical to the earlier CPU runs (48 of 48 chain rows of iteration 2, 93 of 93 block dumps of iteration 1) | 18.12 | 42.8 / 25.4 / 47.6 |
| rms | CPU | XNNPACK 3880 / 4220, 454 | PASS 3/3 (`text_match_cpu_fp32_rms.txt`) | 48 | per-block 31/31 ×3 (`block_match_cpu_fp32_rms_p*.txt`); **byte-identical to raw**: ids equal, 48 of 48 chain rows and 93 of 93 block dumps identical | 18.10 | 38.1 / 26.2 / 48.1 |
| sdpa | CPU | XNNPACK 4104 / 4274, 114 (the compile for the report succeeds) | — | — | the runner's compile fails: `XNNPack delegate failed to reshape runtime`, `Node number 1886 (TfLiteXNNPackDelegate) failed to prepare`, `Failed to allocate tensors`; no output | — | — |
| rms + sdpa | CPU | not run (sdpa alone fails on the CPU) | — | — | — | — | — |
| raw | Metal fp32 | 3994 / 3994, 2 | PASS 3/3 (`text_match_metal_fp32.txt`, iteration 2) | 48 | chain 16/16 ×3 (`kv_chain_metal_fp32_p*.txt`); per-block 30/31 ×3, `final_norm` export fails (row 10) | 8.06 | 28.8 / 24.8 / 32.1 |
| rms | Metal fp32 | 4220 / 4220, 2 | PASS 3/3 (`text_match_metal_fp32_rms.txt`) | 48 | chain rows max abs difference 9.3e-5 against the reference; not byte-identical to Metal raw (0 of 48 rows; at most 5.1e-5 apart) | **7.86** | 26.9 / 26.5 / 29.8 |
| sdpa | Metal fp32 | 4274 / 4274, 2 | PASS 3/3 (`text_match_metal_fp32_sdpa.txt`) | 48 | chain rows max 1.1e-4; 0 of 48 rows byte-identical to Metal raw (at most 5.0e-5 apart) | 8.60 | 26.2 / 24.9 / 33.0 |
| rms + sdpa | Metal fp32 | 4500 / 4500, 2 | PASS 3/3 (`text_match_metal_fp32_rms_sdpa.txt`) | 48 | chain rows max 1.0e-4; 0 of 48 rows byte-identical to Metal raw (at most 4.4e-5 apart) | 8.30 | 27.5 / 26.2 / 30.4 |

Chosen form per backend (fp32 weights):

- CPU: **rms** — byte-identical to raw and 18.10 against 18.12 ms/token. The two are within the
  spread of the per-prompt medians (17.98–18.54 for raw, 18.07–18.19 for rms), so on the CPU the
  composite neither costs nor gains time measurably; raw is the verification form, and rms gives the
  same bytes.
- Metal fp32: **rms** — the fastest form that passes the text match: 7.86 ms/token against 8.06 for raw
  (−2.5%). sdpa is slower than raw at these shapes (8.60), and rms + sdpa (8.30) is slower than rms
  alone.

Notes:

- The node totals grow with the composites (3994 raw, 4220 rms, 4274 sdpa, 4500 both). How the runtime
  counts a composite and its decomposition in these totals was not inspected: unverified.
- The sdpa composite is the only form that fails on the CPU; the same graph runs on Metal with every node
  delegated and passes the text match. The fold with the mask as a fourth operand was not tried in an
  un-folded presentation.
