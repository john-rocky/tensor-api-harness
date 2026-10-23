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

## Iteration 11 — the attention core as the `odml.sdpa_transposed` composite

Two more forms (`--attention` of `graph/qwen3_main`, `--tensor_api_attention` of the LiteRT-LM
executor; `iterations/ITERATION_11.md`). They follow the composite boundary (operands and
attributes) LiteRT-Torch exports for it (`litert_torch/generative/export_hf/experimental/composites/sdpa.py`,
`scaled_dot_product_attention_transposed`, main at `117ac7b8`):

- `sdpa_transposed`: operands query × 1/√128 `[1, 16, S, 128]`, the key cache `[1, 8, KV, 128]`,
  the value cache `[1, 8, 128, KV]` and the additive fp32 mask `[1, 1, S, KV]`; attributes
  `k_ts_idx` 2, `v_ts_idx` 3 and `is_causal` false (this graph's choice, so the mask decides; a
  text-only LiteRT-Torch export sets it true); no `scale` attribute. The value caches of both
  signatures are stored transposed and written at a second position input. Inside the region the
  decomposition packs the two query heads of each KV group along the sequence axis, so it reads the
  caches as they are.
- `sdpa_transposed_param`: the same, plus `param_tensor` (`[1, 1, 1, 7]` int32
  `{start, end, end, end − start, 0, 0, 0}`, the values the composite reads in a LiteRT-Torch export
  after its graph sets element 3) as the fifth operand, the mask as bool (the additive mask equal to
  0) and `is_causal` and `from_cache_update` true on both signatures. The caches are still written with
  `DYNAMIC_UPDATE_SLICE`; LiteRT-Torch's GPU export writes them with the `odml.cache_update` composite.
  The decomposition does not read the param tensor.

Harness runner (fp32 weights; CPU = XNNPACK, Metal = fp32 delegate precision with buffer storage;
prompts of 5, 4 and 18 tokens = p0, p1, p2, KV cache 64, greedy 16). Decode ms/token per run = the
median of the three per-prompt medians, timed through the logits readback; the forms in turn in one
quiet window (the machine's measurement lock held, so no other job's measurement or build started),
the first pass discarded (the second window; a shorter set from the first window is in
`sdpa_transposed.json`). "Mask input buffer" is the buffer the compiled model
creates for the `attention_mask` input of the prefill and decode signatures.

| form | backend | nodes delegated / total, partitions | mask input buffer | ids /48 (p0, p1, p2) | decode ms/token |
|---|---|---|---|---|---|
| sdpa_transposed | CPU | CpuAccelerator 3656 / 3882, 170 | decode: host memory; prefill: host memory | 48 (16, 16, 16) | — |
| sdpa_transposed_param, first try (param element 3 = 0) | CPU | CpuAccelerator 3600 / 3938, 170 | decode: host memory; prefill: host memory | 48 (16, 16, 16) | — |
| raw | Metal fp32 | GPU Metal 3994 / 3994, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 48 (16, 16, 16) | 8.113 (8 runs, 8.003 to 8.291) |
| sdpa | Metal fp32 | GPU Metal 4274 / 4274, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 48 (16, 16, 16) | 8.625 (8 runs, 8.475 to 8.786) |
| sdpa_transposed | Metal fp32 | GPU Metal 3882 / 3882, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 0 (0, 0, 0) | 7.836 (8 runs, 7.758 to 8.218) |
| sdpa_transposed_param, first try (param element 3 = 0) | Metal fp32 | GPU Metal 3938 / 3938, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 0 (0, 0, 0) | 7.748 (8 runs, 7.662 to 7.905) |
| sdpa_transposed_param | Metal fp32 | GPU Metal 3938 / 3938, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 0 (0, 0, 0) | 7.753 (8 runs, 7.637 to 7.897) |
| raw_vt (control) | CPU | CpuAccelerator 3880 / 3994, 58 | decode: host memory; prefill: host memory | 48 (16, 16, 16) | — |
| raw_vt (control) | Metal fp32 | GPU Metal 3994 / 3994, 2 | decode: Metal buffer packed; prefill: Metal buffer packed | 48 (16, 16, 16) | 8.057 (8 runs, 8.035 to 8.229) |

Through LiteRT-LM on WebGPU (the Tensor API executor with int8 per-channel weights, an int8 lm_head
copy and fp32 precision; LiteRT-LM's benchmark entry with prefill 1024 forced, decode 256, KV cache
2048; one process per run, the forms in turn in one quiet window, the first run of each discarded):

| form | nodes delegated / total, partitions | mask input buffer | ids /48 (p0, p1, p2) | decode tok/s | prefill tok/s | peak memory footprint (MB) |
|---|---|---|---|---|---|---|
| raw | GPU WebGPU 5889 / 5889, 3 | decode: WebGPU buffer packed; prefill_1024: WebGPU buffer packed; prefill_128: WebGPU buffer packed | 48 (16, 16, 16) | 79.29 (7 runs, 79.04 to 79.43) | 3951.90 (7 runs, 3804.94 to 4014.41) | 18189.7 |
| sdpa | GPU WebGPU 6299 / 6299, 3 | decode: WebGPU buffer packed; prefill_1024: WebGPU buffer packed; prefill_128: WebGPU buffer packed | 48 (16, 16, 16) | 59.26 (3 runs, 59.15 to 59.51) | 4222.02 (3 runs, 4109.46 to 4252.37) | 18129.0 |
| sdpa_transposed | GPU WebGPU 5725 / 5725, 3 | decode: WebGPU buffer packed; prefill_1024: WebGPU buffer packed; prefill_128: WebGPU buffer packed | 48 (16, 16, 16) | 87.82 (3 runs, 87.38 to 88.05) | 4474.95 (3 runs, 4363.82 to 4488.99) | 18140.2 |
| sdpa_transposed_param, first try (param element 3 = 0) | GPU WebGPU 5807 / 5807, 3 | decode: WebGPU buffer packed; prefill_1024: WebGPU buffer packed; prefill_128: WebGPU buffer packed | 0 (0, 0, 0) | 89.24 (3 runs, 88.81 to 90.22) | 5283.92 (3 runs, 5126.21 to 5457.36) | 18125.1 |
| sdpa_transposed_param | GPU WebGPU 5807 / 5807, 3 | decode: WebGPU buffer packed; prefill_1024: WebGPU buffer packed; prefill_128: WebGPU buffer packed | 0 (0, 0, 0) | 89.25 (3 runs, 89.03 to 89.28) | 5500.45 (3 runs, 5405.53 to 5533.27) | 18170.6 |

A control, in a second quiet window: `raw_vt` keeps the raw attention ops and stores the value
caches transposed as `sdpa_transposed` does (that cache layout without the composite). Through
LiteRT-LM on WebGPU, in turn with the raw graph and `sdpa_transposed`:

| form | nodes delegated / total, partitions | ids /48 (p0, p1, p2) | decode tok/s | prefill tok/s | peak memory footprint (MB) |
|---|---|---|---|---|---|
| raw | GPU WebGPU 5889 / 5889, 3 | 48 (16, 16, 16) | 80.77 (3 runs, 80.15 to 80.82) | 3939.45 (3 runs, 3915.01 to 4075.02) | 18224.6 |
| sdpa_transposed | GPU WebGPU 5725 / 5725, 3 | 48 (16, 16, 16) | 88.77 (3 runs, 88.67 to 89.43) | 4321.99 (3 runs, 4311.53 to 4465.78) | 18210.0 |
| raw_vt (control) | GPU WebGPU 5889 / 5889, 3 | 48 (16, 16, 16) | 90.99 (3 runs, 90.28 to 91.86) | 3932.26 (3 runs, 3899.45 to 4023.69) | 18149.0 |

With the transposed value caches alone, the raw ops decode at 90.99 tok/s against 80.77
for the raw graph (+12.7 %), and the composite form measures 88.77 (-2.4 % against the
control; 3 runs each, under 10 % apart, so the order of those two is not established): the decode
gain of `sdpa_transposed` over the raw graph comes from the cache layout. Prefill: in the first
window the composite form is ahead of the raw graph (4474.95 against 3951.90 tok/s,
+13.2 %); in the second the two are under 10 % apart with 3 runs each (4321.99 against
3939.45, +9.7 %), as the composite form is against `raw_vt` (3932.26), so those orders
are not established. On the CPU and on Metal, `raw_vt` matches the reference (first table).

Notes:

- On the CPU both new forms match the reference, where the `odml.scaled_dot_product_attention`
  form does not prepare in XNNPACK.
- On Metal both new forms give wrong ids from the first generated token, and on WebGPU the form
  with the param tensor does, while the same graphs match the reference on the CPU, which computes
  the decomposition. No log line names the kernel the GPU ran for the composite.
- The mask input stays a GPU buffer in every GPU form. In `sdpa_transposed_param` the bool mask is
  made in the graph from the fp32 input, so that input has a GPU consumer whether or not the region
  reads the mask; the buffer type does not show whether a fused kernel dropped it.
- A first try of `sdpa_transposed_param` passed `{start, end, end, 0, 0, 0, 0}` (what LiteRT-LM
  writes into the signature input, before an exported graph sets element 3). It gave the same ids
  as the second try on both GPUs ("first try" rows).
- The node totals of the new forms are below the raw graph's. How the runtime counts a composite and
  its decomposition was not inspected (as for the other composites above).
- A model with only this composite (`tools/repro_sdpa_transposed_metal/`, 2026-10-23, same LiteRT
  commit and Metal accelerator): without the param tensor, Metal is wrong whenever the query heads
  are grouped over two or more key/value heads (16 over 8 as here, also 8 over 2 and 4 over 2):
  up to 8 query tokens query head 0 matches the CPU and heads 1 and up do not, in decode and in
  prefill, and from 9 tokens on every head differs; with 8 over 8 or 16 over 16 heads it matches
  (8 over 1 up to 8 tokens), and so do the same ops without the composite. The
  `sdpa_transposed_param` form here sets `from_cache_update` while the caches are written with
  `DYNAMIC_UPDATE_SLICE`, not `odml.cache_update`; in the minimal model that form is wrong on
  Metal also with 16 query heads over 16 key/value heads.
