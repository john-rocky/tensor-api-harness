# `odml.sdpa_transposed` on the GPU accelerator: a minimal reproduction

A model whose only op is the `odml.sdpa_transposed` composite, run from the same `.tflite` file on
the CPU and on the GPU accelerator and compared with a float64 computation of the same inputs.
It was written for the Metal accelerator on macOS, where the Qwen3-0.6B graph of this repository
gives wrong ids from the first generated token with this composite (iteration 11,
`models/qwen3-0.6b/results/attention_forms.md`).

## Run

In a LiteRT checkout whose `litert/prebuilt/` files are present (`git lfs pull`), with bazelisk:

```
tools/repro_sdpa_transposed_metal/run.sh <LiteRT checkout> [<out dir>]
```

`run.sh` links this directory into the checkout as `tensor/examples/sdpa_transposed_repro`,
builds `//tensor/examples/sdpa_transposed_repro:sdpa_repro_main`, and runs every case of
`cases.tsv` with the checkout's GPU accelerator library (`litert/prebuilt/macos_arm64`). It prints
one line per case and exits 0 when every case passes, 1 when one fails, 2 on a setup or run error.
`--dry-run` checks the setup and prints the commands; `--no-build --bin <binary>` reuses a binary.

Each case writes to `<out dir>/<case>/`: the model, the inputs and outputs as raw float32
(`query.f32`, `key_cache.f32`, `value_cache.f32`, `attention_mask.f32` or `.u8` for a bool mask,
`cpu.f32`, `gpu_run1.f32` … `gpu_run3.f32`, `reference.f32`), `log.txt` and `summary.json`.

## A case

`sdpa_repro_main` builds the model with the Tensor API: the query, already scaled by 1/√head_dim,
`[1, q_heads, seq, head_dim]`; the key cache `[1, kv_heads, kv_len, head_dim]`; the value cache
stored transposed `[1, kv_heads, head_dim, kv_len]`; the mask `[1, 1, seq, kv_len]`; and, with
`--param_tensor`, the `[1, 1, 1, 7]` int32 param tensor. Attributes `k_ts_idx` 2 and `v_ts_idx` 3,
no `scale` attribute: the operand layout LiteRT-Torch exports for this composite. The
decomposition packs the query heads of each key/value head along the sequence axis (query head h
reads key/value head h / (q_heads / kv_heads)), multiplies by the keys, adds the mask (or selects
with a bool mask, −10000 where hidden), takes Softmax and multiplies by the transposed values.
The CPU computes the decomposition (XNNPACK takes 6 of its 7 ops).

Inputs are uniform random from seed 1 (keys and values in [−1.5, 1.5] at every cache position,
the query in [−1.5, 1.5] / √head_dim). The float mask is causal: query token s sees keys
0..position + s (position 0 for several tokens, 5 for one), −30000 elsewhere. The CPU runs once
and the GPU three times on the same inputs. A case passes when every GPU run is within 1e-3 of the
CPU for every output element (1e-2 at fp16 precision).

`--composite=false` emits the same ops without the composite around them, the control.
`--context=harness` puts the composite where the Qwen3-0.6B graph has it: the query scaled in the
graph, the new key and value rows written into the caches with `DYNAMIC_UPDATE_SLICE`, the
output transposed and reshaped.

## Result on an Apple M4 Max

LiteRT main at `bea3bc69` (2026-10-14) with its `litert/prebuilt/macos_arm64/libLiteRtMetalAccelerator.dylib`
(sha256 `5531feca4ff0e2c67628e09d635a6f79da74534e88b7a3d262efa5dcda59e95d`; main at `6e72784f`,
2026-10-23, has the same file), macOS 27.0 (26A428), fp32 precision, buffer storage. In every case
the CPU is within 6.7e-7 of the float64 computation and the three Metal runs are identical.

| cases | query / key-value heads | query tokens | Metal against the CPU |
|---|---|---|---|
| A1–A4, F8 | 16 / 8 | 1, 2, 4, 5, 8 | FAIL: head 0 matches, heads 1–15 differ by up to 2.99 |
| B3–B9, D1, D2 | 16 / 8 | 5 or 1 | FAIL, the same pattern: no mask, a mask of zeros, a bool mask, no `is_causal`, 128 cache entries, default storage, fp16, the harness graph's surroundings |
| B11, F3 | 4 / 2 | 5, 1 | FAIL: heads 1–3 differ |
| B12 | 8 / 2 | 4 | FAIL: heads 1–7 differ |
| A5, F9–F14 | 16 / 8 | 9 to 12, 16 to 18 | FAIL: every head; values down to −30,002 in the output |
| F2, F16 | 8 / 1 | 18, 12 | FAIL: every head; values down to −30,002 in the output |
| B1, B10, B2, F15, F1 | 8 / 8, 16 / 16, 8 / 1, 8 / 1, 16 / 16 | 5, 5, 5, 8, 18 | PASS (at most 5.6e-7) |
| R1–R3 | 16 / 8, the same ops without the composite | 5, 1, 18 | PASS (at most 5.5e-7) |
| C1, C2, D3, F4, F5, F18 | param tensor, bool mask, `is_causal` and `from_cache_update` true | 1, 5 | FAIL, also with 16 / 16 heads |
| C3, F6, F17 | param tensor without `from_cache_update` | 5 | FAIL |
| F7 | 16 / 16, param tensor without `from_cache_update`, element 2 = 32 | 5 | PASS |

With 16 / 8 heads (and 4 / 2), Metal's output for query head h, h smaller than the number of
key/value heads, equals the attention of query head h over key/value head h (at most 2e-7 from a
float64 computation of that pairing, cases A1, A4, B3, B11, F3); the decomposition pairs query
head h with key/value head h / (q_heads / kv_heads). The other heads match no query / key-value
head pairing. With 18 query tokens (16 / 8), 4,775 of the 36,864 output values are below −100
(every correct value is within ±1.5), down to −30,002; the mask's hidden entries are −30,000.

In the `from_cache_update` cases the caches are plain inputs (or written with
`DYNAMIC_UPDATE_SLICE` in D3), not outputs of `odml.cache_update`. The param tensor is
`{position, position + seq, position + seq, seq, 0, 0, 0}` unless a case sets it.

LiteRT's own tools on the same accelerator:
- `litert/tools:gpu_numerics_check --graph=<case>/model.tflite --gpu_backend=metal`, with its own
  inputs: A4 max_abs_diff 1.99184251 and diff_ratio 0.9375 (15 of 16 heads), the same with
  `--gpu_use_ir_model=true` and `=false`; A1 1.97374821 and 0.9375; R1 and B1 4.17e-7.
- `litert/ats:ats --backend=gpu --data_seed=42 --do_register=SdpaTransposed --iters_per_test=16`:
  every `SdpaTransposed` test fails (f32 and f16; no mask, bool mask, float mask, with and without
  the param tensor), each on the generator's grid rows with 8 query heads over 2 or 4 key/value
  heads (rows 4, 5, 6, 11, 12, 13 of `litert/test/generators/sdpa_transposed.h`).
  `CacheUpdateSdpaTransposed` passes without a mask and fails rows 2, 5 and 7 with a bool mask and
  rows 3 and 12 (f32) or 12 (f16) with a float mask.

## Files

- `sdpa_repro_main.cc` — the model, the runs on the CPU and the GPU, the float64 reference and
  the comparison.
- `BUILD` — the target, built from the LiteRT checkout's root.
- `run.sh` — links, builds, runs `cases.tsv` and prints the table.
- `cases.tsv` — the cases: id, what changes, the flags.
