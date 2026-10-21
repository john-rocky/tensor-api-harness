# Iteration 11 — Qwen3-0.6B: the attention core as the `odml.sdpa_transposed` composite on the Mac's two GPU paths

## Read first
- `results/attention_forms.md` and `results/litert_lm_levers.md`: the attention core as the
  `odml.scaled_dot_product_attention` composite (iteration 3's graph option). On Metal it ran fully
  delegated and 7 % slower per decode token than the raw ops (8.60 against 8.06 ms/token); through
  LiteRT-LM on WebGPU it took decode from 79.37 to 59.38 tok/s; on the CPU it does not prepare.
- LiteRT-Torch at `117ac7b8` (main, 2026-10-20):
  `litert_torch/generative/export_hf/experimental/composites/sdpa.py`,
  `scaled_dot_product_attention_transposed`: the composite it emits as `odml.sdpa_transposed` — operands
  (query × scale, key cache, value cache, mask, param tensor) in that order, attributes `k_ts_idx`,
  `v_ts_idx`, `is_causal` and, with the param tensor, `from_cache_update`; no `scale` attribute; the
  grouped-query packing inside the composite. `core/cache.py`: the cache layout it exports by default,
  `k_ts_idx` 2 (keys `[B, N, S, H]`) and `v_ts_idx` 3 (values `[B, N, H, S]`, transposed).
  `core/attention.py`: a text-only causal export sets `is_causal` on every global attention layer.
- LiteRT at `d469ea48`: `tflite/delegates/ynnpack/attention.cc`, LiteRT's YNNPACK delegate's reading
  of the same op (the sequence axis of the keys is 2 and of the values 3; without a `scale` attribute
  the scale is 1). It is not the harness runner's CPU path: the runner uses XNNPACK, which leaves this
  composite to its decomposition.
- LiteRT-LM at `d78488c4`: `runtime/executor/litert_compiled_model_executor_utils.cc`. The values it
  writes into `param_tensor` (`FillSingleBufferCacheParamTensor`: start, end, end, then zeros), and
  how it tells that the GPU computes the causal mask from `param_tensor`
  (`ShouldSkipGlobalCausalAttentionMask`: the compiled model gives the bool mask input a host
  memory buffer because no GPU consumer reads it).

## Goal (one)
Emit the same Qwen3-0.6B graph with the attention core as the `odml.sdpa_transposed` composite in the
form LiteRT-Torch exports it, and measure it on the Mac's two GPU paths — the harness runner on Metal
and the graph as a LiteRT-LM executor on WebGPU — under the conditions of the earlier composite rows,
with the raw graph and the earlier composite in the same quiet window.

The two forms, written before the runs (both off by default; the earlier `sdpa` form is unchanged):
- `sdpa_transposed`: the composite boundary above without the param tensor. The value caches of both
  signatures are stored transposed (`[1, 8, 128, KV]`) and written at a second position input,
  `value_cache_position` (`{0, 0, 0, position}`). The mask is the additive fp32 mask, `is_causal`
  false (this graph's choice, so the mask decides; a text-only LiteRT-Torch export sets it true).
- `sdpa_transposed_param`: the same with the `param_tensor` input (`[1, 1, 1, 7]` int32, the host
  writes LiteRT-LM's values `{position, position + n, position + n, 0, 0, 0, 0}`) as the fifth
  operand, the mask as bool (the additive mask equal to 0) and `is_causal` and `from_cache_update`
  true on both signatures.
The decomposition packs the two query heads of each KV group along the sequence axis, so it reads
the caches as they are; it does not read the param tensor.

## Acceptance
- Output match: greedy ids equal to the PyTorch reference for at least 44 of 48 (3 prompts × 16
  tokens; the harness runner on the CPU and on Metal, LiteRT-LM on WebGPU). A form under the bar is
  measured and listed, not compared.
- Metal (harness runner, fp32 weights, fp32 precision, buffer storage, prompts of 4 to 18 tokens, KV
  cache 64, 15 decode steps): one value per run = the median of the three per-prompt median decode
  ms/token; raw, the earlier composite and the two new forms in turn in one window, the first pass
  discarded, 8 measured passes; median, min and max.
- WebGPU (LiteRT-LM's benchmark entry, forced prefill 1024 / decode 256 / KV cache 2048; int8
  weights, an int8 lm_head copy, fp32 precision): the graph without composites, the earlier
  composite and the two new forms in turn in one window, the first run of each discarded, at least
  three measured, 8 each when the graph without composites and the fastest other form are within
  10 %; decode and prefill tok/s, median and range.
- Whether the fused kernels ran is stated only from what a run shows: a log line, the delegated node
  and partition counts, or the buffer type of the mask input; otherwise "not established".
- With the new flags off, the binaries give the earlier ids. `tools/iteration_close_check.py` passes;
  a `## Time` section.

## Stop conditions
- Waiting for quiet windows adds up to more than four hours, or eight hours pass from the start.
- Less than 40 GB free on the disk before a build.

## Scope
The graph sources (`graph/qwen3_config.h`, `qwen3_graph.{h,cc}`), the harness runner
(`graph/qwen3_main.cc`), the LiteRT-LM executor and its flags (`litert_lm/`), `results/attention_forms.md`,
a new `results/sdpa_transposed.json`, this file and its line in `README.md`.

## Do not
- Do not change the earlier forms' graphs, `checks/`, `tools/` or the output-match bar.
- Do not time a form outside the quiet window, and do not publish a number that was not measured.

## Progress
- The graph sources gained the two forms and, after the first window, a control (`raw_vt`: the raw
  attention ops with the value caches stored transposed, the cache layout of `sdpa_transposed`
  without the composite). The harness runner and the LiteRT-LM executor pass the flags, write the
  second position input and the param tensor, and print, in their delegation reports, the buffer
  type the compiled model gives the `attention_mask`, `param_tensor` and `key_cache_0` inputs.
- First window (23 minutes): the builds (the runner incremental, LiteRT-LM from a clean output
  base), the output match of the new forms on the CPU and of every form on Metal and WebGPU, WebGPU
  timing of four forms,
  a second try of `sdpa_transposed_param` (Corrections) and a first Metal set (3 or 4 runs per form).
- Second window (18 minutes): the control's builds and output match, WebGPU timing of the raw
  graph, `sdpa_transposed` and the control in turn, and the full Metal set (six forms, eight
  measured passes).
- The tables are in `results/attention_forms.md` ("Iteration 11"), every run in
  `results/sdpa_transposed.json`.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0; the skill table is unchanged):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Output match:** both new forms 48 of 48 on the CPU; `sdpa_transposed` 48 of 48 on WebGPU; 0 of 48
  for `sdpa_transposed` on Metal and for `sdpa_transposed_param` on Metal and WebGPU, from the first
  generated token. The control `raw_vt` 48 of 48 on the CPU, Metal and WebGPU.
- **Metal speed:** met in the second window (six forms, eight measured passes each, in turn with the
  raw graph); the first window's set stopped at its time limit after 3 or 4 runs per form.
- **WebGPU speed:** met: three or more measured runs per form in turn with the raw graph and the
  earlier composite; the raw graph and the fastest passing form are more than 10 % apart, so no
  8-run extension. The control's comparison is a second window's, three runs each.
- **Fused kernels:** not established. No log line names a kernel; the node and partition counts
  are those of a fully delegated graph; the mask buffer does not decide it (Op and delegate
  observations).
- **Default flags:** with the new flags off, the LiteRT-LM binary gives iteration 7's ids byte for
  byte (`results/litert_lm_ids_tapi_gpu_int8.json`), and the runner's raw form matches 48 of 48 on
  Metal.
- **Close check:** `python3 tools/iteration_close_check.py models/qwen3-0.6b/iterations/ITERATION_11.md`:
  `PASS iteration close: models/qwen3-0.6b/iterations/ITERATION_11.md`.

## Corrections
- The goal as written had the host write the param tensor as LiteRT-LM writes it,
  `{position, position + n, position + n, 0, 0, 0, 0}`. In a LiteRT-Torch export the graph then sets
  element 3 to end − start before the composites (`core/exportable_module.py`), so the composite
  reads `{start, end, end, n, 0, 0, 0}`; the hosts now write that. Both gave the same ids.
- The goal as written did not separate the composite from its cache layout. The first window's
  WebGPU gain of `sdpa_transposed` could come from either, so the control `raw_vt` was added for the
  second window: the gain comes from the layout.
- The goal's Metal set (eight measured passes) did not fit in the first window after the builds and
  the output match; it was taken again, complete, in the second window, and the first window's
  shorter set is kept in the JSON.

## Dead ends
- `sdpa_transposed` on Metal: every node on Metal, two partitions, ids wrong from the first
  generated token for all three prompts; the decomposition of the same graph matches on the CPU, and
  the raw ops with the same transposed caches match on Metal.
- `sdpa_transposed_param` on Metal and WebGPU: the same, with either value of param element 3. The
  caches are written with `DYNAMIC_UPDATE_SLICE`, not with LiteRT-Torch's `odml.cache_update`
  composite, which this graph does not emit; whether that is what the GPU's kernel needs was not
  tried.
- The mask buffer as the evidence of a fused kernel: the bool mask of `sdpa_transposed_param` is made
  in the graph from the fp32 input, so that input always has a GPU consumer. A bool mask as the
  signature input was not tried.

## Op and delegate observations
- Metal (the prebuilt Metal accelerator): every form fully delegated in two partitions; node totals
  3994 (raw, `raw_vt`), 4274 (the earlier composite), 3882 (`sdpa_transposed`), 3938
  (`sdpa_transposed_param`). WebGPU (LiteRT-LM's prebuilt WebGPU accelerator): fully delegated in three
  partitions; 5889 (raw, `raw_vt`), 6299, 5725, 5807.
- CPU (XNNPACK): `sdpa_transposed` 3656 of 3882 nodes in 170 partitions, `sdpa_transposed_param` 3600 of
  3938 in 170, `raw_vt` 3880 of 3994 in 58 (the raw graph's count); all three prepare and match.
- Buffer types from the compiled models: the mask input is a Metal or WebGPU buffer (packed) in every
  GPU form, the param tensor too where it exists; on the CPU every input is host memory.
- The GPU logs carry no line that names an attention kernel (the words flash, sdpa, fused and
  composite appear only in this repo's own log lines).
- Versions: LiteRT `bea3bc69` (the runner), LiteRT-LM `d78488c4` with LiteRT `9b8c7f77`.

## Environment
- Mac: Apple M4 Max, 128 GB, macOS 27.0 (26A428). No phone.
- Build: Bazel 7.7.0 (the runner, LiteRT's tree) and 7.6.1 (LiteRT-LM) through bazelisk 1.29.0. The
  runner rebuilt incrementally in 8 s; LiteRT-LM from a clean output base in 206 s; the second try
  and the control 4 to 7 s each.
- Accelerators: Metal `libLiteRtMetalAccelerator.dylib` sha256
  `5531feca4ff0e2c67628e09d635a6f79da74534e88b7a3d262efa5dcda59e95d`, WebGPU
  `libLiteRtWebGpuAccelerator.dylib` sha256
  `b7b7dd7bd6c90a41f6dee7d1f5b2ff76de2df39921f68f1c7e6d525f5136c4cf`; the binaries' sha256 are in
  `results/sdpa_transposed.json`.
- Python 3.14.6 for the logs and the checks.
- Two quiet windows, taken after 6 h 29 min and about 2 h 32 min of waiting (32,492 s in all), held
  1,387 s and 1,078 s; the machine's other applications kept running (load average 10 to 32 at the windows' starts and ends).

## Files left
- `results/attention_forms.md` (the "Iteration 11" section) and `results/sdpa_transposed.json` (every
  timed run, the output-match counts, the delegation and buffer reports, the binaries' sha256). Keep.
- The graph sources, `graph/qwen3_main.cc` and `litert_lm/` of this iteration. Keep.
- Not committed: the run logs, the timing scripts, the binaries and the LiteRT-LM checkout's build
  output.

## Time
Start 02:14, end 12:40 (2026-10-21), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 100 (02:14–02:46 the sources, the code, the scripts and the record generator; 04:44–04:47 one window instead of two; 09:13–09:18 the first window's ids and the second try of the param tensor; 09:28–09:42 the control and the second window's scripts; about 25 of drafts while waiting; 12:19–12:40 the results, these files and the checks) |
| builds | 6 | 4 (238 s of Bazel's elapsed time: the runner 8 s, LiteRT-LM 206 s, the second try and the control 4 to 7 s each) |
| waiting for the machine: quiet windows | 2 | 542 (32,492 s: 23,360 for the first window, about 9,130 for the second) |
| runs and checks | 2 windows | 37 (the two windows' 2,465 s without the builds) |
| rework after a failure | 2 | 5 (a comment block of the runner broken by a text substitution and restored; a wait restarted twice) |
| session not running | 0 | 0 |
| phone in use | 0 | 0 |

The parts overlap: drafts were written while waiting for the machine.
