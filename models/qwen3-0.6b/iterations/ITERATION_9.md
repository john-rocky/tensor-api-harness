# Iteration 9 — Qwen3-0.6B: speed through the same LiteRT-LM binary as the published .litertlm

## Read first
- `iterations/ITERATION_7.md` and `results/litert_lm_registry.md`: the executor registered with
  LiteRT-LM's `ExecutorBackendRegistry`, the binary, the output match through LiteRT-LM, and the
  benchmark entry that ran to the end without a timing plan.
- LiteRT-LM at `d78488c4`: the benchmark flags (`runtime/engine/shared_flags.cc`), the CPU executor's
  XNNPACK flags (`runtime/executor/litert_compiled_model_executor_utils.cc`, `SetCpuOptions`), the GPU
  compilation options, precision and cache file names (`runtime/executor/llm_executor_settings_utils.cc`).

## Goal (one)
One speed table for the Apple M4 Max CPU and GPU: the Tensor API executor (fp32 and int8 weights) and
LiteRT-LM's own executors on the published `Qwen3-0.6B.litertlm`, all run by the binary of iteration
7, with LiteRT-LM's benchmark entry (forced prefill 1024, decode 256, KV cache 2048), timed inside
one quiet window with the forms alternating.

The forms, written before the runs:
- CPU (XNNPACK, 4 threads): the published file on LiteRT-LM's CPU executor; the Tensor API executor
  with fp32 weights and with int8 per-channel weights + the int8 lm_head copy, each once with
  LiteRT-LM's XNNPACK flags (`--tensor_api_litert_lm_xnnpack_flags`) and once with LiteRT's
  defaults.
- GPU (WebGPU): the published file on LiteRT-LM's GPU executor with its default fp16 activations
  and with `--force_f32`; the Tensor API executor with fp32 and with int8 + int8 lm_head weights, fp32
  precision.
- Extra, after these: int8 weights with the fp32 lm_head on the CPU and on the GPU, and fp32
  weights at GPU precision fp16 (greedy ids first).
- The comparisons read first: on the CPU, the published file against int8 + int8 lm_head with the
  same XNNPACK flags; on the GPU, the published file with fp32 activations against int8 + int8
  lm_head at fp32, with the file's default fp16 next to them.

## Acceptance
- The same binary (path and sha256) before and after the runs.
- For every main form: prefill and decode tok/s, time to first token and peak memory footprint as
  median [min–max] (n ≥ 3), timed inside the window; the first run of each form discarded.
- The conditions written down: 1024 / 256 / 2048, the thread count, the GPU API and its library, the
  file's own sampler.
- The output match of every Tensor API form, at least 44 of 48, from iteration 7 (from this
  iteration for the three extra forms).
- A Tensor API form and a LiteRT-LM form on the same accelerator within 10 % of each other in decode
  get 8 measured runs each before their order is stated.
- `tools/iteration_close_check.py` passes; a `## Time` section.

## Stop conditions
- The binary's sha256 differs from iteration 7's.
- A main form does not run to the end with 1024 / 256 / 2048.
- The GPU forms do not all run on one GPU API.
- Waiting for quiet windows adds up to more than three hours before the main forms are timed.
- Less than 30 GB free on the disk.

## Scope
New files in `results/` (`litert_lm_speed.md`, `litert_lm_speed.json`) and this file. The binary,
the executor and the graphs are those of iteration 7, unchanged.

## Do not
- Do not rebuild the binary.
- Do not change `README.md`, the skill tables, `checks/` or `tools/`.
- Do not time a form outside the quiet window, and do not publish a number that was not measured.

## Progress
- Source read at `d78488c4`: LiteRT-LM's CPU executor runs XNNPACK with `number_of_threads` (4 by
  default) and the flags default | `DYNAMIC_FULLY_CONNECTED` | `ENABLE_LATEST_OPERATORS`; YNNPACK is
  off by default; `--force_f32` sets the activations to fp32, anything else gives the GPU fp16; the
  GPU program and weight cache file names carry no precision, so the `--force_f32` form got a cache
  directory of its own; LiteRT-LM's GPU executor samples on the GPU by default.
- One quiet window, 2026-10-20 01:54:02 to 02:12:11 JST (1,089 s), taken without waiting. Its first
  three minutes went to runs that stopped at load (Dead ends) and to the greedy ids of the three extra
  forms, which do not go through `/usr/bin/time` and ran normally. From 01:57:17 the corrected
  script ran 58 runs: all exited 0 with 1024 prefill and 256 decode tokens processed; 12 were the
  discarded first runs, 46 were measured.
- The binary's sha256 was `0195ed42…1860` before and after the runs; `pmset -g therm` recorded no
  warning before or after.
- Output match of the extra forms (greedy, 16 tokens, 3 prompts): int8 weights with the fp32 lm_head
  48 of 48 on the CPU (LiteRT-LM's XNNPACK flags) and on the GPU; fp32 weights at GPU precision fp16
  0 of 48.
- The comparisons read first (medians, `results/litert_lm_speed.md`):
  - CPU: the published file decodes at 29.19 tok/s and prefills at 654.25 tok/s on LiteRT-LM's
    executor; the Tensor API executor with int8 + int8 lm_head weights at 48.62 and 1220.91 tok/s.
    Memory footprint 1198 MB against 5083 MB.
  - GPU: the published file with fp32 activations decodes at 101.75 and prefills at 2312.60 tok/s;
    the Tensor API executor with int8 + int8 lm_head weights at 78.78 and 4059.90 tok/s. With fp16
    activations the published file decodes at 142.75 tok/s. Memory footprint 2423 (fp32) and 1434 MB
    (fp16) against 18099 MB.
- The 10 % rule: after the four turns, only `tapi_cpu` against `lm_cpu` was within 10 % in decode
  (1.094). Both got alternating runs up to 8 measured each: 30.36 against 29.19 tok/s.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0; the skill table is unchanged):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Same binary:** met, sha256 `0195ed42ec32f072f7a1903ee88f58d00027827260946bf61318278a44861860` before
  and after.
- **Table:** met for the nine main forms (3 or 8 measured runs each) and the three extra forms (3),
  all inside the one window.
- **Conditions:** met; every CPU run logged 4 threads, every GPU run WebGPU only.
- **Output match:** met; every Tensor API form passes except the fp16 GPU form (0 of 48, kept in the
  table).
- **10 % rule:** met; one pair got 8 runs each.
- **Close check:** `python3 tools/iteration_close_check.py models/qwen3-0.6b/iterations/ITERATION_9.md`:
  `PASS iteration close: models/qwen3-0.6b/iterations/ITERATION_9.md`.

## Corrections
None to earlier iterations. The forms with LiteRT's default XNNPACK flags share the flags of the
harness's own runner but not its thread count: 4 threads here, no thread pool in iterations 1 to 8.

## Dead ends
- **`/usr/bin/time -l` in front of the binary:** macOS removes the `DYLD_*` variables when it starts
  a protected system binary such as `/usr/bin/time`, so the binary no longer found LiteRT-LM's
  prebuilt libraries (`Library not loaded: @rpath/libGemmaModelConstraintProvider.dylib`, exit 134
  within a second). `/usr/bin/time -l env DYLD_LIBRARY_PATH=<prebuilt/macos_arm64> <binary>` sets the
  variable again after `time` starts; `time` still reports the binary's process. The first three
  minutes of the window went to runs that stopped at load; every timed run in the results used the
  second form.

## Op and delegate observations
- GPU API: every GPU run of every form logged `Initializing WebGPU-based API` (LiteRT-LM's executor
  twice per run, from its serialized cache; the Tensor API executor three times, once per
  signature, from the graph) and the adapter `Apple M4 Max, arch=metal-3, vendor=apple,
  backend=Metal` (Dawn on Metal). The only GPU accelerator loaded was
  `libLiteRtWebGpuAccelerator.dylib`; no Metal, OpenCL or other GPU API line.
- LiteRT-LM's GPU executor with `--force_f32` logged `activation_data_type: FLOAT32` and wrote its own
  program cache (4,367,440 bytes against 4,127,016 for the default fp16 activations; the converted
  weights 596,008,376 bytes in both).
- WebGPU at precision fp16 with this repo's graphs: 0 of 48 (each prompt repeats one id from its
  first generated token: 624, 5183 and 0). Metal fp16 failed with the same first wrong id for
  each prompt in iteration 2 (`results/text_match_metal_fp16.txt`; there block_27 was `nan`). No
  dumps were taken on WebGPU, so where it breaks there is unverified.
- int8 decoder weights with the fp32 lm_head on the CPU, through LiteRT-LM with its XNNPACK flags
  and 4 threads: 48 of 48. Iteration 3's harness runner gave 41 of 48 for the same weights (LiteRT's
  default flags, no thread pool, KV cache 64, the first generated token from the prefill logits).
  Which difference matters was not isolated.
- The decode graph attends over every entry of the KV cache at each step, with a mask
  (`graph/qwen3_graph.cc`, the attention mask is `[1, 1, seq, max_seq]`); here `max_seq` is 2048.
  How much of the decode time that is was not measured.
- Memory (medians of the peak memory footprint): the Tensor API process peaks at 4.8 to 5.4 GB with
  int8 weights and 11.1 GB with fp32 weights on the CPU, 17.5 to 25.8 GB on the GPU (maximum
  resident set size 12.5 to 18.2 GB there); LiteRT-LM's processes at 1.2 to 2.4 GB. Where it goes
  was not measured.
- Init: the Tensor API executor builds and compiles its graphs at every start, 0.90 to 1.95 s on
  the CPU and 8.2 to 12.9 s on the GPU; LiteRT-LM's executors 0.11 to 0.13 s from their caches.
- Threads: `number_of_threads: 4` in every CPU run's settings, and the Tensor API executor's line
  `CPU threads 4` in each of its CPU runs.
- Runtime: LiteRT `9b8c7f77` linked into the binary, LiteRT-LM `d78488c4`.

## Environment
- Mac: Apple M4 Max (12 performance and 4 efficiency cores), 128 GB, macOS 27.0 (26A428). No phone.
- The binary of iteration 7, unchanged (no build in this iteration); the GPU libraries from the
  LiteRT-LM checkout's `prebuilt/macos_arm64`.
- Python 3.14.6 for reading the logs and for the checks.
- One quiet window for every run (the lock taken once, after 0 s of waiting, held 1,089 s); the
  machine's other applications kept running (load average 3.9 before and 6.1 after).

## Files left
- `results/litert_lm_speed.md` and `results/litert_lm_speed.json` (every run's values, the flags of
  each form, the ids of the three extra forms). Keep.
- Not committed: the run logs, the per-run values with local paths, and the GPU cache of the
  `--force_f32` form.

## Time
Start 01:41, end 02:19 (2026-10-20), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 36 (01:41–01:54 the source, the scripts and a rehearsal against a stub binary; 01:56–02:12 the fix and drafts of the results, written while the window ran; 02:12–02:19 the results, this file and the checks) |
| builds | 0 | 0 |
| waiting for the machine: quiet windows | 1 | 0 (0 s) |
| runs and checks | 1 window | 18 (1,089 s: 58 timed runs, 9 greedy runs, and the runs that stopped at load) |
| rework after a failure: `/usr/bin/time` and `DYLD_LIBRARY_PATH` | 1 | 3 (the window's first part, 01:54–01:57) |
| phone in use | 0 | 0 |
