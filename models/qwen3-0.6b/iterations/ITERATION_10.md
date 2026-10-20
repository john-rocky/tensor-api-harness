# Iteration 10 — Qwen3-0.6B: where the LiteRT-LM backend's GPU decode, memory and init go

## Read first
- `iterations/ITERATION_9.md` and `results/litert_lm_speed.md`: the speed table through LiteRT-LM's
  benchmark entry (forced prefill 1024, decode 256, KV cache 2048). There the Tensor API executor's
  GPU decode (int8 weights, fp32 precision) was slower than LiteRT-LM's GPU executor on the
  published file with fp32 activations, its peak memory footprint was 4 to 13 times LiteRT-LM's,
  and its GPU init took 8 to 13 s.
- `litert_lm/`: how the executor builds, compiles and runs its graphs (one compiled model with three
  signatures, the KV caches as signature inputs and outputs, the logits read back every step).
- LiteRT-LM at `d78488c4`: the GPU and CPU compilation options of its own executors
  (`runtime/executor/llm_executor_settings_utils.cc`, `runtime/executor/litert_compiled_model_executor_utils.cc`).
- LiteRT at `9b8c7f77`: `litert/cc/options/litert_gpu_options.h` (constant tensor sharing, the
  serialization of programs and weights), `litert/cc/options/litert_cpu_options.h` (the XNNPACK weight
  cache) and `litert/cc/litert_compiled_model.h` (a model from a buffer is read in place, a model from a
  file is mapped).

## Goal (one)
Measure where a GPU decode step, the process's memory and the GPU init of the Tensor API executor go
inside LiteRT-LM, try the levers below, and give the fastest and the smallest forms that pass the
output match, in one binary and one quiet window together with LiteRT-LM's own executors on the
published file.

The levers, written before the runs (all flags off by default):
- L1, graph forms: the attention core as the `odml.scaled_dot_product_attention` composite, every
  RMSNorm as the `odml.rms_norm` composite, and both (iteration 3's graph options), on the GPU and on
  the CPU (the CPU attention composite failed in XNNPACK's reshape with a KV cache of 64 in iteration
  3, so it was tried once with 2048).
- L2, memory: the footprint after each stage first; then the serialized model held as a mapped file
  instead of the executor's own buffer, shared constant tensors between the signatures on the GPU,
  and an XNNPACK weight cache file on the CPU.
- L3, the parts of a decode step: writing the step inputs, `Run`, reading the logits back, and the
  time between two calls (LiteRT-LM's sampler and session), each step timed inside the executor.
- L4, init: the GPU program cache in the cache directory, and the serialized model file reused
  across starts.
- L5, the other compile options of LiteRT-LM's GPU executor: the KV caches as external tensor
  patterns, its remaining options as one set, and command buffers prepared two steps ahead.

## Acceptance
- Every new form: greedy ids through LiteRT-LM (3 prompts × 16 tokens) equal to the PyTorch reference
  for at least 44 of 48. Forms under the bar are measured and listed, not used for the comparisons.
- Speed as in iteration 9 (one run per process, the first run of each form discarded, at least three
  measured, the forms in turn, 8 runs each for the compared pairs within 10 %), with LiteRT-LM's own
  executors and iteration 9's fastest passing Tensor API forms in the same window.
- The breakdowns as numbers: the decode step of the GPU and CPU int8 forms, the memory footprint at
  each stage, the init phases.
- The same binary (path and sha256) before and after the runs; with every new flag off the binary
  gives iteration 7's ids.
- `tools/iteration_close_check.py` passes; a `## Time` section.

## Stop conditions
- With the default flags the ids differ from iteration 7's.
- The build does not pass within two hours of building and fixing.
- Waiting for quiet windows adds up to more than four hours before the speed runs.
- Less than 30 GB free on the disk.

## Scope
`litert_lm/` (the executor, the backend flags, the build file, its README), new files in `results/`
(`litert_lm_levers.md`, `litert_lm_levers.json`) and this file. The graph sources are unchanged; the
executor passes the graph options that iteration 3 added.

## Do not
- Do not change `README.md`, the skill tables, `checks/` or `tools/`.
- Do not time a form outside the quiet window, and do not publish a number that was not measured.
- Do not change the output-match bar.

## Progress
- Source read before the code: LiteRT-LM's GPU executor sets more than twenty GPU compile options
  (shared constants, the program and weight caches, the KV caches as external tensors, command
  buffers prepared ahead, an active wait in its benchmark, and more), the Tensor API executor two
  (precision and buffer storage); LiteRT-LM's CPU executor uses an XNNPACK weight cache file. These became the levers
  L2, L4 and L5, written down before any run with the rule for the combined forms.
- The published `.litertlm`'s graph, read from its TFLite section: every RMSNorm is the
  `odml.rms_norm` composite, the attention is raw operators, the weights are int8, the KV caches hold
  4096 entries and the only prefill signature takes 128 tokens (`results/litert_lm_levers.md`).
- The first quiet window (2026-10-20 04:46:11 to 04:58:18 JST, after 6,415 s of waiting) built the
  binary at the first attempt (12 s), confirmed iteration 7's ids with every new flag off, ran the
  greedy ids of the 14 new forms and the breakdown runs, held one `G4` run for `footprint`, and chose
  the combined forms: `G4best` (shared constants, program cache, model file reused, LiteRT-LM's GPU
  options) and `C3best` (XNNPACK weight cache, model file reused).
- Output match of the new forms (3 prompts × 16 tokens): 48 of 48 for the 11 that ran and for both
  combined forms; `C3a`, `G4k` and `G4kr` stopped at their first run (below).
- The second quiet window (07:57:22 to 08:19:14 JST, after 10,743 s of waiting) timed 18 forms; the
  19th, `C5`, stopped at start (Dead ends). Two pairs within 10 % in decode got 8 runs each.
- The comparisons (medians, `results/litert_lm_levers.md`):
  - CPU: the published file decodes at 31.48 tok/s on LiteRT-LM's executor, the Tensor API executor
    with the XNNPACK weight cache and the reused model file at 46.25; peak memory footprint 1195
    against 1966 MB (4766 MB before the levers); init 0.11 against 0.14 s (1.84 s before).
  - GPU: the published file with fp32 activations decodes at 102.07 tok/s, the fastest passing Tensor
    API forms at 79.23 to 80.24 (their order not established); with every lever the Tensor API
    process peaks at 8083 MB (18190 before, 2421 for the published file) and inits in 0.78 s (10.69 s
    before, 0.13 s for the published file).
  - A GPU decode step of this repo's graph is about 12.4 ms of GPU work; the logits readback (0.3 ms)
    and LiteRT-LM's CPU sampler (0.06 ms) are 2 to 3 % of it. The levers that change memory and init
    do not change it, the norm composite does not either, and the attention composite makes it
    slower (59.38 tok/s).

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0; the skill table is unchanged):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Output match:** met for every form that ran (48 of 48); three forms could not run (Dead ends).
- **Speed:** met for 18 forms in one window with LiteRT-LM's executors and iteration 9's `C3` and
  `G4`; not met for `C5` (no measured run).
- **Breakdowns:** met: the decode step of the GPU and CPU int8 forms, the memory after each stage, the
  init phases, all as numbers from the timed window.
- **Same binary:** met, sha256 `3f8e5ad9a7d5ec18af1e6bca2739467ed9be35ff87627032b7d96dc61f606f76`
  before and after both windows; with every new flag off, iteration 7's ids.
- **Close check:** `python3 tools/iteration_close_check.py models/qwen3-0.6b/iterations/ITERATION_10.md`:
  `PASS iteration close: models/qwen3-0.6b/iterations/ITERATION_10.md`.

## Corrections
- The lever list allowed for a compiled model that keeps its own copy of the model, so that the
  executor could free its buffer after the compile. It does not: a model made from a buffer reads it
  in place (LiteRT `litert/core/util/flatbuffer_tools.cc`, `MakeAllocation`), so the lever became a
  model file that the compiled model maps.
- The lever list allowed for a second compile in the executor. The default path compiles once
  (`litert_lm/tensor_api_executor.cc`); only `--tensor_api_delegation_report` compiles again.
- Iteration 9 left open where the GPU decode time goes; the readback and the sampler were candidates.
  They take 2 to 3 % of a step; the step is the GPU's work on this graph.

## Dead ends
- **The KV caches as external tensor patterns** (`--tensor_api_gpu_external_caches=stock` and
  `runner`): the first decode stops with `INVALID_ARGUMENT: Input buffer type or shape mismatch`
  because `LitertDynamicRunner::SetInputBuffer` refuses the previous call's cache output as the next
  input under these options. Passing the caches some other way was not tried.
- **The CPU attention composite with a KV cache of 2048:** `XNNPack delegate failed to reshape
  runtime`, `Node number 1839 (TfLiteXNNPackDelegate) failed to prepare.`, as with 64 in iteration 3.
  One try, as written before the runs.
- **A timing script that does not create the cache directories:** `C5` had not run in the first
  window, so its `--cache_dir` did not exist, and LiteRT-LM stopped at start with `Cache directory
  does not exist or is not writable`. Create every form's cache directory before the first run.
- **Command buffers prepared two steps ahead:** no faster than without (median steps 12.78 against
  12.70 ms), and two of three runs had long steps.

## Op and delegate observations
- GPU (WebGPU, every GPU run; no other GPU API line): every Tensor API form is delegated whole, three
  partitions (one per signature): 5889 nodes raw, 6222 with the norm composite, 6299 with the attention
  composite, 6632 with both. With the program cache every compile logs `Initializing WebGPU-based API
  from serialized data.`; the cache files are 1,874,269,264 bytes (programs) and 600,020,960 bytes
  (the shared weights) for `G4cp`.
- CPU (XNNPACK): 5718 of 5889 nodes delegated in 85 partitions raw; with the norm composite 5718 of
  6222 in 669 partitions (the composite is not taken by XNNPACK); with the attention composite 6046 of
  6299 in 167 partitions, after which the preparation fails. The XNNPACK weight cache file is
  601,946,856 bytes.
- With LiteRT-LM's GPU options, which wait actively for each run in the benchmark, `Run` takes the
  GPU's work (12.4 ms); without them it returns after enqueueing (1.8 ms) and the first `Lock` of the
  logits waits (10.7 ms).
- Runtime: LiteRT `9b8c7f77` linked into the binary, LiteRT-LM `d78488c4`.

## Environment
- Mac: Apple M4 Max (12 performance and 4 efficiency cores), 128 GB, macOS 27.0 (26A428). No phone.
- Build: Bazel 7.6.1, Xcode 27.0, one build (12 s) in the first window.
- Python 3.14.6 for the logs and the checks; the published graph was read with the TFLite schema
  module of `ai-edge-litert-nightly` 2.4.0.dev20261018.
- Two quiet windows (the lock taken after 6,415 s and 10,743 s of waiting, held 726 s and 1,311 s);
  the machine's other applications kept running.

## Files left
- `results/litert_lm_levers.md` and `results/litert_lm_levers.json` (every timed run with its report
  values, the first window's breakdown runs, the `footprint` categories, the published graph's
  operator counts). Keep.
- `litert_lm/`: the executor, the backend flags, the build file and the README of this iteration.
  Keep.
- Not committed: the run logs, the cache files (model files, GPU program and weight caches, the
  XNNPACK weight cache; 13 GB), and the scripts that ran the windows.

## Time
Start 02:25, end 10:31 (2026-10-20), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 69 (02:25–02:59 the source, the code, the scripts and a rehearsal against a stub binary; 02:59–03:10 drafts while waiting; 04:58–05:05 the first window's results; 10:14–10:31 the results, this file and the checks) |
| builds | 1 | 0.2 (12 s, in the first window) |
| waiting for the machine: quiet windows | 2 | 286 (6,415 s for the first, 10,743 s for the second) |
| runs and checks | 2 windows | 34 (726 s and 1,311 s) |
| rework after a failure | 0 | 0 (the form whose first run stopped was not re-run) |
| session not running | 1 | 309 (05:05–10:14; the second window's wait and runs happened in this time, through the script that waited for it) |
| phone in use | 0 | 0 |
