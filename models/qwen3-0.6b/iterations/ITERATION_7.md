# Iteration 7 — Qwen3-0.6B: the graph as a LiteRT-LM executor backend

## Read first
- LiteRT-LM at `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33`: `runtime/executor/executor_backend_registry.h`
  and its test, `runtime/executor/llm_executor_base.h`, `runtime/core/engine_advanced_impl.cc`
  (`EngineAdvancedImpl::Create`), `runtime/engine/litert_lm_advanced_main.cc`.
- `iterations/ITERATION_2.md` (the KV cache rebind) and `results/quantization.md` (how ids are
  counted).

## Goal (one)
Run the Tensor API graph of `graph/` inside LiteRT-LM: register an executor for it with
`ExecutorBackendRegistry`, build one binary that runs both this executor and LiteRT-LM's own CPU
and GPU executors with the published `Qwen3-0.6B.litertlm`, and compare the greedy ids of all of
them with the PyTorch reference.

## Acceptance
- The executor registered under a backend name, chosen with `--backend` on the command line, and
  the log line of its creator.
- Greedy 16 tokens for the three prompts, ids counted as in `results/quantization.md`: the Tensor
  API executor at fp32 on the CPU and on the GPU, at least 44 of 48 each (iteration 2 measured
  48 of 48 for both with the harness's own host loop). The ids that enter prefill equal those in
  `reference/prompts.json`.
- The same binary (one file, by sha256) runs LiteRT-LM's CPU and GPU executors on the published
  file; their counts are recorded without a bar.
- int8 weights with the int8 lm_head copy run on the CPU and the GPU through the executor and
  produce ids (counts recorded, no bar).
- The benchmark entry (`--benchmark`, forced prefill 1024, decode 256, KV cache 2048) runs to the
  end for the four fp32 forms. No time is reported.
- The graph files change only by additions that are off by default, and the previous runner
  gives the same model bytes and the same ids with its default flags.

## Stop conditions
- LiteRT-LM's engine does not reach a registered creator (the registry only swaps containers or
  resources).
- The executor needs a LiteRT version other than the one LiteRT-LM builds with, or a change in
  LiteRT-LM's core.
- The build fails for three hours of build and fix time.
- The Tensor API executor at CPU fp32 is below 44 of 48 and one hour of comparison against the
  host loop does not find the cause.

## Scope
New: `litert_lm/` (the executor, the registration, a `ModelResourcesRegistry` entry used for the
output match, the BUILD file, the overlay script, a README), `results/litert_lm_registry.md` and
its id files, this file. `graph/qwen3_config.h`, `graph/qwen3_graph.h` and `graph/qwen3_graph.cc`:
one switch, off by default. No file of the LiteRT-LM checkout changes.

## Do not
- Do not change the thresholds in `checks/`.
- Do not report times: the runs of this iteration are not a timing plan.
- Do not commit weights, the `.litertlm` file, generated graphs, logs or binaries.

## Progress
- Registration: `litert_lm/tensor_api_backend.cc` registers `tensor_api_cpu` and `tensor_api_gpu`.
  The model path stays the published `.litertlm`, which LiteRT-LM loads with its own resources
  class; the creator receives the executor settings and those resources, and the executor
  (`litert_lm/tensor_api_executor.cc`) builds the graph from `model.safetensors` in memory. No file
  of the LiteRT-LM checkout changed; the graph code built unchanged against the LiteRT commit
  LiteRT-LM builds with (`9b8c7f77`).
- The published file samples at temperature 0.6 (TOP_P, k 20, p 0.95, from its metadata), and the
  advanced main has no flag to change that. For the output match the model path is a link named
  `*.greedy.litertlm`, which a `ModelResourcesRegistry` entry
  (`litert_lm/greedy_litertlm_resources.cc`) loads with greedy sampler parameters and an
  id-logging tokenizer.
- One binary (sha256 `0195ed42ec32f072f7a1903ee88f58d00027827260946bf61318278a44861860`) ran eight
  forms, greedy 16 tokens for the three prompts, KV cache 2048. Ids equal to the reference before
  the first mismatch, out of 48:
  - Tensor API CPU fp32: 48 (16, 16, 16). GPU fp32: 48 (16, 16, 16). Both pass the bar.
  - Tensor API CPU int8 + int8 lm_head: 48. GPU int8 + int8 lm_head: 48.
  - Tensor API CPU fp32 and CPU int8 + int8 lm_head with LiteRT-LM's XNNPACK flags: 48 each.
  - LiteRT-LM CPU, published file: 31 (5, 16, 10). LiteRT-LM GPU, published file: 48.
  - The prompt ids were the reference ids in every run.
- The benchmark entry (forced prefill 1024, decode 256, KV cache 2048) ran to the end for the four
  fp32 forms: each printed `Prefill Turn 1: Processed 1024 tokens` and
  `Decode Turn 1: Processed 256 tokens`.
- The first build's benchmark log showed that a GPU prefill returned before the GPU finished: the
  1024-token prefill took less time than its arithmetic allows. The session asks every prefill to
  complete, and the executor had ignored that. It now locks one cache output after such a GPU
  prefill. The ids of the four forms that both builds ran are byte-identical.
- The previous runner, rebuilt from the changed graph files, writes the same model bytes as before
  with its default flags, and its CPU fp32 greedy run passes 3/3.
- Details and tables: `results/litert_lm_registry.md`.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (the skill table is unchanged in this
iteration):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Registration:** met. `--backend=tensor_api_cpu` and `--backend=tensor_api_gpu`; the creator's
  log line is in `results/litert_lm_registry.md`.
- **Tensor API fp32, 48 of 48 on the CPU and the GPU:** met (bar: at least 44). The prompt ids equal
  `reference/prompts.json`.
- **Same binary for LiteRT-LM's executors:** met. 31 of 48 on the CPU and 48 of 48 on the GPU, no bar.
- **int8 with the int8 lm_head:** ran on both, 48 of 48 each, no bar.
- **Benchmark entry:** met for the four fp32 forms; no time is reported.
- **Graph files:** one switch, off by default; the previous runner's model bytes and ids are
  unchanged.

## Corrections
- The plan expected to download the published `.litertlm`. The local model cache already held
  that revision; its size and sha256 matched.
- The plan asked for greedy runs of the published file with LiteRT-LM's own executors. The file's
  metadata sets TOP_P with k 20 at temperature 0.6. LiteRT-LM copies it into the session when the
  session sets none, and the advanced main has no flag for it. The greedy ids came through the
  `*.greedy.litertlm` entry (`litert_lm/greedy_litertlm_resources.cc`).
- The engine path calls more of `LlmExecutorBase` than prefill and decode. Its resource manager
  also needs `CreateNewContext`, `RestoreContext(LlmContext)` and `GetProcessedTokens`
  (`runtime/framework/resource_management/resource_manager.cc`), and the step bookkeeping follows
  the pending-token scheme of `runtime/executor/llm_litert_compiled_model_executor.cc`.
- The GPU accelerator of this macOS build is WebGPU, not Metal: LiteRT tries WebGPU before Metal on
  macOS (`litert/runtime/accelerators/gpu_registry.cc`), and both GPU paths loaded it.

## Dead ends
- **Waiting for a free machine by polling the lock.** Measurement jobs of other work held the Mac
  without a gap for 80 minutes, so a build that polled for a free moment never started. A build
  or run now queues for the lock and holds it itself, at most 25 minutes per build slice.
- **An unqualified `Create(...)` inside the executor's member functions.** It names the class's
  own static `Create`, not `litert::tensor::Create`; the calls are qualified.

## Op and delegate observations
- CPU (XNNPACK), the executor's model with three signatures (prefill 128, prefill 1024, decode):
  5718 of 5889 nodes delegated, 85 partitions, the same with LiteRT-LM's XNNPACK flags. Which ops
  stay on the TFLite kernels was not printed: unverified.
- GPU (WebGPU, `libLiteRtWebGpuAccelerator.dylib`): 5889 of 5889 nodes, 3 partitions (one per
  signature), with the GPU precision set to fp32 for both the fp32 and the int8 weights. No compile
  or runtime failure.
- On WebGPU, `CompiledModel::Run` returned before the GPU work finished. A prefill reported less
  time than its arithmetic allows until the executor locked an output buffer after it. A decode
  call reads its logits and was not affected.
- Two LiteRT environments in one process (the engine's and the executor's) each registered the
  WebGPU and CPU accelerators. All runs ended with exit code 0.
- LiteRT-LM's CPU executor on the published file was not run-to-run identical with k 1 (p2, from the
  14th generated token). The Tensor API executor was. The cause was not investigated.
- The published file is 613,722,264 bytes, against 2,384,789,652 bytes for this repo's fp32 graph.
  LiteRT-LM's loader logs a default fp16 activation type for the text decoder on its own executors;
  the precision its GPU path used was not checked.
- Runtime: LiteRT `9b8c7f77` linked into the binary, LiteRT-LM `d78488c4`.

## Environment
- Mac: Apple M4 Max, 128 GB, macOS 27.0 (26A428), Xcode 27.0. No phone.
- LiteRT-LM `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33` (a clean checkout plus `litert_lm/overlay.sh`),
  Bazel 7.6.1 through bazelisk (the checkout's `.bazelversion`), `-c opt` from its `.bazelrc`.
- The previous runner: LiteRT `bea3bc69` as in `ENVIRONMENT.md`.
- Python 3.14.6 for the checks; the `.litertlm` metadata was read with `litert_lm_builder` 0.18.0.
- Every build and run held the Mac's measurement lock itself; no other measurement ran inside.
  The lock was taken 7 times (4 builds, 2 run sessions, 1 check session); waiting for it took
  14,398 s in all, holding it 648 s.

## Files left
- `litert_lm/`: the executor, the registration, the `.greedy.litertlm` entry, the BUILD file, the
  overlay script and the README. Keep.
- `graph/qwen3_config.h`, `graph/qwen3_graph.h` and `graph/qwen3_graph.cc`: the `omit_logits`
  switch. Keep.
- `results/litert_lm_registry.md`, eight `results/litert_lm_ids_<form>.json` and eight
  `results/text_match_litert_lm_<form>.txt`. Keep.
- Not committed: the binary, the logs, the XNNPACK and GPU caches and the serialized models of the
  previous-runner check (deleted after the comparison).

## Time
Start 19:26 (2026-10-19), end 01:40 (2026-10-20), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | about 175 (19:26–21:08 reading LiteRT-LM and the design; 21:08–21:35 the code; about 45 for the records, mostly written while waiting for the lock) |
| builds | 4 | 4.1 (228.7 s, 6.1 s, 5.3 s and 5.5 s; a first attempt was stopped after 10 s) |
| waiting for the machine: the polling wait, then the lock queue | 1 + 7 | 321 (4,842 s; 14,398 s) |
| runs and checks | 2 run sessions, 1 check session | 6.5 (167 s, 220 s, under 1 s) |
| rework after a failure: the GPU prefill wait | 1 | 5 (the code; its build and second run are in the rows above) |
| phone in use | 0 | 0 |
