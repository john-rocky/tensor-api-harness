# Iteration 2 — Qwen3-0.6B: KV chain, Metal delegation, GPU precision and buffer storage on the Mac

## Read first
- `skill_table.md` at the repo root, then `models/qwen3-0.6b/skill_table.md`.
- `iterations/ITERATION_1.md` (this model): the graphs, the loader, the reference and the bars this
  iteration builds on.
- LiteRT's `tensor/runners/litert/litert_dynamic_runner.h` (`SetInput` with a `TensorHandle`,
  `SetInputBuffer`, `GetOutput`), `litert/cc/litert_compiled_model.h` (`GetDelegationMetrics`),
  `litert/cc/options/litert_gpu_options.h` (precision, buffer storage type) and
  `litert/cc/litert_environment_options.h` (minimum logger severity).

## Goal (one)
Rows 7, 9, 10 and 11 on the Mac: chained decode with the KV cache compared with a stateless
reference (CPU fp32), then the same graphs on the Metal delegate at the real shapes, with the
delegation report, fp32 and fp16 output match, and buffer storage against the default storage.
Rows 8, 12, 13 and 14 stay `not yet`.

## Acceptance
- Row 7: for each of the three prompts, the prefill logits at the last prompt position and the
  logits of 15 chained decode steps (16 tensors) pass `checks/block_match.py` against one stateless
  fp32 forward over the prompt and its 16 reference ids (no cache; row i = the logits at position
  S - 1 + i). The bar is written into `reference/manifest_chain_p*.json` when the reference is
  dumped: cosine >= 0.9999 and max abs difference <= max(5e-2, 1e-3 × max|ref|). The way the cache
  outputs are bound to the next call's inputs is named with file and line numbers.
- Row 10, fp32: Metal fp32 with buffer storage passes `checks/block_match.py` 31/31 for all three
  prompts with the iteration 1 manifests unchanged, and `checks/text_match.py` passes 3/3. fp16:
  the block match and text match results are recorded, pass or fail.
- Rows 9 and 11: `results/delegation_metal.md` gives, for each form run (precision × storage), the
  total node count and the delegated node count and partition count per accelerator; the ops that
  fell back to the CPU (or `unverified` when the runtime does not print them); a check that the GPU
  output is not byte-identical to the CPU output; any compile failure text. Default storage and
  buffer storage are compared at the real shapes.
- `tools/done_check.py` lists exactly rows 8, 12, 13 and 14; `tools/iteration_close_check.py` passes.

## Stop conditions
- No idle machine for the runs within seven hours of waiting.
- Eight hours from the start without rows 7 and 9 filled: fill what has evidence and close.
- The Metal compile does not finish within 20 minutes twice for the same form: do not try a third
  time; record the form and the last log line and go on with the other forms. If every form fails,
  rows 9 to 11 become `blocked` with the failure text.

## Scope
`models/qwen3-0.6b/` only: `graph/` (host loop and flags), `reference/dump_reference.py` and the
new chain manifests, new files in `results/`, this file, rows 7, 9, 10 and 11 of `skill_table.md`
and an added section in `ENVIRONMENT.md`. Everything else in the repository is read only. Weights,
generated `.tflite` files and dumps are not committed.

## Do not
- Do not write `tried` without an evidence path that exists.
- Do not turn a `not yet` row into `not applicable` or `blocked` to make the done check pass; `blocked` needs the cause after the colon.
- Do not change the thresholds in `checks/` during the iteration; a new bar is a new iteration.
- Do not commit weights, generated graphs or dumps.
- Do not change the attention form or the weight precision of the graph (rows 8 and 12 come later).
- No timing numbers: the speed rows come later, so no millisecond number is recorded.

## Progress
- Row 7: `reference/dump_reference.py chain` ran one stateless fp32 forward per prompt over the prompt
  and its 16 reference ids and wrote 16 logits rows and `reference/manifest_chain_p*.json`; every
  row's argmax equals the next reference id. The graph's prefill and 15 chained decode steps
  (`--dump_chain`, CPU fp32) pass `BLOCK MATCH PASS 16/16` for p0, p1 and p2
  (`results/kv_chain_p*.txt`). The largest row max abs difference is 1.1e-4. The rebind form is
  named with line numbers in `results/kv_chain.md`: the output handle goes to `SetInput`, which binds
  a `Duplicate()` of the output `TensorBuffer` with `SetInputBuffer` and copies nothing.
- Row 9: on Metal, all 3994 nodes are delegated in 2 partitions (one per signature) for every
  form: fp32 and fp16, buffer, default and texture2d storage. That includes `EMBEDDING_LOOKUP` and the
  `DYNAMIC_UPDATE_SLICE` cache writes. On the CPU, XNNPACK takes 3880 of the 3994 nodes in 58
  partitions. Every Metal fp32 output differs in bytes from the CPU output, by at most 5.1e-5 at the
  last prompt position (`results/delegation_metal.md`).
- Row 10, fp32: greedy text match `TEXT MATCH PASS 3/3` (`results/text_match_metal_fp32.txt`) and the
  Metal chain `BLOCK MATCH PASS 16/16` for each prompt (`results/kv_chain_metal_fp32_p*.txt`). The
  per-block match is 30 of 31 for each prompt, not 31 of 31: the embedding, all 28 blocks and the
  last-position logits pass. `final_norm` fails on every prompt (`results/block_match_metal_fp32_p*.txt`).
  Its exported values are mostly zeros with stray values (`nan` for p0), while the logits computed
  from it pass. So the acceptance line "31/31 for all three prompts" is not met. The bar is
  unchanged and the failure is recorded in `results/delegation_metal.md`; its cause is not
  established.
- Row 10, fp16: recorded. The per-block difference grows from block to block, block_27 is `nan` for
  p0 and p2, the last-position logits have cosine 0.598 / 0.426 / 0.000, and greedy decode fails at
  the first generated token of every prompt (`results/text_match_metal_fp16.txt`,
  `results/block_match_metal_fp16_p*.txt`).
- Row 11: default storage and texture2d storage give the same delegation as buffer storage and
  byte-identical outputs (48 of 48 chain rows for default, 16 of 16 for texture2d on p0);
  `results/text_match_metal_fp32_default.txt` 3/3. Only the shapes this model uses here were run
  (KV cache length 64).

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 1):

```
models/qwen3-0.6b/skill_table.md: tried 11, not applicable 0, blocked 0, not yet 4
NOT DONE models/qwen3-0.6b/skill_table.md: 4 rows not yet
  #8 Attention form
  #12 Quantization
  #13 Speed floor
  #14 Phone run
```

Against the acceptance list: row 7 met; rows 9 and 11 met; row 10 fp32 met for the text match and
the chain but not for the per-block match (30/31 per prompt, `final_norm`); row 10 fp16 recorded.

## Corrections
- The runner's constructor from a `CompiledModel` is private, so the delegation report compiles the
  model a second time with the same options (`graph/qwen3_main.cc`, `PrintDelegationReport`). Each
  run therefore compiles every prompt's model twice.
- The logger severity functions are declared in `litert/c/internal/litert_logging.h`, not in
  `litert/c/litert_logging.h`. The binary sets the severity through the environment option
  `kMinLoggerSeverity` (`litert/cc/litert_environment_options.h`), which calls
  `LiteRtSetMinLoggerSeverity` on the default logger.
- With the handle rebind, the decode signature's cache input and cache output are the same buffer
  from the second decode step on (`Duplicate()` adds a reference to one handle,
  `litert/cc/litert_tensor_buffer.h`). `DynamicUpdateSlice` therefore updates the cache in place. It
  matched the stateless reference on the CPU and on Metal.

## Dead ends
- None. One build, and every run finished on its first attempt (no hang, no compile failure). The
  `--rebind=copy` diagnostic was written for the case where the Metal fp32 text match failed. It was
  not needed.

## Op and delegate observations
- Metal (prebuilt `libLiteRtMetalAccelerator.dylib`): 3994 / 3994 nodes, 2 partitions, fully
  accelerated, for fp32 and fp16 and for buffer, default and texture2d storage. No op fell back. No
  compile failure. At `--log_level=verbose` no line names an op.
- CPU: XNNPACK 3880 / 3994 nodes in 58 partitions. The 114 nodes on built-in kernels are not named
  by the runtime: unverified.
- Metal fp32 against CPU fp32: last-position logits max abs difference 4.5e-5 / 3.5e-5 / 5.1e-5;
  never byte-identical. Chain against the stateless reference: CPU max 1.1e-4 (median 4.0e-5),
  Metal fp32 max 1.0e-4 (median 4.6e-5) over 48 rows.
- Metal fp32 `final_norm` exported as a prefill output reads back as mostly zeros. It is the only
  dumped tensor consumed by the lm_head `FULLY_CONNECTED`; the logits are correct. Cause not
  established.
- Metal fp16: prefill argmax differs at every position (27 of 27 over the three prompts). Greedy
  decode repeats one id. The residual stream reaches about 6,500 in magnitude from block_02 on.
  Whether an fp16 overflow causes the `nan` at block_27 is not established.
- Storage: default, buffer and texture2d give byte-identical outputs at KV cache length 64.

## Environment
- `ENVIRONMENT.md`, section "Iteration 2": the same LiteRT commit, toolchain, machine and weights as
  iteration 1. The Metal dylib's sha256 is recorded, the run command form and every flag used. No
  phone. No timing was taken. Runs started only once no other job was timing on the machine.

## Files left
- `graph/qwen3_main.cc`, `graph/BUILD`: the new flags (`--dump_chain`, `--delegation_report`,
  `--log_level`, `--rebind`). Keep.
- `reference/dump_reference.py` (`chain` subcommand) and `reference/manifest_chain_p{0,1,2}.json`.
  Keep.
- `results/`: `kv_chain.md`, `kv_chain_p*.txt`, `kv_chain_metal_fp32_p*.txt`, `delegation_metal.md`,
  `block_match_metal_fp32_p*.txt`, `block_match_metal_fp16_p*.txt`, `text_match_metal_fp32.txt`,
  `text_match_metal_fp32_default.txt`, `text_match_metal_fp16.txt`. Keep.
- Not committed and deletable: the chain dumps (reference and graph, 16 × 0.6 MB per prompt and
  side), the Metal block dumps and the serialized `.tflite` files, all outside the repository. The
  `.tflite` files were deleted after each run.

## Time
Start 17:29, end 17:41 (JST, 2026-10-15).

| part | count | minutes |
|---|---|---|
| coding and reading | — | 8 (reading the LiteRT headers, the code, the reference script, the run script, the write-ups) |
| builds | 1 | under 1 (4.6 s, with the earlier build's cache warm) |
| waiting for the machine (another measurement was running) | 0 | 0 (the machine was idle at the first check) |
| runs and checks | 1 script, 8 binary runs | 3 (17:35 to 17:38, build included) |
| rework after a failure | 0 | 0 (no rebuild; nothing was fixed and rerun) |
| phone in use | 0 | 0 |
