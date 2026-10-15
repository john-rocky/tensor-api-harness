# Iteration 3 — Qwen3-0.6B: attention and norm forms, int8 and int4 weights, and the speed floor on the Mac

## Read first
- `skill_table.md` at the repo root, then `models/qwen3-0.6b/skill_table.md`.
- `iterations/ITERATION_1.md` and `iterations/ITERATION_2.md` (this model): the graphs, the loader, the
  bars, the Metal delegation report and the `final_norm` export that failed on Metal fp32.
- LiteRT's `tensor/arithmetic.h` (`StableHLOComposite`, `Tile`), `tensor/internal/graph.h`
  (`PerChannelAffineQuantization`, `BlockwiseQuantization`), `tensor/datatypes.h` (`Type::kI8`,
  `Type::kI4`, `fp16_t`), `litert/cc/options/litert_gpu_options.h`
  (`EnableAllowSrcQuantizedFcConvOps`) and `checks/speed_floor.py`.

## Goal (one)
Rows 8, 12 and 13 on the Mac: the attention core and the RMSNorms as raw ops and as the
`odml.scaled_dot_product_attention` and `odml.rms_norm` composites, the decoder weights as fp32, int8
per-channel and int4 blockwise-32, and the decode time of every form against the CPU fp32 floor; plus,
for row 10, one check of the Metal fp32 `final_norm` export in a form where it is not the lm_head's
direct input. Row 14 stays `not yet`.

## Acceptance
- Row 8: `results/attention_forms.md` has one table of form × backend (delegated nodes and partitions,
  text match, block match where run, time) and names the chosen form per backend. On the CPU the
  `odml.rms_norm` form is byte-identical to the raw form (if not, the difference is given in numbers).
- Row 12: `results/quantization.md` gives, for int8 and int4 on the CPU (and on Metal for each form whose
  compile finishes), the number of generated ids that equal the reference (3 prompts × 16 tokens = 48,
  counted up to each prompt's first mismatch), the weight bytes, and "passes" (at least 44 of 48) or
  "fails". The bar is the one written before iteration 2 and is not changed.
- Row 13: `results/timing.json` holds every timed form in the form `checks/speed_floor.py` reads, with
  the floor = the CPU fp32 raw-op form's decode ms per token; `results/speed_floor.txt` is the check's
  output with its exit code. Every number in it was measured while no other measurement ran on the
  machine; nothing timed outside that window is written into `results/`.
- Row 10 (extra): `results/block_match_metal_fp32_identity_p{0,1,2}.txt` exist, pass or fail.
- `tools/done_check.py` lists exactly row 14; `tools/iteration_close_check.py` passes; this file has a
  `## Time` section.

## Stop conditions
- No idle machine for the match runs within seven hours of waiting, or more than five hours of waiting
  in total before the timing has run.
- Eight hours from the start without row 13: fill what has evidence and close.
- The Metal compile does not finish within 20 minutes twice for the same form: do not try a third time;
  record the form and go on.
- A composite's decomposition needs an op the Tensor API does not have.

## Scope
`models/qwen3-0.6b/` only: `graph/` (graph forms, loader quantization, timing flags), new files in
`results/`, this file, rows 8, 10, 12 and 13 of `skill_table.md` and an added section in
`ENVIRONMENT.md`. Everything else in the repository is read only. Weights, generated `.tflite` files
and dumps are not committed. With every flag at its default, the graph is the one iteration 2 ran.

## Do not
- Do not write `tried` without an evidence path that exists.
- Do not turn a `not yet` row into `not applicable` or `blocked` to make the done check pass; `blocked` needs the cause after the colon.
- Do not change the thresholds in `checks/` during the iteration; a new bar is a new iteration.
- Do not commit weights, generated graphs or dumps.
- Timing only inside the machine's quiet window; report the median, keep every measured form.

## Progress
- Row 8: the `odml.rms_norm` form (`--norms=composite`) on the CPU is byte-identical to the raw form:
  the same ids, 48 of 48 chain rows and 93 of 93 block dumps equal in bytes, block match 31/31 per
  prompt. XNNPACK takes 3880 of its 4220 nodes in 454 partitions. The
  `odml.scaled_dot_product_attention` form (`--attention=sdpa`, GQA pre-folded, mask as the fourth
  operand) fails on the CPU when the runner allocates its tensors (`XNNPack delegate failed to reshape
  runtime`, `Node number 1886 (TfLiteXNNPackDelegate) failed to prepare`). On Metal fp32 the rms, sdpa
  and rms + sdpa forms are fully delegated (4220, 4274 and 4500 nodes, 2 partitions) and pass the text
  match 3/3; none is byte-identical to Metal raw (at most 5.1e-5 apart). Chosen forms: CPU rms (equal
  bytes, 18.10 against 18.12 ms/token), Metal fp32 rms (7.86 against 8.06 ms/token for raw; sdpa 8.60,
  rms + sdpa 8.30) (`results/attention_forms.md`). The `odml.runtime_bmm` form was not built or run.
- Row 12: int8 per-channel and int4 blockwise-32 weights, with the fp32 tied lm_head and with an int8
  lm_head copy (`results/quantization.md`). Weights stored: fp32 1,761,607,680 bytes, int8 441,778,176,
  int4 247,726,080 (+156,190,208 for the int8 lm_head copy). Ids equal to the reference out of 48: CPU
  int8 41 (fails), CPU int4 6 (fails), CPU int8 + lm_head int8 48 (passes), CPU int4 + lm_head int8 20
  (fails), Metal fp32 with int8 weights 48 (passes, 3994 of 3994 nodes delegated), Metal int8 + lm_head
  int8 48 (passes). Metal with int4 weights ends with a segmentation fault (exit 139) about one second
  after `Initializing Metal-based API from graph.` in the first compile, with and without the int8
  lm_head copy; not retried.
- Row 13: 13 forms timed in one quiet window of 100 s (`results/speed.md`, `results/timing.json`). The
  floor, CPU fp32 raw, is 18.12 ms per decode token. The fastest form that passes the text match is
  Metal fp32 with int8 weights and the int8 lm_head, 4.72 ms/token; `checks/speed_floor.py` exits 0
  (`results/speed_floor.txt`). Metal fp16 is faster than Metal fp32 (5.76) but fails the text match
  (0 of 48).
- Row 10 (extra): with `--dump_final_norm_form=identity` the exported `final_norm` is a separate
  multiply by 1.0, so it is no longer the lm_head's direct input. The Metal fp32 per-block match is
  still 30 of 31 for each prompt: `final_norm` fails (max abs `nan` / 4.94e1 / `nan`) while every
  block and the last-position logits pass. The logits are byte-identical to the direct form's
  (`results/block_match_metal_fp32_identity_p*.txt`). Being the `FULLY_CONNECTED` input is therefore
  not the cause; the cause is still not established.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 1):

```
models/qwen3-0.6b/skill_table.md: tried 14, not applicable 0, blocked 0, not yet 1
NOT DONE models/qwen3-0.6b/skill_table.md: 1 rows not yet
  #14 Phone run
```

Against the acceptance list: row 8 met (the CPU sdpa form has no output; the failure is in the
table); row 12 met (Metal int4 has no count because its compile does not finish); row 13 met; row 10
extra recorded (fails, as in iteration 2).

## Corrections
- `EnableAllowSrcQuantizedFcConvOps` / `SetAllowSrcQuantizedFcConvOps` (`litert/c/options/litert_gpu_options.h`)
  does not decide whether the GPU accepts quantized FC weights: it lets the GPU delegate quantize the
  *input* of fully connected and convolution ops to 8 bit, and needs constant tensor sharing. It was
  not used; Metal took every node of the int8 graph without it.
- The sdpa composite on the CPU does not run with wrong output here; it fails before the first
  inference, when the XNNPACK delegate prepares node 1886.
- The Metal fp32 `final_norm` export failure is not explained by the tensor being the lm_head's direct
  input (row 10 above).
- The weight code takes the integer codes with `std::lround`. The build and the runs of this iteration
  used the float-returning function with the same rounding (halves away from zero), which gives the
  same value for every code (|w / scale| is at most about 127). The change was syntax-checked, not
  rebuilt or rerun.

## Dead ends
- The CPU sdpa form: the runner's compile fails as above. The un-folded presentation was not tried.
- Metal with int4 weights: segmentation fault in the delegate's first compile, once per form.
- A second run script for the `odml.runtime_bmm` attention form stopped at its fifth line: it read its
  shared settings with `source <(...)`, which the system bash (3.2) runs as an empty file, so a
  variable was unset under `set -u`. Nothing was built or run for that form, and the source changes
  for it were taken out again before the commit. With bash 3.2, source a written file, not a process
  substitution.

## Op and delegate observations
- CPU (XNNPACK), fp32: raw 3880 / 3994 nodes, 58 partitions; rms 3880 / 4220, 454 partitions; sdpa
  4104 / 4274, 114 partitions in the delegation report's compile, then the failure above in the
  runner's compile. int8 and int4 weight graphs: 3880 / 3994, 58 partitions.
- Metal (prebuilt `libLiteRtMetalAccelerator.dylib`), fp32: every node delegated in 2 partitions for
  raw, rms, sdpa, rms + sdpa and int8 weights. int4 blockwise-32 weights: segmentation fault.
- First failing tensor of the per-block match with quantized weights (fp32 bars): CPU int8 and int4 at
  `block_00`; Metal int8 at `block_01`.
- Decode ms/token in the window: CPU fp32 18.12, CPU int8 9.23, CPU int4 11.39, CPU int8 + lm_head int8
  6.06 (6.055), Metal fp32 8.06, Metal fp16 5.76, Metal int8 5.88, Metal int8 + lm_head int8 4.72.
- The XNNPACK thread count is not printed by the runtime: unverified.

## Environment
- `ENVIRONMENT.md`, section "Iteration 3": the same LiteRT commit, toolchain, machine, weights and Metal
  dylib as iteration 2; the new flags, the quantization formulas, how the time was taken and the
  window. One timing window (18:32:53–18:34:33 JST); there was no second one. No phone.

## Files left
- `graph/`: `qwen3_config.h`, `qwen3_graph.{h,cc}` (rms and sdpa composites, the identity export,
  the lm_head choice), `qwen3_weights.{h,cc}` (int8 and int4 quantization), `qwen3_main.cc` (the new
  flags and the timing), `BUILD` (`@flatbuffers`). Keep.
- `results/`: `attention_forms.md`, `quantization.md`, `speed.md`, `timing.json`, `speed_floor.txt`,
  `text_match_*` and `block_match_*` files of the new forms. Keep.
- Not committed and deletable: the dumps, outside the repository. The serialized `.tflite` files were
  deleted after each run; the per-form timing files are kept outside the repository as the source of
  `timing.json`.

## Time
Start 17:49, end 20:54 (JST, 2026-10-15).

| part | count | minutes |
|---|---|---|
| coding and reading | — | 33 (17:49–17:57 reading and code; 18:19–18:38 results, docs and the unused attention form; 20:48–20:54 closing) |
| builds | 1 | under 1 (4.3 s) |
| waiting for the machine: idle wait before the match runs | 2 | 18 (1081 s, then 0 s for the second script) |
| waiting for the machine: lock wait before the timing window | 1 | 14 (822 s) |
| runs and checks | 1 match script, 1 timing window | 5 (18:15:01–18:18:35 matches, 18:32:53–18:34:33 timing) |
| rework after a failure | 1 | 130 (18:38–20:48: the second script had stopped at its start and nothing ran until it was noticed) |
| phone in use | 0 | 0 |
