# Iteration 8 — Qwen3-0.6B: does XNNPACK quantize the activations of an int8-weight FULLY_CONNECTED

## Read first
- `iterations/ITERATION_3.md` and `results/quantization.md`: CPU int8 matched 41 of 48 ids, Metal with
  the same int8 weights 48 of 48; the first decoder block differed more on the CPU (block_00 cosine
  0.99937 against 0.99993). How the CPU kernel treats the inputs of an int8-weight FULLY_CONNECTED
  was not inspected.
- LiteRT's XNNPACK delegate (`tflite/delegates/xnnpack/xnnpack_delegate.{h,cc}`), the CPU
  accelerator (`litert/runtime/accelerators/xnnpack/xnnpack_accelerator.cc`), the CPU options
  (`litert/cc/options/litert_cpu_options.h`), and XNNPACK at the commit LiteRT pins.

## Goal (one)
Answer the question with the source lines that decide it and with one A/B that takes only the
activation quantization away, and say whether that explains iteration 3's CPU int8 numbers.

The A/B keeps the rounded int8 weights and changes the FC inputs:
- A1: int8 per-channel decoder weights, lm_head fp32 (iteration 3's 41 of 48 form).
- A2: A1 plus the int8 lm_head copy (iteration 3's 48 of 48 form).
- B1: the same rounded weight values as A1 (code * scale) stored in fp32 tensors, so every FC takes
  fp32 inputs.
- B2: the same for A2, the lm_head copy included.
- A source line that turns the activation quantization off adds A1 and A2 with that setting.

## Acceptance
- The source lines (file and line, with links at the pinned commits) for: the flags LiteRT passes to
  XNNPACK and how to change them; the delegate's branch for fp32 input with int8 per-channel weights;
  the XNNPACK node, operator and micro-kernel and the quantization grain, zero point and rounding;
  the lm_head; whether a flag turns the quantization off.
- A1 and A2 reproduce iteration 3's ids and block_00 cosines, or today's numbers replace them.
- For every form: block_00 cosine for the three prompts, ids out of 48, decode ms/token (median,
  range and run count, all forms timed inside one quiet window, alternating).
- The decision, by the line written before the runs (block_00 cosine is the minimum of the three
  prompts, from `checks/block_match.py`):
  - explained: B1 block_00 cosine at least 0.99991 (Metal int8's level, 0.99993) and B1 ids at least
    44 of 48;
  - refuted: B1 block_00 cosine at most 0.9995 (A1's level) or B1 ids at most 43 of 48;
  - otherwise: partly, with the numbers.
- With the defaults, the serialized graph is byte-identical to the previous binary's.
- `tools/iteration_close_check.py` passes; a `## Time` section.

## Stop conditions
- A1 does not reproduce iteration 3 and the cause is not found within one hour (binary version,
  options and weight bytes compared).
- The source shows that this graph's FULLY_CONNECTED is not delegated to XNNPACK.
- Waiting for quiet windows adds up to more than four hours before the timing is done.
- Less than 30 GB free on the disk.

## Scope
`graph/` (additions only; with the defaults the graph and the outputs are unchanged), new files in
`results/`, and this file. The bars and `checks/` are unchanged.

## Do not
- Do not change the thresholds in `checks/`.
- Do not change `README.md`, the skill tables, `checks/` or `tools/`.
- Do not commit weights, generated graphs, binaries or dumps.
- Do not time a form outside the quiet window, and do not publish a number that was not measured.

## Progress
- Source read at the pinned commits (`results/xnnpack_int8_fc.md`, "Where the source decides it").
  With the default CPU options the delegate quantizes the input of every int8-weight
  FULLY_CONNECTED to int8 per row, with a zero point; on the M4 Max the matmul is KleidiAI's SME2
  `qai8dxp` × `qsi8cxp` kernel. The delegate flag `DISABLE_DYNAMICALLY_QUANTIZED_OPS` hands the op
  to TFLite's builtin kernel, which quantizes the input per row too (symmetrically). No flag keeps
  the inputs in fp32, so the B forms keep the weight rounding and store the values as fp32.
- `graph/`: `--weights_dequant_fp32` (the loader stores code × scale as fp32 tensors, the lm_head
  copy too) and `--cpu_disable_dynamic_quantization` (CPU options with the delegate's default flags
  plus 0x800; the default path still sets no CPU options). With the defaults, the serialized graph
  is byte-identical to the previous binary's, with fp32 and with int8 weights.
- A1 reproduces iteration 3 byte for byte: 41 of 48, block_00 cosine 0.99937331 / 0.99935327 /
  0.99931778 (`block_match_cpu_int8_p*.txt` and `text_match_cpu_int8.txt` unchanged). The previous
  binary gives byte-identical dumps and ids.
- A/B (`results/xnnpack_int8_fc.md`, "Results"): A1 41/48, A2 48/48, B1 48/48, B2 48/48, A1-off
  33/48, A2-off 22/48. block_00 cosine, the minimum of the three prompts: A1 and A2 0.99931778, B1
  and B2 0.99992909 (Metal int8: 0.99992909), A1-off and A2-off 0.99884960.
- Decode ms/token in one quiet window, the forms alternating (8 measured runs each,
  `results/xnnpack_int8_fc_timing.txt`): A1 9.208, A2 6.130, B1 18.045, B2 18.007, A1-off 16.599,
  A2-off 15.661, CPU fp32 weights 18.107. B1 takes the fp32 time; A1's int8 speed comes with the
  input quantization. A2 stays the fastest CPU form that meets the bar.
- Block 0 computed again in numpy (`results/xnnpack_int8_fc_block0.py`): each form's block_00 dump
  matches one arithmetic to about 1e-6. A1 matches the per-row asymmetric quantization of every
  FULLY_CONNECTED input, B1 the fp32 inputs, A1-off the symmetric quantization.
- `/usr/bin/sample` shows the kernels that ran (`results/xnnpack_int8_fc_kernels.txt`): KleidiAI's
  input quantization and SME2 matmuls in A1, only fp32 GEMMs in B1, TFLite's hybrid FULLY_CONNECTED
  in A1-off.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0; the skill table is unchanged):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Source lines:** met (`results/xnnpack_int8_fc.md`).
- **A1 and A2 reproduce iteration 3:** met, byte-identical files.
- **Every form:** block_00 cosine for three prompts, ids out of 48 and decode ms/token, the timing in
  one window: met (241 s, 8 measured runs per form).
- **The decision:** explained. B1's minimum block_00 cosine is 0.99992909 (at least 0.99991) and its
  ids are 48 of 48 (at least 44).
- **Defaults unchanged:** met, the serialized graphs are byte-identical.
- **Close check:** `python3 tools/iteration_close_check.py models/qwen3-0.6b/iterations/ITERATION_8.md`:
  `PASS iteration close: models/qwen3-0.6b/iterations/ITERATION_8.md`.

## Corrections
- Iteration 3's open question ("how the CPU kernel treats the activations of an int8-weight
  FULLY_CONNECTED was not inspected") is answered: XNNPACK quantizes them per row
  (`results/xnnpack_int8_fc.md`).
- The thread count of the CPU runs, unknown since iteration 3: the delegate gets `num_threads` 0
  and creates no thread pool, and one thread was sampled in each form
  (`results/xnnpack_int8_fc_kernels.txt`).

## Dead ends
- **Metal with `--gpu_allow_src_quantized_fc`:** a check failure at the first compile,
  `model_builder.cc:3146] EMBEDDING_LOOKUP: Unsupported external weights type: 1`. The option needs
  constant tensor sharing; with it, the delegate does not accept the lookup's fp32 table as an
  external weight (the cause beyond that message is unverified). Two runs, the same failure; not
  retried.
- **Disabling the dynamic quantization by flag:** `DISABLE_DYNAMICALLY_QUANTIZED_OPS` does not keep
  fp32 inputs; it moves the op to TFLite's builtin kernel, which quantizes them symmetrically
  (A1-off: 33 of 48). Storing the rounded weights as fp32 is the way to keep the weights and drop the
  input quantization.
- **The flags option:** `CpuOptions::SetXNNPackFlags` replaces the default flags; passing 0x800 alone
  also drops the signed and unsigned 8-bit flags. The binary passes the default (3) plus 0x800.

## Op and delegate observations
- XNNPACK delegation: A1, A2, B1, B2 and fp32 3880 of 3994 nodes, 58 partitions; A1-off 3488 of
  3994 and A2-off 3486, 282 partitions each (the int8 FULLY_CONNECTED ops of both signatures run on
  TFLite's builtin kernel).
- Kernels seen in `/usr/bin/sample` (one greedy run each): A1 `kai_run_lhs_quant_pack_qai8dxp_f32`,
  `kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot` (decode),
  `kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa` (prefill),
  `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` (weight packing at compile, the largest count in that
  run); B1 `xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2` and `_32x32__neonsme2` only; A1-off
  `tflite::ops::builtin::fully_connected::EvalHybridDense` with
  `tflite::tensor_utils::NeonSymmetricQuantizeFloats`.
- The XNNPACK driver `xnn_compute_inline_packed_qp8gemm` runs the fp32 packed GEMMs too; the leaf
  kernel names say which arithmetic ran.
- Metal fp32 with int8 weights (prebuilt `libLiteRtMetalAccelerator.dylib`, sha256
  `5531feca4ff0e2c67628e09d635a6f79da74534e88b7a3d262efa5dcda59e95d`): 3994 of 3994 nodes,
  2 partitions, 48 of 48; block files as in iteration 3 apart from the broken `final_norm` export.
- LiteRT `bea3bc69df855f80e062ea5a7fb72b5445d2a64c`, XNNPACK `35997e71119d191c807068a4d2aa28bf6048b59b`,
  KleidiAI `4bc7bd457930de119f8b826ecfdebfbd9c08cf8c`.
- Unverified: YNNPACK; XNNPACK's `qd8` kernels (`XNN_FLAG_NO_INLINED_LHS_PACKING`, not passed by
  LiteRT's options); other prompt lengths, models and machines; why the int8 lm_head copy brings A2
  to 48 of 48 (one run per form).

## Environment
- Apple M4 Max (16 cores: 12 performance, 4 efficiency; `hw.optional.arm.FEAT_SME2` 1), macOS 27.0
  (26A428), Xcode 27.0 (27A266a), bazel 7.7.0 through bazelisk 1.29.0, `-c opt`.
- Checks and the block 0 computation: Python 3.14.6, numpy 2.4.4. Weights: `Qwen/Qwen3-0.6B`,
  `model.safetensors` 1,503,300,328 bytes (the snapshot of iterations 1 to 6).
- Quiet windows of the machine (each held the machine for this iteration alone): the build (151 s,
  after waiting 50 s for the lock), the match runs (127 s, after 725 s), the timing (241 s, after
  1,899 s) and the block 0 computation (under 1 s, after 1,876 s). The load average was 22.2 when
  the timing window opened (the previous minute) and 3.5 when it closed.
- No phone.

## Files left
- `graph/`: `BUILD`, `qwen3_main.cc`, `qwen3_weights.cc`, `qwen3_weights.h` (the two flags). Keep.
- `results/`, new, kept:
  - `xnnpack_int8_fc.md`: the question, the source lines, the A/B and the decision;
  - `block_match_cpu_int8_lm_head_int8_p*.txt`, `block_match_cpu_int8_dequant_fp32_p*.txt`,
    `block_match_cpu_int8_lm_head_int8_dequant_fp32_p*.txt`, `block_match_cpu_int8_builtin_fc_p*.txt`,
    `block_match_cpu_int8_lm_head_int8_builtin_fc_p*.txt`, and the four matching `text_match_*.txt`;
  - `xnnpack_int8_fc_block0.py` and `xnnpack_int8_fc_block0.txt`: the numpy computation of block 0
    and its output;
  - `xnnpack_int8_fc_kernels.txt`: the frames seen by `/usr/bin/sample`;
  - `xnnpack_int8_fc_timing.txt`: the decode and prefill times of the 8 measured runs per form.
- Not committed: the binaries, the saved `.tflite` files (deleted after each run), the dumps, the
  per-run timing JSON and the `sample` reports.

## Time
Start 19:26, end 22:15 (2026-10-19), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 91 (19:26–19:49 source, code and scripts; 21:03–21:15 scripts; 21:15–22:01 records during the lock waits; 22:05–22:15 closing) |
| builds | 1 host, 2 probes | 2.5 (140 s cold, in a copy of the LiteRT checkout; the probes 9.4 s and 3.2 s) |
| waiting for the machine: quiet windows | 4 | 50 (3,011 s of wall clock; the four lock waits add up to 4,550 s, two of them overlapping) |
| waiting for a decision | 1 | 74 (19:49–21:03, waiting for the next instruction) |
| runs and checks | 3 windows of runs, checks | 6 (127 s match runs, 241 s timing, under 1 s block 0, checks) |
| rework after a failure | 1 | 1 (the first build script could not enter a window and was stopped; its probe ran again in the build window) |
| phone in use | 0 | 0 |
