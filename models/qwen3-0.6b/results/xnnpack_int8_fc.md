# Does XNNPACK quantize the inputs of an int8-weight FULLY_CONNECTED?

In iteration 3, the CPU (XNNPACK) form with int8 per-channel decoder weights matched the reference
for 41 of 48 greedy ids. Metal with the same weights matched 48 of 48. The first decoder block
already differed more on the CPU: block_00 cosine 0.99937 on the CPU against 0.99993 on Metal
(`quantization.md`). How the CPU kernel treats the inputs of those FULLY_CONNECTED ops was not
checked then. Iteration 8 answers that from the source and with one A/B.

**Yes.** With LiteRT's default CPU options, the XNNPACK delegate runs every FULLY_CONNECTED that
has fp32 input and int8 weights as a dynamically quantized op. Before each matmul, the input is
quantized to int8 per row (one scale and one zero point per token). On this Mac (Apple M4 Max) the
matmul then runs in KleidiAI's SME2 kernels.

The A/B keeps the rounded weights and takes only that quantization away. The same weight values
are stored as fp32, so every FULLY_CONNECTED takes fp32 inputs. That brings block_00 to Metal's
numbers (cosine 0.99993 / 0.99993 / 0.99993) and the ids to 48 of 48. By the line written before
the runs, this explains iteration 3's CPU int8 result.

Versions: LiteRT `bea3bc69df855f80e062ea5a7fb72b5445d2a64c` (the same commit as iterations 1 to 6),
which pins XNNPACK `35997e71119d191c807068a4d2aa28bf6048b59b`, KleidiAI
`4bc7bd457930de119f8b826ecfdebfbd9c08cf8c` and cpuinfo `8ce83db858065145192c97af90cb668ad72a12e9`
(`litert_workspace.bzl` lines 84 to 129, no patches). Line numbers below are at those commits.

## Where the source decides it

LiteRT (`https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/`):

| what | file | lines |
|---|---|---|
| The CPU accelerator creates the XNNPACK delegate from the CPU options, or from the defaults when the model was compiled without any (`qwen3_main` sets none for the CPU) | [`litert/runtime/accelerators/xnnpack/xnnpack_accelerator.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/litert/runtime/accelerators/xnnpack/xnnpack_accelerator.cc#L72-L102), [`litert/runtime/litert_cpu_options.h`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/litert/runtime/litert_cpu_options.h#L42-L45) | 72–102; 42–45 |
| The default flags: signed and unsigned 8-bit on (the build's default defines), `DISABLE_DYNAMICALLY_QUANTIZED_OPS` off, `num_threads` 0 | [`tflite/delegates/xnnpack/xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L7891-L7918), [`tflite/delegates/xnnpack/BUILD`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/BUILD#L62-L82) | 7891–7918; 62–82 |
| How to change them: `CpuOptions::SetXNNPackFlags`; the value replaces the default flags, it is not added to them | [`litert/cc/options/litert_cpu_options.h`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/litert/cc/options/litert_cpu_options.h#L97-L101), [`litert/runtime/litert_cpu_options.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/litert/runtime/litert_cpu_options.cc#L93-L96) | 97–101; 93–96 |
| FULLY_CONNECTED: `dynamically_quantized` is true for fp32 (or fp16) input with int2, int4 or int8 weights unless the flag is set | [`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L5039-L5045) | 5039–5045 |
| The input becomes a `qdint8` value with one set of quantization parameters per row (`num_non_batch_dims = 1`), made by a convert node | [`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L5145-L5172) | 5145–5172 |
| The weights become a channelwise `qcint8` value (the per-channel fp32 scales of the tensor); the op is defined with an fp32 output | [`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L5188-L5230) | 5188–5197, 5224–5230 |
| With the flag, the delegate does not take the op ("unsupported mixed types") | [`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L5050-L5058), [`xnnpack_delegate.h`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.h#L59-L61) | 5050–5058; 59–61 |
| TFLite's builtin FULLY_CONNECTED, which then runs it, also quantizes the input per row, symmetrically (`asymmetric_quantize_inputs` is false in the graph) | [`tflite/kernels/fully_connected.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/kernels/fully_connected.cc#L1068-L1070), [`portable_tensor_utils.h`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/kernels/internal/portable_tensor_utils.h#L158-L176), [`arithmetic_tflite.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tensor/backends/tflite/arithmetic_tflite.cc#L557) | 1068–1070; 158–176; 557 |
| The tied lm_head has fp32 weights, so it is not quantized dynamically and runs as an fp32 FULLY_CONNECTED | [`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L5238-L5247) | 5238–5247 |

XNNPACK (`https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/`) and
KleidiAI (`https://gitlab.arm.com/kleidi/kleidiai/-/blob/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/`):

| what | file | lines |
|---|---|---|
| The packed-LHS pass: an op with `qdint8` input, `qcint8` weights and fp32 output takes `qpint8` (KleidiAI's packed format) when a `qp8_f32_qc8w` GEMM config exists; the convert node goes away and the FULLY_CONNECTED operator quantizes its fp32 input itself | [`src/subgraph.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/subgraph.c#L4121-L4284) | 4121–4124, 4189–4192, 4242–4284 |
| The unsigned `qduint8` variant applies only when its config is better; on arm64 that config is empty (only x86 sets one) | [`src/subgraph.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/subgraph.c#L2371-L2444), [`src/configs/gemm-config.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/configs/gemm-config.c#L4134-L4209) | 2371–2444; 4134–4209, 6810–6816 |
| The `qp8_f32_qc8w` config on arm64 with KleidiAI and SME2: `xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2` and `..._16x64c4__neonsme2` | [`src/configs/gemm-config.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/configs/gemm-config.c#L3095-L3120) | 3095–3120 |
| One row (a decode step) takes the 1-row kernel, more rows (prefill) the 16-row kernel | [`src/operators/fully-connected-nc.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/operators/fully-connected-nc.c#L3346-L3354) | 3346–3354 |
| The two kernels are KleidiAI's `kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot` and `kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa` | [`1x64c4-neonsme2.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-minmax-1x64c4-neonsme2.c#L21-L32), [`16x64c4-neonsme2.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-minmax-16x64c4-neonsme2.c#L21-L34) | 21–32; 21–34 |
| SME2 comes from cpuinfo, which reads `hw.optional.arm.FEAT_SME2` (1 on this Mac); KleidiAI and SME2 kernels are built by default for aarch64, macOS arm64 included | [`src/configs/hardware-config.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/configs/hardware-config.c#L253-L260), [cpuinfo `src/arm/mach/init.c`](https://github.com/pytorch/cpuinfo/blob/8ce83db858065145192c97af90cb668ad72a12e9/src/arm/mach/init.c#L603), [`BUILD.bazel`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/BUILD.bazel#L2071-L2088), [`build_config/BUILD.bazel`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/build_config/BUILD.bazel#L333-L345) | 253–260; 603; 2071–2088, 1649–1663; 167–173, 333–345 |
| The input quantization is KleidiAI's `kai_run_lhs_quant_pack_qai8dxp_f32` | [`src/configs/pack-lh-config.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/configs/pack-lh-config.c#L37-L44), [`src/x8-packq/x8-packq-aarch64-neon-f32qp8-u2.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/x8-packq/x8-packq-aarch64-neon-f32qp8-u2.c#L21-L35) | 37–44; 21–35 |
| Per row: the minimum and maximum, the range widened to include 0, scale 255 / range, a zero point clamped to [−128, 127] and set with `rintf`, each value multiplied by the scale and converted to the nearest integer with halves away from zero (`vcvtaq_s32_f32`), plus the zero point, clamped to [−128, 127]; the row's scale and zero point are stored with it | [`kai_lhs_quant_pack_qai8dxp_f32.c`](https://gitlab.arm.com/kleidi/kleidiai/-/blob/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c#L77-259) | 77–259 (78–134 the parameters, 155 and 183 the conversion, 251–259 the stored values) |

No flag of the XNNPACK delegate runs an int8-weight FULLY_CONNECTED with fp32 inputs. XNNPACK has
the type for it (fp32 input with `qcint8` weights,
[`src/subgraph/fully-connected.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/subgraph/fully-connected.c#L248-L251)
lines 248–251), but the delegate never builds that combination. The flag
`DISABLE_DYNAMICALLY_QUANTIZED_OPS` moves the op to TFLite's own FULLY_CONNECTED, which quantizes a
float input for int8 weights in every kernel variant
([`fully_connected.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/kernels/fully_connected.cc#L1807-L1839)
lines 1807–1839). XNNPACK's `XNN_FLAG_NO_INLINED_LHS_PACKING` (a `qd8` kernel in place of the `qp8`
one) is a runtime flag that LiteRT's CPU options do not pass. YNNPACK (LiteRT's other CPU path, a
build option) was not checked.

## The forms

All forms run on the CPU through LiteRT's CPU accelerator, the XNNPACK delegate (in the A-off forms
the int8 FULLY_CONNECTED ops go to TFLite's builtin kernel), with the binary of this iteration. The
decoder weights are the 196 matmul weights of iteration 3: int8 per output channel, nearest integer
codes, scale max|w| / 127.

| form | what the FULLY_CONNECTED ops get | flags |
|---|---|---|
| A1 `cpu_int8` | int8 weights, inputs quantized by XNNPACK per row; the tied lm_head fp32 | `--weights_quant=int8` |
| A2 `cpu_int8_lm_head_int8` | A1 with the int8 lm_head copy (its input quantized per row too) | `--weights_quant=int8 --weights_quant_lm_head` |
| B1 `cpu_int8_dequant_fp32` | A1's weight values, code × scale, in fp32 tensors; fp32 inputs | `--weights_quant=int8 --weights_dequant_fp32` |
| B2 `cpu_int8_lm_head_int8_dequant_fp32` | A2's weight values in fp32 tensors, the lm_head copy included; fp32 inputs | `--weights_quant=int8 --weights_quant_lm_head --weights_dequant_fp32` |
| A1-off `cpu_int8_builtin_fc` | A1 with the delegate flags default \| `DISABLE_DYNAMICALLY_QUANTIZED_OPS` (3 \| 0x800 = 2051): the int8 FULLY_CONNECTED ops run on TFLite's builtin kernel, inputs quantized per row symmetrically | `--weights_quant=int8 --cpu_disable_dynamic_quantization` |
| A2-off `cpu_int8_lm_head_int8_builtin_fc` | A2 the same way | `--weights_quant=int8 --weights_quant_lm_head --cpu_disable_dynamic_quantization` |

## Results

block_00 and the logits are compared with the PyTorch fp32 reference by `checks/block_match.py`
(the manifests' fp32 bars; for quantized weights a FAIL line records where the difference starts),
the ids by `checks/text_match.py`. "ids /48" counts the generated ids equal to the reference before
each prompt's first mismatch. Decode ms/token: one quiet window on 2026-10-19 (241 s, no other
measurement on the machine), the seven forms alternating, one discarded run and 8 measured runs per
form; one run is the three prompts with prefill and 15 decode steps, each call timed until the
logits buffer is read (the definition of iteration 6), and its value is the median of the three
per-prompt medians (`xnnpack_int8_fc_timing.txt`).

| form | block_00 max abs (p0 / p1 / p2) | block_00 cosine (p0 / p1 / p2) | first failing tensor | ids /48 (p0, p1, p2) | XNNPACK nodes, partitions | decode ms/token: median [min, max], runs |
|---|---|---|---|---|---|---|
| A1 | 4.6737e-02 / 4.2312e-02 / 4.4537e-02 | 0.99937331 / 0.99935327 / 0.99931778 | block_00 | 41 (16, 16, 9) | 3880 / 3994, 58 | 9.208 [9.053, 9.357], 8 |
| A2 | 4.6737e-02 / 4.2312e-02 / 4.4537e-02 | 0.99937331 / 0.99935327 / 0.99931778 | block_00 | 48 (16, 16, 16) | 3880 / 3994, 58 | 6.130 [6.008, 6.324], 8 |
| B1 | 1.7496e-02 / 1.7676e-02 / 1.9908e-02 | 0.99993395 / 0.99993384 / 0.99992909 | block_01 | 48 (16, 16, 16) | 3880 / 3994, 58 | 18.045 [17.932, 18.908], 8 |
| B2 | 1.7496e-02 / 1.7676e-02 / 1.9908e-02 | 0.99993395 / 0.99993384 / 0.99992909 | block_01 | 48 (16, 16, 16) | 3880 / 3994, 58 | 18.007 [17.921, 18.752], 8 |
| A1-off | 7.5921e-02 / 6.3262e-02 / 6.7404e-02 | 0.99886457 / 0.99902245 / 0.99884960 | block_00 | 33 (16, 16, 1) | 3488 / 3994, 282 | 16.599 [16.372, 17.800], 8 |
| A2-off | 7.5921e-02 / 6.3262e-02 / 6.7404e-02 | 0.99886457 / 0.99902245 / 0.99884960 | block_00 | 22 (5, 16, 1) | 3486 / 3994, 282 | 15.661 [15.578, 17.137], 8 |
| Metal fp32, int8 weights (for comparison) | 1.7496e-02 / 1.7676e-02 / 1.9908e-02 | 0.99993394 / 0.99993384 / 0.99992909 | block_01 | 48 (16, 16, 16) | Metal 3994 / 3994, 2 | — |
| CPU fp32 weights (for comparison) | 1.4305e-06 / 1.3411e-06 / 1.4305e-06 | 1.00000000 / 1.00000000 / 1.00000000 | none (31/31) | 48 (16, 16, 16) | 3880 / 3994, 58 | 18.107 [17.991, 18.959], 8 |

Files: `block_match_<form>_p0.txt` to `_p2.txt` and `text_match_<form>.txt` for the forms by name
(A1 and A2's text match, Metal and CPU fp32 are the existing files: `block_match_cpu_int8_p*.txt`,
`text_match_cpu_int8.txt`, `text_match_cpu_int8_lm_head_int8.txt`, `block_match_metal_fp32_int8_p*.txt`,
`text_match_metal_fp32_int8.txt`, `block_match_p*.txt`, `text_match.txt`). Today's reruns of those
gave byte-identical files, except Metal's `final_norm` line, which is the broken export of iteration 2.

- In XNNPACK nodes, A1-off has 392 fewer than A1: the 196 int8 FULLY_CONNECTED ops in each of the two
  signatures. A2-off has 2 more out: the int8 lm_head copy in each signature.
- B1's cosines are within 2e-8 of Metal int8's for block_00, block_01 and the last position's
  logits, on all three prompts.
- The delegate flags of the A-off forms, as the binary printed them: 2051 (the default 3 plus 0x800).

Speed: A1 decodes in about half of B1's time (9.208 against 18.045 ms/token), and B1 is as fast as
the fp32 weights (18.107; the two differ by 0.3%). On this CPU the int8 speed comes with the input
quantization; keeping the inputs in fp32 costs the fp32 time. A2, at 6.130 ms/token and 48 of 48,
is the fastest form on the CPU that meets the bar. Iteration 3 timed it at 6.06 with the run-only
definition. TFLite's builtin kernel (the A-off forms) decodes slower than XNNPACK's int8 kernels,
and its prefill is 3.5 to 9.8 times longer than the matching XNNPACK form's. Prefill ms, the median
of the 8 runs, p0 (S 5) / p1 (S 4) / p2 (S 18):

| form | prefill ms |
|---|---|
| A1 | 14.7 / 14.3 / 28.0 |
| A2 | 11.1 / 10.1 / 21.6 |
| B1 | 25.4 / 25.4 / 46.0 |
| B2 | 25.4 / 24.6 / 45.4 |
| A1-off | 89.2 / 77.0 / 99.1 |
| A2-off | 108.8 / 97.4 / 123.7 |
| CPU fp32 weights | 25.8 / 24.9 / 45.8 |

## The line written before the runs, and the decision

The line, from the instruction of this iteration (block_00 cosine = the minimum over the three
prompts):
- explained: B1 block_00 cosine at least 0.99991 and B1 ids at least 44 of 48;
- refuted: B1 block_00 cosine at most 0.9995, or B1 ids at most 43 of 48;
- otherwise: partly.

**Explained.** B1's minimum block_00 cosine is 0.99992909 and its ids are 48 of 48. With the same
rounded weights, the CPU's block_00 goes from cosine 0.99932 (A1) to 0.99993 (B1) when the
FULLY_CONNECTED inputs stay fp32. That is Metal's value with the same weights.

## Block 0 computed again

`xnnpack_int8_fc_block0.py` computes decoder block 0 in numpy from the reference embedding rows,
four ways, and compares each with the reference and with the runner's block_00 dumps
(`xnnpack_int8_fc_block0.txt`). max abs, p0 / p1 / p2; the matching pairs have cosine 1.00000000:

| computed with | against A1's dump | against B1's dump | against A1-off's dump | against the reference |
|---|---|---|---|---|
| fp32 weights, fp32 inputs | 4.7e-02 / 4.2e-02 / 4.5e-02 | 1.7e-02 / 1.8e-02 / 2.0e-02 | 7.6e-02 / 6.3e-02 / 6.7e-02 | **3.1e-06 / 2.1e-06 / 2.9e-06** |
| int8 weights, fp32 inputs | 3.8e-02 / 4.1e-02 / 4.2e-02 | **3.8e-06 / 1.9e-06 / 6.7e-06** | 7.2e-02 / 6.0e-02 / 6.5e-02 | 1.7e-02 / 1.8e-02 / 2.0e-02 |
| int8 weights, inputs quantized per row as KleidiAI does | **2.4e-06 / 4.8e-07 / 1.9e-06** | 3.8e-02 / 4.1e-02 / 4.2e-02 | 8.4e-02 / 7.8e-02 / 8.0e-02 | 4.7e-02 / 4.2e-02 / 4.5e-02 |
| int8 weights, inputs quantized per row symmetrically | 8.4e-02 / 7.8e-02 / 8.0e-02 | 7.2e-02 / 6.0e-02 / 6.5e-02 | **9.5e-07 / 4.8e-07 / 9.5e-07** | 7.6e-02 / 6.3e-02 / 6.7e-02 |

Each runner form matches one computation to about 1e-6, the fp32 rounding level, and differs from
the others by 1e-2 to 1e-1. A1 is the int8 weights with KleidiAI's per-row asymmetric quantization
of every FULLY_CONNECTED input. B1 is the int8 weights with fp32 inputs. A1-off is the symmetric
per-row quantization of TFLite's builtin kernel.

## What ran

`/usr/bin/sample` of one greedy run per form (every 1 ms, the whole process;
`xnnpack_int8_fc_kernels.txt`, function names and call-graph sample counts):
- A1: `kai_run_lhs_quant_pack_qai8dxp_f32` (the input quantization), `kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot`
  (decode) and `kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa` (prefill), `kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon`
  (the weights packed at compile), and `xnn_pf32_gemm_minmax_ukernel_{1x32,32x32}__neonsme2` (the fp32 ops, the lm_head among them).
- B1: no `kai_` frame; `xnn_pf32_gemm_minmax_ukernel_{1x32,32x32}__neonsme2` only. The driver
  `xnn_compute_inline_packed_qp8gemm` appears in every form: XNNPACK uses it for every GEMM with a
  packed left-hand side, the fp32 one included ([`fully-connected-nc.c`](https://github.com/google/XNNPACK/blob/35997e71119d191c807068a4d2aa28bf6048b59b/src/operators/fully-connected-nc.c#L3684-L3695) lines 3684–3695).
- A1-off: `tflite::ops::builtin::fully_connected::EvalHybridDense`, `tflite::tensor_utils::NeonSymmetricQuantizeFloats`
  and `tflite::tensor_utils::NeonMatrixBatchVectorMultiplyAccumulateImpl`.
- One thread was sampled in each form: the delegate gets `num_threads` 0, so it has no thread pool
  ([`xnnpack_delegate.cc`](https://github.com/google-ai-edge/LiteRT/blob/bea3bc69df855f80e062ea5a7fb72b5445d2a64c/tflite/delegates/xnnpack/xnnpack_delegate.cc#L671-L678) lines 671–678).

## The same outputs as iteration 3

- A1's block and text match files are byte-identical to iteration 3's.
- The binary of iteration 6 (the one before this iteration's flags) gives byte-identical dumps (31
  files per prompt) and ids for A1.
- With the defaults, the serialized graph (prefill length 18) of this binary and of the one before are
  byte-identical, with fp32 weights and with int8 weights.

## Not checked

- The Metal side of the mirror: `--gpu_allow_src_quantized_fc` (it needs constant tensor sharing) stops
  at the first compile with `model_builder.cc:3146] EMBEDDING_LOOKUP: Unsupported external weights type: 1`
  (a check failure, in both runs), so Metal with 8-bit FULLY_CONNECTED inputs was not measured.
- Why A2 (the int8 lm_head copy) reaches 48 of 48 while A1 reaches 41: their blocks are identical, and
  the logits differ (`block_match_cpu_int8_lm_head_int8_p*.txt`). One run per form. This is not a
  claim that an int8 lm_head is more accurate.
- The `qd8` kernels (`XNN_FLAG_NO_INLINED_LHS_PACKING`) and other prompt lengths, models or machines.
