# Qwen3-0.6B — environment record

Everything iteration 1 (rows 1–6, CPU fp32) ran on. Paths are written as `<LiteRT checkout>`,
`<HF cache>` and `<this repo>`.

## LiteRT and toolchain

- LiteRT: `google-ai-edge/LiteRT` main at `bea3bc69df855f80e062ea5a7fb72b5445d2a64c` (2026-10-14),
  a shallow clone. No local change to the checkout; `graph/` is linked in as
  `tensor/examples/qwen3-0.6b` by `tools/overlay.sh`.
- Bazel 7.7.0 (the checkout's `.bazelversion`) through bazelisk 1.29.0. No `--config`: the checkout's
  `.bazelrc` applies `build:macos` by itself (`build --enable_platform_specific_config`).
- Xcode 27.0 (27A266a), macOS 27.0 (26A428), Apple M4 Max, 128 GB.
- Backend: CPU (XNNPACK delegate), fp32.

## Python

- Reference dumps: Python 3.12.11, torch 2.14.1, transformers 5.19.0, numpy 2.5.4, safetensors 0.8.0.
  CPU, fp32, `attn_implementation="eager"`.
- Checks (`checks/`, `tools/`): Python 3.14.6, numpy 2.4.4.

## Weights

- `Qwen/Qwen3-0.6B`, snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, `model.safetensors`
  (bf16, 1,503,300,328 bytes), sha256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Loaded as fp32 on both sides. The graph reads `model.embed_tokens.weight` once and uses it for the
  lookup and for the tied lm_head; `lm_head.weight` (the same bytes) is not read.

## Commands

```
# reference (dumps are not committed; prompts.json, ref_ids.json and the manifests are)
python reference/dump_reference.py dump --model <HF cache>/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --dumps <dump dir>/ref --repo <this repo>/models/qwen3-0.6b/reference

# build (from <LiteRT checkout>, after tools/overlay.sh <LiteRT checkout> qwen3-0.6b)
bazel build //tensor/examples/qwen3-0.6b:qwen3_main

# row 3: synthetic weights (no --weights), both signatures
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --ids_file=<dump dir>/ref/p0/ids.txt \
  --dump_blocks --dump_dir=<dump dir>/got_synth --greedy=3 --out_ids=<dump dir>/got_synth/ids.json \
  --accelerator=cpu --tflite_path=<tmp>/qwen3_synth.tflite

# row 5: per-block dumps for the three prompts
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --weights=<HF cache>/.../model.safetensors \
  --ids_file=<dump dir>/ref/ids_all.txt --dump_blocks --dump_dir=<dump dir>/got \
  --accelerator=cpu --tflite_path=<tmp>/qwen3_cpu_dump.tflite
python3 checks/block_match.py --got <dump dir>/got/p0 --ref <dump dir>/ref/p0 \
  --manifest models/qwen3-0.6b/reference/manifest_p0.json        # and p1, p2

# row 6: greedy 16 tokens, production signatures (no --dump_blocks)
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --weights=<HF cache>/.../model.safetensors \
  --ids_file=<dump dir>/ref/ids_all.txt --greedy=16 --out_ids=<dump dir>/got/got_ids.json \
  --accelerator=cpu --tflite_path=<tmp>/qwen3_cpu.tflite
python3 checks/text_match.py --got <dump dir>/got/got_ids.json --ref models/qwen3-0.6b/reference/ref_ids.json
```

Flags left at their defaults: `--max_seq=64` (the KV cache length; prompt + 16 tokens fit),
`--gpu_precision` and `--gpu_buffer_storage` (unused on the CPU).

## Iteration 2 (rows 7, 9, 10, 11)

Same LiteRT commit, toolchain, machine, Python environments and weights as above. No phone.

- Metal accelerator: the prebuilt `libLiteRtMetalAccelerator.dylib` that the build fetches as
  `@litert_prebuilts` (the checkout's `third_party/litert_prebuilts/workspace.bzl` points at a
  `latest` archive, so the commit does not pin it). sha256 of the file used:
  `5531feca4ff0e2c67628e09d635a6f79da74534e88b7a3d262efa5dcda59e95d`. The log does not print the
  GPU's name; the machine is the Apple M4 Max above.
- Metal runs, from `<LiteRT checkout>` (the working directory is the checkout root):

```
DYLD_LIBRARY_PATH=<LiteRT checkout>/bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main.runfiles/litert_prebuilts/macos_arm64 \
  bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --weights=<HF cache>/.../model.safetensors \
  --ids_file=<dump dir>/ref/ids_all.txt --greedy=16 --dump_chain --dump_dir=<dump dir>/got_metal_fp32 \
  --out_ids=<dump dir>/got_metal_fp32/ids.json --accelerator=gpu --gpu_precision=fp32 \
  --gpu_buffer_storage=buffer --tflite_path=<tmp>/qwen3_gpu.tflite
```

- Flags used in this iteration: `--accelerator=cpu|gpu`, `--gpu_precision=fp32|fp16` (always set
  explicitly; the delegate's default is fp16), `--gpu_buffer_storage=buffer|default|texture2d`,
  `--dump_chain` (with `--greedy=16` and `--dump_dir`), `--dump_blocks`, `--log_level=verbose`
  (one Metal run), `--delegation_report` (default on), `--rebind=handle` (the default; `copy` was
  not needed). `--max_seq` stays 64.
- Row 7 reference: `python reference/dump_reference.py chain --model <HF cache>/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca --dumps <dump dir>/ref --repo <this repo>/models/qwen3-0.6b/reference`
  (writes `manifest_chain_p*.json`), then
  `python3 checks/block_match.py --got <dump dir>/got_chain/p0 --ref <dump dir>/ref/p0 --manifest models/qwen3-0.6b/reference/manifest_chain_p0.json`
  (and p1, p2; the Metal chain uses the same manifests).
- Rows 9 to 11: the Metal block dumps use the iteration 1 manifests unchanged
  (`reference/manifest_p*.json`).

## Iteration 3 (rows 8, 12, 13; row 10 final-norm export form)

Same LiteRT commit, toolchain, machine, Python environments, weights and Metal dylib (same sha256)
as above. No phone.

- New flags of `qwen3_main` (each default gives the iteration 2 graph; a CPU run with the defaults
  was checked byte-identical to the iteration 1 and 2 dumps on every build of this iteration):
  `--norms=raw|composite`, `--attention=raw|sdpa`, `--weights_quant=none|int8|int4`,
  `--weights_quant_lm_head`, `--gpu_allow_src_quantized_fc` (built, not used: Metal took every node of
  the int8 graph without it), `--dump_final_norm_form=direct|identity`, `--time`, `--timing_out`,
  `--form`.
- Composites: `odml.rms_norm` with attribute `epsilon` (1e-6);
  `odml.scaled_dot_product_attention` with attribute `scale` (1/√128) on the GQA pre-fold
  (q `[8, 2·S, 1, 128]`, k/v `[8, 64, 1, 128]`, mask tiled to `[1, 1, 2·S, 64]`). In both composites
  the decomposition is the raw math. The `odml.runtime_bmm` form was not built or run.
- Weight quantization (in the loader, nearest integer with halves away from zero, symmetric, zero point 0), on the 196 decoder
  matmul weights:
  - int8: one scale per output channel, `scale = max|w| / 127`, codes `w / scale` to the nearest integer,
    clamped to ±127 (`PerChannelAffineQuantization`, dimension 0).
  - int4: one scale per block of 32 input elements, `scale = max|w| / 7`, rounded to fp16 before the
    codes `w / scale` to the nearest integer, clamped to ±7, are taken (the runtime reads fp16 block scales); two codes
    per byte, even index in the low nibble (`Type::kI4`, `BlockwiseQuantization`, block 32).
  - The embedding table stays fp32 for the lookup. `--weights_quant_lm_head` adds an int8 per-channel
    copy of it (same int8 formula) that only the lm_head reads.
- Timing (`--time`): `std::chrono::steady_clock` around each `runner.Run` call only (prefill once per
  prompt, then each of the 15 decode steps of a greedy-16 run). Input setup, the cache rebind and the
  logits read are outside. Per form: the median over 15 steps per prompt, then the median of the three
  prompt medians. The model is compiled once per prompt (prefill length = prompt length); the
  delegation report is off in timed runs (`--delegation_report=false`), so there is one compile per
  prompt. No separate warm-up call: the first decode call of each prompt is in the data (it is the
  slowest; the median is not moved by it).
- Timing windows: the machine has a lock that every measurement job takes. Timing ran only while this
  iteration held it, with no build or match run inside: one window, 18:32:53–18:34:33 JST (100 s),
  for every timed form. There was no second window. The 1-minute load average at the start of the
  window was 10.43, with no other measurement running.
- CPU threads: the runtime's default; the log does not print the XNNPACK thread count. The machine has
  16 logical CPUs (12 performance, 4 efficiency; `sysctl hw.ncpu`, `hw.perflevel0.logicalcpu`,
  `hw.perflevel1.logicalcpu`).
- Commands (from `<LiteRT checkout>`; GPU runs add the `DYLD_LIBRARY_PATH` of iteration 2 and
  `--accelerator=gpu --gpu_precision=fp32 --gpu_buffer_storage=buffer`):

```
# a form's match run (greedy 16 + chain), e.g. Metal fp32 with int8 weights
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --weights=<HF cache>/.../model.safetensors \
  --ids_file=<dump dir>/ref/ids_all.txt --greedy=16 --dump_chain --dump_dir=<dump dir>/metal_int8 \
  --out_ids=<dump dir>/metal_int8/ids.json --weights_quant=int8 --accelerator=gpu \
  --gpu_precision=fp32 --gpu_buffer_storage=buffer --tflite_path=<tmp>/qwen3.tflite
python3 checks/text_match.py --got <dump dir>/metal_int8/ids.json --ref models/qwen3-0.6b/reference/ref_ids.json

# a form's timed run (inside the measurement window)
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main --weights=<HF cache>/.../model.safetensors \
  --ids_file=<dump dir>/ref/ids_all.txt --greedy=16 --out_ids=<dump dir>/time_ids.json \
  --time --timing_out=<dump dir>/time_metal_int8.json --delegation_report=false \
  --form="metal int8" --weights_quant=int8 --accelerator=gpu --gpu_precision=fp32 \
  --gpu_buffer_storage=buffer --tflite_path=<tmp>/qwen3.tflite
python3 checks/speed_floor.py --timing models/qwen3-0.6b/results/timing.json

# row 10: final_norm export with the identity form
bazel-bin/tensor/examples/qwen3-0.6b/qwen3_main ... --dump_blocks --dump_final_norm_form=identity \
  --dump_dir=<dump dir>/metal_identity --accelerator=gpu --gpu_precision=fp32 --gpu_buffer_storage=buffer
```

- `results/timing.json` is assembled from the per-form timing files (one per form, written by
  `--timing_out`): floor = the "cpu fp32 raw" form, `ms_per_token` = the median of the prompt medians,
  `match` = PASS when at least 44 of the 48 generated ids equal the reference before each prompt's
  first mismatch.
