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
