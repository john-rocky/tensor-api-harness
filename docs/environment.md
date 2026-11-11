# Environment

- LiteRT: `google-ai-edge/LiteRT` main at `bea3bc69df855f80e062ea5a7fb72b5445d2a64c` (2026-10-14). The
  Metal accelerator is the `latest` prebuilt the build fetches (its sha256 is in
  `models/qwen3-0.6b/ENVIRONMENT.md`); the Android GPU accelerator is the prebuilt
  `libLiteRtClGlAccelerator.so` from the same fetch.
- LiteRT-LM (iterations 7 to 10): `google-ai-edge/LiteRT-LM` main at
  `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33` (2026-10-18), with the LiteRT it pins
  (`9b8c7f77e7a29e6073e614a05e1e38be7b761229`); its macOS GPU path loads the prebuilt WebGPU
  accelerator (`libLiteRtWebGpuAccelerator.dylib`). The published model is
  `litert-community/Qwen3-0.6B` `Qwen3-0.6B.litertlm`, sha256
  `c5f39b4191926786c3e6668b62af4d6eb15248dfc7a5348252a9b406f33f2108`.
- Bazel 7.7.0 through bazelisk 1.29.0. Xcode 27.0 (27A266a) on macOS 27.0. Android NDK 28.2.13676358
  for the arm64 cross-compile (`bazel build -c opt --config=android_arm64`).
- Python 3.12.11 with torch 2.14.1 and transformers 5.19.0 for the reference dumps; Python 3.14.6
  with numpy 2.4.4 for the checks.
- Mac: Apple M4 Max, 128 GB. Phones: a Galaxy S26 (SM-S942U1), Android 16, a lab unit reached
  through Google's Developer Device Platform (the job pushes the binary and the saved graphs, runs a
  shell script, and the outputs are pulled back; no adb), and a Galaxy S26 (SM-S942Q), Android 16,
  attached over USB and driven with adb for the OpenCL rows.
