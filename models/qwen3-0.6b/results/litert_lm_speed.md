# Speed through LiteRT-LM: the Tensor API executor and the published .litertlm in one binary

The binary of `litert_lm_registry.md` runs LiteRT-LM's benchmark entry with a forced prefill of 1024
tokens, 256 decode steps and a KV cache of 2048 entries. With `--backend=tensor_api_cpu` or
`tensor_api_gpu` the steps run on this repo's graphs (the executor of `litert_lm/`); with
`--backend=cpu` or `gpu` they run on LiteRT-LM's own executors and the published
`Qwen3-0.6B.litertlm`. The model file, the tokenizer, the sampler and the timing come from LiteRT-LM
in both cases. Apple M4 Max, 2026-10-20.

## Conditions

- One process per run, with the flags of LiteRT-LM's advanced main:
  `--benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048`,
  `--use_session --async=false`, `--model_path=<Qwen3-0.6B.litertlm>` (the published file for every
  form), `--tensor_api_weights=<model.safetensors>`, `--tensor_api_log_ids=false` and
  `--cache_dir=<dir>`. The benchmark replaces the prompt with 1024 forced tokens. The forms differ
  only in `--backend`, in the flags listed in `litert_lm_speed.json`, and in the cache directory of
  the `--force_f32` form (below).
- Sampler: the file's own (TOP_P, k 20, p 0.95, temperature 0.6), unchanged. LiteRT-LM's GPU
  executor samples on the GPU (its log: `Sampler will handle decode input tensors.`); the Tensor API
  executor returns the logits and LiteRT-LM's CPU sampler picks the token (the registration's
  default, `litert_lm/README.md`). Both are inside the decode time.
- CPU: XNNPACK with 4 threads for every CPU form, LiteRT-LM's default `number_of_threads`; every CPU
  run logged `number_of_threads: 4`, and the Tensor API executor's runs also `CPU threads 4`.
  LiteRT-LM's CPU executor passes the XNNPACK flags default | `DYNAMIC_FULLY_CONNECTED` |
  `ENABLE_LATEST_OPERATORS` (`SetCpuOptions` in `runtime/executor/litert_compiled_model_executor_utils.cc`);
  `--tensor_api_litert_lm_xnnpack_flags` gives the Tensor API executor the same flags, and without it
  the executor keeps LiteRT's default flags, as the harness's own runner does. (The runner of
  iterations 1 to 8 ran XNNPACK without a thread pool, `iterations/ITERATION_8.md`.)
- GPU: LiteRT's WebGPU accelerator for every GPU form, the one this macOS build of LiteRT-LM loads:
  `libLiteRtWebGpuAccelerator.dylib` (sha256
  `b7b7dd7bd6c90a41f6dee7d1f5b2ff76de2df39921f68f1c7e6d525f5136c4cf`) with `libwebgpu_dawn.dylib`
  (sha256 `5541ab53c035d9c054b9cdacf2c6e5a0191618732c820a2f08fb0ea4aa9e97bc`), from the checkout's
  `prebuilt/macos_arm64`. Every GPU run logged `Initializing WebGPU-based API` and the adapter
  `Apple M4 Max, arch=metal-3, vendor=apple, backend=Metal`, and no other GPU API. LiteRT-LM's GPU
  executor runs fp16 activations by default and fp32 with `--force_f32`; the Tensor API executor ran
  at precision fp32 (`--tensor_api_gpu_precision=fp32`) except in one fp16 form.
- Caches: LiteRT-LM's executors keep their compiled programs and converted weights per model file in
  `--cache_dir`. The cache file names carry the model's name, time and size but not the precision
  (`runtime/executor/llm_executor_settings_utils.cc`), so the `--force_f32` form had a directory of
  its own. The Tensor API executor builds and compiles its graphs at every start; that is in its
  init time, not in the prefill time, the decode time or the time to first token.
- Order: every run below in one quiet window of the machine (no other measurement or build ran; the
  desktop and its applications did, load average 3.9 before and 6.1 after), 2026-10-20 from
  01:57:17 to 02:12:11 JST. The nine main forms ran in turn four times, the first form shifting by
  one each time, and each form's first run was discarded. A Tensor API form and a LiteRT-LM form on
  the same accelerator whose decode medians were within 10 % of each other got alternating runs up
  to 8 measured runs each (only the CPU pair `tapi_cpu` and `lm_cpu`). The three extra forms ran
  after the main ones. 5 s between runs. `pmset -g therm` recorded no thermal or performance warning
  before or after.
- Values: as printed by LiteRT-LM's `BenchmarkInfo` (prefill and decode speed, time to first token,
  init phases), and from `/usr/bin/time -l` of the process (`peak memory footprint`, `maximum
  resident set size`; MB = 10^6 bytes). Each entry is median [min–max] (number of measured runs).
  `litert_lm_speed.json` holds every run, the discarded ones included.

## Versions and files

- Binary: `//tensor_api/qwen3-0.6b:litert_lm_advanced_main_tensor_api`, 23,856,848 bytes, sha256
  `0195ed42ec32f072f7a1903ee88f58d00027827260946bf61318278a44861860`, the same file before and after
  the runs (LiteRT-LM `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33`, LiteRT
  `9b8c7f77e7a29e6073e614a05e1e38be7b761229`; the build is in `litert_lm_registry.md`).
- Published model: `litert-community/Qwen3-0.6B` at `e6687177e00897731e7716267eebd4eba0dde7c2`,
  `Qwen3-0.6B.litertlm`, 613,722,264 bytes, sha256
  `c5f39b4191926786c3e6668b62af4d6eb15248dfc7a5348252a9b406f33f2108`. How its weights are stored was
  not inspected.
- Weights of the Tensor API executor: `Qwen/Qwen3-0.6B`, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`, `model.safetensors`; the int8 forms quantize them
  while loading.
- Machine: Apple M4 Max (12 performance and 4 efficiency cores, 128 GB), macOS 27.0 (26A428).

## Table

"output match" counts the greedy ids equal to the PyTorch reference before each prompt's first
mismatch, out of 48, through the same binary: `litert_lm_registry.md` for the main forms, this
iteration for the three extra forms (`tapi_cpu_int8_lmf32_xf`, `tapi_gpu_int8_lmf32`,
`tapi_gpu_fp16`; their ids are in `litert_lm_speed.json`). The bar for the Tensor API forms is at
least 44; LiteRT-LM's own executors are recorded without a bar. "int8" weights are the decoder's
matmul weights, per output channel. The Tensor API graphs keep fp32 activations; on the CPU,
XNNPACK quantizes the inputs of the int8 matmuls per row with LiteRT's default flags
(`xnnpack_int8_fc.md`), which was not inspected with LiteRT-LM's flags.

| form | executor | weights | precision, options | output match | prefill tok/s | decode tok/s | time to first token, s | peak memory footprint, MB | maximum resident set size, MB |
|---|---|---|---|---|---|---|---|---|---|
| `lm_cpu` | LiteRT-LM CPU (XNNPACK) | the published file | LiteRT-LM's defaults | 31/48 | 654.25 [472.99–691.95] (8) | 29.19 [24.89–30.32] (8) | 1.60 [1.51–2.20] (8) | 1198 [1196–1211] (8) | 1892 [1888–1903] (8) |
| `tapi_cpu_xf` | Tensor API CPU (XNNPACK) | fp32 | XNNPACK flags of LiteRT-LM's CPU executor | 48/48 passes | 993.96 [964.13–996.71] (3) | 25.21 [25.06–30.11] (3) | 1.07 [1.06–1.10] (3) | 11102 [11100–11103] (3) | 11164 [11162–11165] (3) |
| `tapi_cpu_int8_xf` | Tensor API CPU (XNNPACK) | int8 per-channel + int8 lm_head copy | XNNPACK flags of LiteRT-LM's CPU executor | 48/48 passes | 1220.91 [397.22–1248.83] (3) | 48.62 [36.83–48.88] (3) | 0.86 [0.85–2.60] (3) | 5083 [4762–5236] (3) | 5152 [4833–5305] (3) |
| `tapi_cpu` | Tensor API CPU (XNNPACK) | fp32 | LiteRT's default XNNPACK flags | 48/48 passes | 1053.65 [978.95–1084.26] (8) | 30.36 [29.49–31.50] (8) | 1.00 [0.98–1.08] (8) | 11102 [11099–11103] (8) | 11164 [11162–11165] (8) |
| `tapi_cpu_int8` | Tensor API CPU (XNNPACK) | int8 per-channel + int8 lm_head copy | LiteRT's default XNNPACK flags | 48/48 passes | 1209.30 [1090.31–1269.71] (3) | 48.91 [48.85–48.96] (3) | 0.87 [0.83–0.96] (3) | 4764 [4761–5081] (3) | 4834 [4830–5149] (3) |
| `lm_gpu` | LiteRT-LM GPU (WebGPU) | the published file | fp16 activations (LiteRT-LM's default) | 48/48 | 2848.69 [2809.26–2901.01] (3) | 142.75 [141.88–144.03] (3) | 0.37 [0.36–0.37] (3) | 1434 [1434–1439] (3) | 369 [368–371] (3) |
| `lm_gpu_f32` | LiteRT-LM GPU (WebGPU) | the published file | fp32 activations (`--force_f32`) | not run | 2312.60 [2130.61–2336.39] (3) | 101.75 [100.86–102.00] (3) | 0.45 [0.45–0.49] (3) | 2423 [2423–2424] (3) | 372 [372–375] (3) |
| `tapi_gpu` | Tensor API GPU (WebGPU) | fp32 | precision fp32 | 48/48 passes | 4209.83 [4006.50–4246.53] (3) | 60.88 [60.41–61.39] (3) | 0.26 [0.26–0.27] (3) | 25812 [25181–25949] (3) | 18207 [18143–18485] (3) |
| `tapi_gpu_int8` | Tensor API GPU (WebGPU) | int8 per-channel + int8 lm_head copy | precision fp32 | 48/48 passes | 4059.90 [3765.10–4110.09] (3) | 78.78 [74.13–80.24] (3) | 0.26 [0.26–0.29] (3) | 18099 [18048–18182] (3) | 12532 [12399–12578] (3) |
| `tapi_cpu_int8_lmf32_xf` | Tensor API CPU (XNNPACK) | int8 per-channel, lm_head fp32 | XNNPACK flags of LiteRT-LM's CPU executor | 48/48 passes | 1199.08 [1144.67–1202.97] (3) | 42.00 [32.55–42.10] (3) | 0.88 [0.87–0.92] (3) | 5396 [5388–5399] (3) | 5465 [5456–5466] (3) |
| `tapi_gpu_int8_lmf32` | Tensor API GPU (WebGPU) | int8 per-channel, lm_head fp32 | precision fp32 | 48/48 passes | 4043.78 [4024.28–4055.64] (3) | 74.51 [74.18–74.70] (3) | 0.27 [0.27–0.27] (3) | 19394 [19294–19602] (3) | 13027 [12984–13162] (3) |
| `tapi_gpu_fp16` | Tensor API GPU (WebGPU) | fp32 | precision fp16 | 0/48 misses | 5299.52 [5262.93–5376.99] (3) | 85.68 [81.42–85.99] (3) | 0.20 [0.20–0.21] (3) | 17533 [17532–17878] (3) | 13872 [13816–13967] (3) |

## Reading the table

The figures are the medians of the table.

- CPU, the published file on LiteRT-LM's CPU executor against the Tensor API executor with int8
  weights and the int8 lm_head copy, both with LiteRT-LM's XNNPACK flags and 4 threads: decode 29.19
  against 48.62 tok/s (the Tensor API executor 1.67 times faster), prefill 654.25 against 1220.91
  tok/s (1.87 times), time to first token 1.60 against 0.86 s. The Tensor API process peaked at
  5083 MB of memory footprint, the LiteRT-LM one at 1198 MB.
- GPU, the published file on LiteRT-LM's GPU executor with fp32 activations against the Tensor API
  executor with int8 weights and the int8 lm_head copy at fp32: decode 101.75 against 78.78 tok/s
  (LiteRT-LM 1.29 times faster), prefill 2312.60 against 4059.90 tok/s (the Tensor API executor 1.76
  times faster), time to first token 0.45 against 0.26 s, memory footprint 2423 against 18099 MB.
  With its default fp16 activations LiteRT-LM's GPU executor decodes at 142.75 tok/s and prefills
  at 2848.69 tok/s, at 1434 MB.
- The Tensor API graphs at GPU precision fp16 decode at 85.68 tok/s and match 0 of 48 ids: each
  prompt repeats one id from its first generated token, the same first wrong id per prompt as Metal
  at fp16 in iteration 2 (`text_match_metal_fp16.txt`).
- With 8 runs each, the fp32 Tensor API form with LiteRT's default XNNPACK flags decodes 4.0 %
  faster than the published file on LiteRT-LM's CPU executor (30.36 against 29.19 tok/s; the ranges
  overlap). Other forms within 10 % of each other ran 3 times each (for example the two int8 CPU
  forms, or the int8 GPU forms with and without the int8 lm_head), and their order is not
  established here.
- The Tensor API executor builds and compiles its graphs at every start: init 0.90 to 1.95 s on the
  CPU and 8.2 to 12.9 s on the GPU over the measured runs, against 0.11 to 0.13 s for LiteRT-LM's
  executors, which read their caches.
- Some CPU runs were slower than the rest of their form (the minimum columns; for example
  `tapi_cpu_int8_xf`'s first measured run prefilled at 397.22 tok/s and its second decoded at 36.83
  tok/s). Their cause was not measured.

## Not measured

- LiteRT-LM's Metal accelerator: this build loads WebGPU on macOS; a Metal build of LiteRT-LM is a
  different build setting and was not made.
- Cold start: each form's first run was discarded (its values are in `litert_lm_speed.json`).
- Other prefill lengths, decode lengths and KV cache lengths; batch sizes above 1; the int4 forms;
  energy; phones.
- The output match of the published file with `--force_f32`.
- Where the time of a decode step goes (graph, logits readback, sampler), and where the Tensor API
  process's memory goes. The decode graph attends over all 2048 cache entries at every step, with a
  mask (`graph/qwen3_graph.cc`); how much of the decode time that takes was not measured.
