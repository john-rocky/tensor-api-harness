# Where the Tensor API executor's GPU decode, memory and init go inside LiteRT-LM, and what moves them

Iteration 9 (`litert_lm_speed.md`) timed this repo's graphs as a LiteRT-LM executor next to
LiteRT-LM's own executors on the published `Qwen3-0.6B.litertlm`, with LiteRT-LM's benchmark entry
(forced prefill 1024, decode 256, KV cache 2048) on an Apple M4 Max. On the GPU the Tensor API
executor decoded slower than LiteRT-LM's executor with fp32 activations, its process peaked at 4 to
13 times LiteRT-LM's memory footprint, and its GPU init took 8 to 13 s. This page measures where a
decode step, the memory and the init go, and tries the levers below, all written down before the
runs. Apple M4 Max, 2026-10-20.

## Levers

All are flags of the backend (`litert_lm/README.md`), off by default.

- L1, graph forms: `--tensor_api_norms=composite` (every RMSNorm as the `odml.rms_norm` composite),
  `--tensor_api_attention=sdpa` (the attention core as the `odml.scaled_dot_product_attention`
  composite), both. These are iteration 3's graph options (`attention_forms.md`).
- L2, memory: `--tensor_api_gpu_share_constants` (the GPU's constant tensors shared between the
  three signatures), `--tensor_api_model_file=fresh` (the serialized model written to the cache
  directory and compiled from the mapped file; the graph, its weight tensors and the in-memory model
  are released before the compile), `--tensor_api_xnnpack_weight_cache` (an XNNPACK weight cache
  file, as LiteRT-LM's CPU executor uses).
- L3, the parts of a decode step, timed inside the executor (`--tensor_api_report`).
- L4, init: `--tensor_api_gpu_program_cache` (the compiled programs and, with shared constants, the
  converted weights serialized to the cache directory), `--tensor_api_model_file=reuse` (a model file
  of the same key compiled without loading the checkpoint or building the graph).
- L5, the other compile options of LiteRT-LM's GPU executor: `--tensor_api_gpu_external_caches`
  (`stock`: the KV caches as external tensor patterns with the external tensors mode off, as
  LiteRT-LM's GPU executor does for its `kv_cache_` tensors; `runner`: with the mode on, as
  `LitertDynamicRunner` does for feedback loops), `--tensor_api_gpu_litert_lm_options` (the rest of
  that executor's options as one set), `--tensor_api_gpu_command_buffer_steps=2`.
- Combined, chosen from the first window's results by a rule written before it (each lever that
  passes the output match; a graph form or a decode option only if its single run there was at least
  3 % faster): `G4best` (shared constants, program cache, model file reused, LiteRT-LM's GPU options)
  and `C3best` (XNNPACK weight cache, model file reused).

Two items of the lever list were settled by the source before any run. A compiled model made from a
buffer reads that buffer in place (LiteRT `litert/core/util/flatbuffer_tools.cc`: `MakeAllocation`
wraps it in a non-owning `tflite::MemoryAllocation`), so the executor cannot free its serialized
model after the compile; the model-file lever maps a file instead. And the executor compiles one
model once (`tensor_api_executor.cc`; a second compile happens only with
`--tensor_api_delegation_report`), so there is no second compiled model to drop.

## Conditions

- As in iteration 9 (`litert_lm_speed.md`): one process per run with
  `--benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048`,
  `--use_session --async=false`, the published file as `--model_path` for every form, the file's own
  sampler (TOP_P, k 20, p 0.95, temperature 0.6), 4 XNNPACK threads, LiteRT's WebGPU accelerator for
  every GPU form, `/usr/bin/time -l` for the memory, MB = 10^6 bytes.
- Every Tensor API run carried `--tensor_api_report` (the init phases, the footprint after each
  stage, and the parts of each decode step). Its cost per decode step is seven clock reads and one
  record; it is part of every Tensor API number on this page, including the re-timed forms of
  iteration 9 (`C3`, `G4`).
- Every Tensor API form had a cache directory of its own; the cache forms wrote their caches in the
  first window, so every timed run read them. LiteRT-LM's executors used iteration 9's directories.
- Two quiet windows of the machine. The first (2026-10-20 04:46:11 to 04:58:18 JST) built the binary
  and ran the output match, the breakdown runs and the footprint below; its numbers are not in the
  tables except where marked. The second (07:57:22 to 08:19:14 JST) timed everything in the tables:
  the forms in turn four times, the first form shifting by one each time, each form's first run
  discarded; pairs within 10 % in decode got alternating runs to 8 measured each (`C3` and `C3best`,
  `G4` and `G4best`). 5 s between runs. `pmset -g therm` recorded no thermal or performance warning
  before or after; the desktop and its applications kept running (load average 4.6 before, 5.4
  after).
- Values: LiteRT-LM's `BenchmarkInfo` (prefill and decode speed, time to first token, `Init Total`),
  `/usr/bin/time -l` (`peak memory footprint`, `maximum resident set size`), and the report lines.
  Each entry is median [min–max] (number of measured runs). `litert_lm_levers.json` holds every run.

## Versions and files

- Binary: `//tensor_api/qwen3-0.6b:litert_lm_advanced_main_tensor_api` built from this iteration's
  `litert_lm/`, 23,895,168 bytes, sha256
  `3f8e5ad9a7d5ec18af1e6bca2739467ed9be35ff87627032b7d96dc61f606f76`, the same file before and after
  both windows (LiteRT-LM `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33`, LiteRT
  `9b8c7f77e7a29e6073e614a05e1e38be7b761229`).
- GPU libraries, published model and weights: as in iteration 9 (`libLiteRtWebGpuAccelerator.dylib`
  `b7b7dd7b…c4cf`, `libwebgpu_dawn.dylib` `5541ab53…97bc`; `Qwen3-0.6B.litertlm` `c5f39b41…2108`;
  `Qwen/Qwen3-0.6B` `model.safetensors` at `c1899de2…`).
- With every new flag off, the binary gives iteration 7's ids: the six Tensor API forms of
  `litert_lm_registry.md` produce ids files byte-identical to `litert_lm_ids_*.json`, and the
  published file on the GPU too. On the CPU the published file again matches 31 of 48 (5, 16, 10);
  its ids after the first mismatch differ from iteration 7's file, as they did between iteration 7's
  two passes.

## Table

"output match" counts the greedy ids equal to the PyTorch reference before each prompt's first
mismatch, out of 48, through the same binary (3 prompts × 16 tokens, `*.greedy.litertlm`); the bar
for the Tensor API forms is at least 44. "init" is LiteRT-LM's `Init Total`. Every Tensor API form
uses int8 per-channel decoder weights with the int8 lm_head copy (iteration 9's fastest passing
forms); `C3` and its levers use LiteRT-LM's XNNPACK flags; `G4` and its levers run at precision fp32.

| form | lever | change | output match | prefill tok/s | decode tok/s | time to first token, s | init, s | peak memory footprint, MB | maximum resident set size, MB |
|---|---|---|---|---|---|---|---|---|---|
| `C1` | LiteRT-LM | published file on LiteRT-LM's CPU executor | 31/48 | 769.43 [762.01–774.69] (3) | 31.48 [30.91–33.22] (3) | 1.36 [1.35–1.38] (3) | 0.11 [0.11–0.12] (3) | 1195 [1195–1198] (3) | 1887 [1887–1890] (3) |
| `G1` | LiteRT-LM | published file on LiteRT-LM's GPU executor, fp16 activations (its default) | 48/48 | 2822.14 [2791.66–2829.48] (3) | 141.52 [140.89–141.62] (3) | 0.37 [0.37–0.37] (3) | 0.14 [0.13–0.14] (3) | 1436 [1429–1436] (3) | 366 [359–367] (3) |
| `G2` | LiteRT-LM | published file on LiteRT-LM's GPU executor, fp32 activations (--force_f32) | not run | 2226.41 [2217.44–2250.28] (3) | 102.07 [101.89–102.28] (3) | 0.47 [0.46–0.47] (3) | 0.13 [0.13–0.14] (3) | 2421 [2421–2424] (3) | 366 [365–368] (3) |
| `C3` | base | int8 + int8 lm_head copy, LiteRT-LM's XNNPACK flags (iteration 9) | 48/48 passes | 1238.71 [1216.32–1296.49] (8) | 45.23 [44.10–48.23] (8) | 0.85 [0.81–0.86] (8) | 1.84 [1.76–1.97] (8) | 4766 [4763–5224] (8) | 4835 [4832–5291] (8) |
| `G4` | base | int8 + int8 lm_head copy, precision fp32 (iteration 9) | 48/48 passes | 3914.77 [3842.97–3985.58] (8) | 79.37 [79.02–80.07] (8) | 0.28 [0.27–0.28] (8) | 10.69 [10.46–11.73] (8) | 18190 [18022–18212] (8) | 12518 [12461–12570] (8) |
| `G4n` | L1 | G4 + every RMSNorm as the odml.rms_norm composite | 48/48 passes | 3961.62 [3868.57–4037.12] (3) | 80.24 [66.33–81.11] (3) | 0.27 [0.27–0.28] (3) | 10.88 [10.66–11.78] (3) | 18225 [18118–18249] (3) | 12498 [12497–12566] (3) |
| `G4a` | L1 | G4 + the attention core as the odml.scaled_dot_product_attention composite | 48/48 passes | 4325.47 [4193.41–4361.64] (3) | 59.38 [59.33–59.42] (3) | 0.25 [0.25–0.26] (3) | 10.91 [10.69–11.04] (3) | 18211 [18151–18274] (3) | 12518 [12459–12533] (3) |
| `G4na` | L1 | G4 + both composites | 48/48 passes | 4425.00 [4072.04–4452.62] (3) | 59.85 [59.69–59.90] (3) | 0.25 [0.25–0.27] (3) | 10.82 [10.56–10.92] (3) | 18203 [18085–18263] (3) | 12513 [12504–12585] (3) |
| `C3n` | L1 | C3 + every RMSNorm as the odml.rms_norm composite | 48/48 passes | 1201.21 [1194.06–1249.57] (3) | 43.72 [43.27–44.45] (3) | 0.88 [0.84–0.88] (3) | 1.83 [1.81–1.84] (3) | 4745 [4735–4792] (3) | 4816 [4806–4860] (3) |
| `G4c` | L2 | G4 + constant tensors shared between the signatures | 48/48 passes | 3857.47 [3825.56–4036.00] (3) | 79.48 [79.41–79.98] (3) | 0.28 [0.27–0.28] (3) | 2.96 [2.93–2.97] (3) | 10273 [10271–10285] (3) | 7717 [7716–7728] (3) |
| `C3x` | L2 | C3 + an XNNPACK weight cache file | 48/48 passes | 1198.38 [1192.15–1202.97] (3) | 45.04 [38.77–46.02] (3) | 0.88 [0.87–0.88] (3) | 1.13 [1.08–1.19] (3) | 3288 [3288–3602] (3) | 3960 [3959–4101] (3) |
| `C3f` | L2 | C3 + the serialized model as a file, compiled from the mapped file | 48/48 passes | 1253.50 [1251.12–1289.04] (3) | 45.59 [45.02–47.20] (3) | 0.84 [0.82–0.84] (3) | 2.06 [2.06–3.34] (3) | 3450 [3444–3493] (3) | 4130 [4123–4173] (3) |
| `G4cp` | L4 | G4c + the GPU program cache (with the converted shared weights) | 48/48 passes | 3856.54 [3828.87–3915.63] (3) | 79.63 [79.48–79.75] (3) | 0.28 [0.27–0.28] (3) | 1.90 [1.88–1.95] (3) | 8757 [8756–9060] (3) | 7366 [7364–7971] (3) |
| `G4cpr` | L4 | G4cp + the serialized model file reused (no checkpoint loading or graph building) | 48/48 passes | 3729.92 [3706.77–3745.34] (3) | 79.23 [79.21–79.94] (3) | 0.29 [0.29–0.29] (3) | 0.90 [0.85–0.91] (3) | 8084 [8076–8090] (3) | 5318 [5304–5727] (3) |
| `G4o` | L5 | G4 + the other compile options of LiteRT-LM's GPU executor | 48/48 passes | 3912.25 [3756.23–3987.25] (3) | 77.94 [77.62–78.33] (3) | 0.27 [0.27–0.29] (3) | 10.51 [10.39–10.68] (3) | 17786 [17784–17858] (3) | 11953 [11932–12001] (3) |
| `G4ob` | L5 | G4o + command buffers prepared 2 steps ahead | 48/48 passes | 3962.09 [3917.53–4021.56] (3) | 63.29 [60.55–78.40] (3) | 0.27 [0.27–0.28] (3) | 10.72 [10.41–11.92] (3) | 17786 [17783–17835] (3) | 11932 [11917–11993] (3) |
| `G4best` | combined | G4 + `--tensor_api_gpu_share_constants` `--tensor_api_gpu_program_cache` `--tensor_api_model_file=reuse` `--tensor_api_gpu_litert_lm_options` | 48/48 passes | 3804.39 [3715.71–3878.01] (8) | 77.75 [64.34–78.32] (8) | 0.28 [0.28–0.29] (8) | 0.78 [0.67–0.83] (8) | 8083 [8073–8089] (8) | 5616 [5302–5931] (8) |
| `C3best` | combined | C3 + `--tensor_api_xnnpack_weight_cache` `--tensor_api_model_file=reuse` | 48/48 passes | 1226.38 [1173.74–1253.49] (8) | 46.25 [44.49–48.30] (8) | 0.86 [0.84–0.89] (8) | 0.14 [0.14–0.15] (8) | 1966 [1955–1967] (8) | 2654 [2643–2654] (8) |

Forms that could not run, and one that was not timed:

- `C3a` (C3 + the attention composite, one try as written before the runs): the first greedy run
  stopped while the compiled model prepared its tensors, `XNNPack delegate failed to reshape runtime` and
  `Node number 1839 (TfLiteXNNPackDelegate) failed to prepare.` — the same failure as iteration 3 with
  a KV cache of 64.
- `G4k` and `G4kr` (the KV caches as external tensor patterns, mode off and on): the prefill ran,
  and the first decode stopped with `INVALID_ARGUMENT: Input buffer type or shape mismatch` when the
  executor bound the previous call's cache output as the next input (`LitertDynamicRunner` checks the
  bound buffer's type and layout against the signature's input). With these options the executor
  would need another way to pass the caches between signatures; that was not tried.
- `C5` (iteration 9's int8 form with LiteRT's default XNNPACK flags): not timed. Its first run
  stopped at start because its cache directory did not exist; the script made the directories only
  for the forms of the first window.

## Reading the table

The figures are the medians of the table.

- CPU, the published file on LiteRT-LM's CPU executor against the Tensor API executor with the
  XNNPACK weight cache and the model file reused (`C3best`, the smallest and, by its median, the
  fastest passing CPU form): decode 31.48 against 46.25 tok/s (the Tensor API executor 1.47 times
  faster), prefill 769.43 against 1226.38 tok/s, init 0.11 against 0.14 s, peak memory footprint
  1195 against 1966 MB. Before the levers (`C3`) the Tensor API process peaked at 4766 MB and took
  1.84 s to init. `C3best` and `C3` differ by 2.3 % in decode after 8 runs each, with overlapping
  ranges, so the levers are not shown to change the CPU decode speed.
- GPU, the published file on LiteRT-LM's GPU executor with fp32 activations (`G2`) against the
  Tensor API executor: decode 102.07 against 77.75 to 80.24 tok/s for the passing forms other than
  the attention composites and `G4ob` (LiteRT-LM 1.27 times faster than the fastest of them by
  median, `G4n`), prefill 2226.41 against 3729.92 to 3961.62 tok/s for the same forms (the Tensor API
  executor faster). `G4ob` decodes at 63.29 tok/s by median because two of its three runs contain
  long steps (below); its median step is that of `G4o`. The fastest passing GPU forms (`G4n`, `G4cp`,
  `G4c`, `G4`, `G4cpr`) lie within 1.3 % of each other; their order is not established. The smallest passing GPU forms, `G4best` and `G4cpr`, peak at 8083 and 8084 MB
  against 18190 MB for `G4` and 2421 MB for `G2`, and init in 0.78 and 0.90 s against 10.69 s for `G4`
  and 0.13 s for `G2`. With its default fp16 activations LiteRT-LM's GPU executor decodes at 141.52
  tok/s at 1436 MB.
- The levers that change memory and init do not change the GPU decode: shared constants, the program
  cache and the reused model file decode at 79.23 to 79.63 tok/s against 79.37 for `G4`. LiteRT-LM's
  remaining GPU options decode at 77.94 tok/s; `G4best`, which includes them, at 77.75.
- The attention composite slows the GPU decode to 59.38 tok/s (59.85 with the norm composite too)
  while it passes the output match; the norm composite alone gives 80.24 against 79.37 tok/s, within
  the spread.

## Where a GPU decode step goes

The parts of a decode step as timed inside the executor (`--tensor_api_report`): the median over a
run's 256 steps, then median [min–max] over the measured runs. "between calls" is the time from one
step's return to the next call, which is LiteRT-LM's sampler and session; "step, ms" is 1000 divided
by the decode speed of the table.

| form | step, ms (from decode tok/s) | inputs + bind, µs | Run, µs | caches, µs | logits readback, µs | copy out, µs | whole call, µs | between calls, µs |
|---|---|---|---|---|---|---|---|---|
| `C1` | 31.77 | — | — | — | — | — | — | — |
| `G1` | 7.07 | — | — | — | — | — | — | — |
| `G2` | 9.80 | — | — | — | — | — | — | — |
| `C3` | 22.11 | 29.1 [28.1–31.5] (8) | 21692.1 [20453.4–22319.2] (8) | 17.4 [16.8–18.1] (8) | 9.7 [9.1–10.5] (8) | 7.5 [7.3–7.8] (8) | 21756.2 [20518.5–22384.8] (8) | 51.2 [49.8–52.5] (8) |
| `G4` | 12.60 | 39.2 [32.5–62.5] (8) | 1797.2 [1596.1–2049.6] (8) | 18.8 [15.6–26.6] (8) | 10705.0 [10246.4–10887.1] (8) | 9.0 [8.2–9.3] (8) | 12514.5 [12354.4–12587.5] (8) | 57.2 [51.9–59.1] (8) |
| `G4n` | 12.46 | 56.8 [41.1–61.2] (3) | 1836.3 [1678.3–1847.8] (3) | 20.1 [18.5–23.8] (3) | 10486.5 [10259.4–10635.4] (3) | 9.2 [8.8–9.7] (3) | 12386.1 [12208.5–12422.1] (3) | 57.5 [56.2–60.7] (3) |
| `G4a` | 16.84 | 58.1 [47.4–60.1] (3) | 2259.2 [2074.4–2273.9] (3) | 19.4 [18.1–24.0] (3) | 14392.1 [14376.3–14693.8] (3) | 9.5 [8.8–9.5] (3) | 16767.5 [16755.6–16790.1] (3) | 59.0 [57.0–61.0] (3) |
| `G4na` | 16.71 | 55.5 [47.4–58.8] (3) | 2037.2 [1983.6–2093.9] (3) | 20.5 [19.9–24.2] (3) | 14543.6 [14423.1–14604.8] (3) | 9.1 [9.0–9.4] (3) | 16644.0 [16625.3–16654.8] (3) | 58.3 [58.2–60.2] (3) |
| `C3n` | 22.87 | 30.1 [30.0–31.6] (3) | 22434.1 [22044.9–22712.4] (3) | 17.7 [17.7–17.9] (3) | 9.2 [9.2–9.4] (3) | 7.5 [7.4–7.6] (3) | 22495.9 [22115.8–22782.0] (3) | 52.5 [52.0–52.6] (3) |
| `G4c` | 12.58 | 46.7 [40.2–59.9] (3) | 1879.5 [1787.2–2004.0] (3) | 18.1 [17.1–23.5] (3) | 10569.2 [10352.4–10694.1] (3) | 9.0 [8.8–9.0] (3) | 12527.6 [12402.4–12538.3] (3) | 56.9 [55.8–57.0] (3) |
| `C3x` | 22.20 | 30.0 [29.0–30.6] (3) | 21306.8 [20667.4–21786.4] (3) | 18.3 [16.8–18.8] (3) | 9.5 [9.3–9.8] (3) | 7.4 [7.4–7.6] (3) | 21378.9 [20731.5–21854.3] (3) | 51.0 [51.0–52.5] (3) |
| `C3f` | 21.93 | 29.3 [29.2–29.6] (3) | 21528.0 [20751.5–21814.5] (3) | 17.6 [17.0–18.5] (3) | 9.7 [9.0–9.8] (3) | 7.5 [7.5–7.6] (3) | 21593.8 [20819.9–21877.1] (3) | 51.7 [50.6–52.0] (3) |
| `G4cp` | 12.56 | 56.6 [51.6–57.6] (3) | 1984.5 [1921.5–1990.9] (3) | 20.3 [19.8–22.3] (3) | 10403.6 [10351.2–10499.1] (3) | 9.1 [8.8–9.1] (3) | 12475.9 [12455.8–12479.5] (3) | 57.7 [56.6–58.0] (3) |
| `G4cpr` | 12.62 | 50.3 [49.1–62.2] (3) | 1931.8 [1924.5–2020.5] (3) | 18.6 [17.9–22.5] (3) | 10533.9 [10266.6–10567.6] (3) | 9.1 [9.0–9.3] (3) | 12528.5 [12367.1–12538.6] (3) | 58.4 [57.6–59.8] (3) |
| `G4o` | 12.83 | 35.6 [30.1–41.4] (3) | 12366.8 [12334.5–12404.7] (3) | 19.4 [19.2–19.8] (3) | 277.0 [260.0–299.7] (3) | 8.7 [8.2–8.7] (3) | 12699.3 [12696.4–12745.2] (3) | 54.5 [52.5–57.9] (3) |
| `G4ob` | 15.80 | 41.1 [40.5–45.8] (3) | 12428.1 [12206.2–12431.8] (3) | 22.9 [21.2–24.5] (3) | 297.5 [290.0–308.4] (3) | 9.0 [8.7–9.1] (3) | 12784.8 [12613.6–12788.4] (3) | 59.6 [58.5–60.1] (3) |
| `G4best` | 12.86 | 35.5 [32.9–46.1] (8) | 12413.8 [12210.0–12437.3] (8) | 19.9 [18.1–20.3] (8) | 274.8 [264.6–314.5] (8) | 8.9 [8.0–9.7] (8) | 12749.5 [12624.1–12790.4] (8) | 58.0 [53.5–61.2] (8) |
| `C3best` | 21.62 | 29.9 [28.5–31.0] (8) | 21205.5 [20487.0–22072.7] (8) | 16.9 [15.1–17.6] (8) | 9.7 [9.5–10.7] (8) | 7.3 [7.2–7.5] (8) | 21270.5 [20553.7–22137.0] (8) | 51.6 [50.0–53.0] (8) |

- With the executor's default options, WebGPU's `Run` returns once the work is enqueued (1.80 ms
  for `G4`), and reading the logits back waits for the GPU (10.71 ms). With LiteRT-LM's GPU options
  (`G4o`, `G4best`), whose benchmark setting waits actively for each run, `Run` includes the GPU work
  (12.37 and 12.41 ms) and the logits readback is 0.28 ms. So a GPU decode step of this graph is about
  12.4 ms of GPU work, 0.3 ms of logits readback, 0.06 ms of LiteRT-LM's sampler and session between
  steps, and well under 0.1 ms of the executor's own bookkeeping. LiteRT-LM's executor on the
  published file takes 9.80 ms per step at fp32 (`G2`), sampling included.
- The logits readback (151,936 fp32 values per step, 0.3 ms) and the CPU sampling (0.06 ms) are 2 to
  3 % of the step. LiteRT-LM's own GPU executor samples on the GPU; for this executor that would save
  at most about 0.3 ms per step.
- On the CPU the decode step is `Run` (21.2 to 22.4 ms by form); the logits readback is 0.01 ms and
  the time between calls 0.05 ms.
- Runs with a lower decode speed than the rest of their form (the minimum columns: one run each of
  `G4n`, `G4best` and `C3x`, two of `G4ob`) have longest steps of 80 to 735 ms while their median
  steps match the other runs. Their cause was not measured.

## Where the memory goes

The physical footprint after each stage, its peak so far, and the part tagged as graphics, from the
report lines (macOS `task_info`), median [min–max] over the measured runs.

| form | stage | footprint, MB | peak so far, MB | graphics, MB |
|---|---|---|---|---|
| `C3` | weights loaded | 1543 [1535–1545] (8) | 1543 [1536–1546] (8) | 0 [0–0] (8) |
| `C3` | model built | 2818 [2811–2820] (8) | 2818 [2811–2820] (8) | 0 [0–0] (8) |
| `C3` | compiled | 4679 [4671–4682] (8) | 4679 [4671–4682] (8) | 0 [0–0] (8) |
| `C3` | initialized | 4696 [4687–4699] (8) | 4696 [4687–4699] (8) | 0 [0–0] (8) |
| `C3` | first prefill (1023 tokens) | 4460 [4457–5224] (8) | 4697 [4696–5224] (8) | 0 [0–0] (8) |
| `C3` | first decode step | 4765 [4755–4916] (8) | 4766 [4763–5224] (8) | 0 [0–0] (8) |
| `C3` | decode steps reported | 4765 [4755–4917] (8) | 4766 [4763–5224] (8) | 0 [0–0] (8) |
| `C3` | executor teardown | 4765 [4755–4917] (8) | 4766 [4763–5224] (8) | 0 [0–0] (8) |
| `G4` | weights loaded | 1545 [1535–1548] (8) | 1546 [1536–1549] (8) | 0 [0–0] (8) |
| `G4` | model built | 2820 [2811–2823] (8) | 2820 [2811–2823] (8) | 0 [0–0] (8) |
| `G4` | compiled | 15272 [15200–15390] (8) | 18190 [18022–18212] (8) | 6519 [6514–6676] (8) |
| `G4` | initialized | 15285 [15214–15404] (8) | 18190 [18022–18212] (8) | 6519 [6514–6676] (8) |
| `G4` | first prefill (1023 tokens) | 15893 [15737–16439] (8) | 18190 [18022–18212] (8) | 7826 [7825–7827] (8) |
| `G4` | first decode step | 16358 [16201–16903] (8) | 18190 [18022–18212] (8) | 8290 [8290–8292] (8) |
| `G4` | decode steps reported | 16353 [16196–16897] (8) | 18190 [18022–18212] (8) | 8285 [8285–8285] (8) |
| `G4` | executor teardown | 16353 [16196–16897] (8) | 18190 [18022–18212] (8) | 8285 [8285–8285] (8) |
| `G4c` | weights loaded | 1545 [1535–1545] (3) | 1546 [1536–1546] (3) | 0 [0–0] (3) |
| `G4c` | model built | 2820 [2811–2821] (3) | 2820 [2811–2821] (3) | 0 [0–0] (3) |
| `G4c` | compiled | 10270 [9664–10272] (3) | 10272 [10270–10285] (3) | 3283 [2660–3283] (3) |
| `G4c` | initialized | 10271 [9665–10273] (3) | 10273 [10271–10285] (3) | 3283 [2660–3283] (3) |
| `G4c` | first prefill (1023 tokens) | 9104 [8502–9109] (3) | 10273 [10271–10285] (3) | 4412 [4412–4412] (3) |
| `G4c` | first decode step | 9568 [8966–9573] (3) | 10273 [10271–10285] (3) | 4876 [4876–4876] (3) |
| `G4c` | decode steps reported | 9563 [8961–9567] (3) | 10273 [10271–10285] (3) | 4871 [4871–4871] (3) |
| `G4c` | executor teardown | 9563 [8961–9567] (3) | 10273 [10271–10285] (3) | 4871 [4871–4871] (3) |
| `C3x` | weights loaded | 1543 [1543–1545] (3) | 1543 [1543–1546] (3) | 0 [0–0] (3) |
| `C3x` | model built | 2818 [2818–2822] (3) | 2818 [2818–2822] (3) | 0 [0–0] (3) |
| `C3x` | compiled | 3209 [3209–3211] (3) | 3209 [3209–3211] (3) | 0 [0–0] (3) |
| `C3x` | initialized | 3225 [3225–3227] (3) | 3225 [3225–3227] (3) | 0 [0–0] (3) |
| `C3x` | first prefill (1023 tokens) | 3142 [2982–3602] (3) | 3225 [3225–3602] (3) | 0 [0–0] (3) |
| `C3x` | first decode step | 3288 [3287–3288] (3) | 3288 [3287–3602] (3) | 0 [0–0] (3) |
| `C3x` | decode steps reported | 3288 [3288–3288] (3) | 3288 [3288–3602] (3) | 0 [0–0] (3) |
| `C3x` | executor teardown | 3288 [3288–3288] (3) | 3288 [3288–3602] (3) | 0 [0–0] (3) |
| `C3f` | weights loaded | 1535 [1535–1546] (3) | 1536 [1536–1547] (3) | 0 [0–0] (3) |
| `C3f` | model built | 2811 [2811–2822] (3) | 2811 [2811–2822] (3) | 0 [0–0] (3) |
| `C3f` | graph and buffer released | 359 [354–367] (3) | 2811 [2811–2822] (3) | 0 [0–0] (3) |
| `C3f` | compiled | 2003 [1997–2047] (3) | 2811 [2811–2822] (3) | 0 [0–0] (3) |
| `C3f` | initialized | 2019 [2013–2063] (3) | 2811 [2811–2822] (3) | 0 [0–0] (3) |
| `C3f` | first prefill (1023 tokens) | 2978 [2972–3022] (3) | 2978 [2972–3022] (3) | 0 [0–0] (3) |
| `C3f` | first decode step | 3450 [3444–3493] (3) | 3450 [3444–3493] (3) | 0 [0–0] (3) |
| `C3f` | decode steps reported | 3450 [3444–3493] (3) | 3450 [3444–3493] (3) | 0 [0–0] (3) |
| `C3f` | executor teardown | 3450 [3444–3493] (3) | 3450 [3444–3493] (3) | 0 [0–0] (3) |
| `G4cp` | weights loaded | 1545 [1543–1545] (3) | 1546 [1543–1546] (3) | 0 [0–0] (3) |
| `G4cp` | model built | 2820 [2819–2820] (3) | 2820 [2819–2820] (3) | 0 [0–0] (3) |
| `G4cp` | compiled | 8703 [8696–8704] (3) | 8704 [8703–9060] (3) | 2675 [2675–2675] (3) |
| `G4cp` | initialized | 8704 [8697–8705] (3) | 8705 [8704–9060] (3) | 2675 [2675–2675] (3) |
| `G4cp` | first prefill (1023 tokens) | 8014 [7873–8029] (3) | 8757 [8756–9060] (3) | 3812 [3812–3812] (3) |
| `G4cp` | first decode step | 8478 [8337–8495] (3) | 8757 [8756–9060] (3) | 4277 [4277–4277] (3) |
| `G4cp` | decode steps reported | 8477 [8335–8493] (3) | 8757 [8756–9060] (3) | 4275 [4275–4275] (3) |
| `G4cp` | executor teardown | 8477 [8335–8493] (3) | 8757 [8756–9060] (3) | 4275 [4275–4275] (3) |
| `G4cpr` | compiled | 6008 [6002–6015] (3) | 6008 [6002–6015] (3) | 2675 [2675–2675] (3) |
| `G4cpr` | initialized | 6009 [6003–6017] (3) | 6009 [6003–6017] (3) | 2675 [2675–2675] (3) |
| `G4cpr` | first prefill (1023 tokens) | 7618 [7611–7625] (3) | 7618 [7611–7625] (3) | 3812 [3812–3812] (3) |
| `G4cpr` | first decode step | 8083 [8076–8090] (3) | 8083 [8076–8090] (3) | 4277 [4277–4277] (3) |
| `G4cpr` | decode steps reported | 8081 [8074–8088] (3) | 8084 [8076–8090] (3) | 4275 [4275–4275] (3) |
| `G4cpr` | executor teardown | 8081 [8074–8088] (3) | 8084 [8076–8090] (3) | 4275 [4275–4275] (3) |
| `G4best` | compiled | 6007 [5997–6014] (8) | 6012 [5999–6627] (8) | 2675 [2675–2675] (8) |
| `G4best` | initialized | 6008 [5998–6015] (8) | 6014 [6000–6627] (8) | 2675 [2675–2675] (8) |
| `G4best` | first prefill (1023 tokens) | 7618 [7607–7624] (8) | 7618 [7607–7624] (8) | 3812 [3812–3812] (8) |
| `G4best` | first decode step | 8082 [8072–8088] (8) | 8082 [8072–8088] (8) | 4276 [4276–4276] (8) |
| `G4best` | decode steps reported | 8081 [8071–8087] (8) | 8083 [8073–8089] (8) | 4275 [4275–4275] (8) |
| `G4best` | executor teardown | 8081 [8071–8087] (8) | 8083 [8073–8089] (8) | 4275 [4275–4275] (8) |
| `C3best` | compiled | 518 [507–519] (8) | 518 [507–519] (8) | 0 [0–0] (8) |
| `C3best` | initialized | 535 [524–536] (8) | 535 [524–536] (8) | 0 [0–0] (8) |
| `C3best` | first prefill (1023 tokens) | 1494 [1483–1495] (8) | 1494 [1483–1495] (8) | 0 [0–0] (8) |
| `C3best` | first decode step | 1966 [1955–1966] (8) | 1966 [1955–1966] (8) | 0 [0–0] (8) |
| `C3best` | decode steps reported | 1966 [1955–1967] (8) | 1966 [1955–1967] (8) | 0 [0–0] (8) |
| `C3best` | executor teardown | 1966 [1955–1967] (8) | 1966 [1955–1967] (8) | 0 [0–0] (8) |

- `G4` peaks during the compile (18190 MB), after which the process holds 15272 MB, of which 6519 MB
  are graphics. The weights loaded as tensors (1545 MB) and the serialized model (1275 MB more by
  "model built") are held through the compile.
- Shared constants (`G4c`) drop the compile's graphics from 6519 to 3283 MB and the peak from 18190 to
  10273 MB, and the compile from 9.69 to 1.91 s. LiteRT describes the option as preventing the
  duplication of constant tensors across subgraphs (`litert/cc/options/litert_gpu_options.h`); here
  the three signatures share every weight.
- The reused model file on top of shared constants and the program cache (`G4cpr`, `G4best`) skips
  the weight tensors and the in-memory model: the process holds 6008 MB after the compile, and the
  peak comes at the first decode step (8083 MB), with 4276 MB of graphics.
- On the CPU, the XNNPACK weight cache (`C3x`) drops the footprint after the compile from 4679 to
  3209 MB; releasing the graph and the in-memory model before compiling from a mapped file (`C3f`)
  drops it to 359 MB before the compile and 2003 MB after; both with the reused model file
  (`C3best`): 518 MB after the compile.
- What remains in `C3best` (1966 MB at the first decode step) comes with the first prefill (+959 MB)
  and the first decode (+472 MB): the executor's caches. `LitertDynamicRunner::Create` makes the input
  and output buffers of every signature, 336 KV cache buffers of 8.4 MB (2819 MB) for three
  signatures, and the CPU's footprint counts the pages written. LiteRT-LM's executors keep their
  caches in one state object used by both the prefill and the decode signature (`LitertState` in
  `runtime/executor/llm_litert_compiled_model_executor.cc`); one set of caches for every signature
  was not tried here.
- The footprint of a held `G4` run after its decode steps (`footprint` on the process, first window,
  one run): 16 GB, of which `Owned physical footprint (unmapped) (graphics)` 7889 MB and
  `Malloc Large` 7737 MB, as `footprint` prints them.

The first window's single runs of the fp32-weight forms, for comparison (one run each, not in the
table): `G3`, fp32 weights on the GPU at precision fp32, and `C2`, fp32 weights on the CPU with
LiteRT-LM's XNNPACK flags (iteration 9's `tapi_gpu` and `tapi_cpu_xf`).

| run | stage | footprint, MB | peak so far, MB | graphics, MB |
|---|---|---|---|---|
| `G3_base` | weights loaded | 2491 | 2491 | 0 |
| `G3_base` | model built | 4882 | 4882 | 0 |
| `G3_base` | compiled | 19547 | 25180 | 8081 |
| `G3_base` | initialized | 19547 | 25180 | 8081 |
| `G3_base` | first prefill (1023 tokens) | 18783 | 25180 | 9613 |
| `G3_base` | first decode step | 19248 | 25180 | 10077 |
| `G3_base` | decode steps reported | 19239 | 25180 | 10069 |
| `G3_base` | executor teardown | 19239 | 25180 | 10069 |
| `C2_base` | weights loaded | 2488 | 2489 | 0 |
| `C2_base` | model built | 4880 | 4880 | 0 |
| `C2_base` | compiled | 11084 | 11084 | 0 |
| `C2_base` | initialized | 11101 | 11101 | 0 |
| `C2_base` | first prefill (1023 tokens) | 10820 | 11101 | 0 |
| `C2_base` | first decode step | 10170 | 11101 | 0 |
| `C2_base` | decode steps reported | 10171 | 11101 | 0 |
| `C2_base` | executor teardown | 10171 | 11101 | 0 |

## Where the init goes

Init phases inside the executor (ms): loading the checkpoint and quantizing, building the graphs,
writing or reading the model file, the environment and options, the compile (LiteRT's compiled model
and the runner's buffers), the rest; then LiteRT-LM's `Init Total` for the whole engine.

| form | load + quantize | build | model file | environment, options | compile | rest | total (executor) | Init Total (LiteRT-LM) |
|---|---|---|---|---|---|---|---|---|
| `C3` | 843 [820–919] (8) | 107 [69–135] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 858 [850–921] (8) | 1 [1–1] (8) | 1830 [1750–1956] (8) | 1844 [1761–1969] (8) |
| `G4` | 841 [826–920] (8) | 102 [72–136] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 9692 [9526–10821] (8) | 1 [1–1] (8) | 10674 [10450–11722] (8) | 10686 [10463–11735] (8) |
| `G4n` | 881 [850–920] (3) | 118 [116–137] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 9805 [9676–10769] (3) | 1 [1–1] (3) | 10863 [10643–11769] (3) | 10875 [10657–11782] (3) |
| `G4a` | 869 [862–903] (3) | 124 [116–128] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 9901 [9699–9999] (3) | 1 [1–1] (3) | 10898 [10678–11028] (3) | 10912 [10689–11042] (3) |
| `G4na` | 870 [844–917] (3) | 125 [113–138] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 9813 [9594–9856] (3) | 1 [1–1] (3) | 10808 [10552–10912] (3) | 10822 [10561–10925] (3) |
| `C3n` | 847 [846–867] (3) | 124 [110–128] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 833 [827–851] (3) | 1 [1–1] (3) | 1811 [1800–1827] (3) | 1826 [1815–1841] (3) |
| `G4c` | 875 [871–901] (3) | 122 [120–154] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 1914 [1909–1921] (3) | 0 [0–0] (3) | 2933 [2909–2946] (3) | 2957 [2934–2971] (3) |
| `C3x` | 873 [841–907] (3) | 116 [102–128] (3) | 0 [0–0] (3) | 0 [0–1] (3) | 124 [124–135] (3) | 1 [1–1] (3) | 1114 [1068–1171] (3) | 1128 [1084–1187] (3) |
| `C3f` | 847 [843–855] (3) | 111 [108–116] (3) | 227 [227–397] (3) | 0 [0–1] (3) | 878 [873–1967] (3) | 1 [1–1] (3) | 2059 [2057–3337] (3) | 2065 [2062–3342] (3) |
| `G4cp` | 917 [897–924] (3) | 128 [123–133] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 840 [839–873] (3) | 0 [0–0] (3) | 1880 [1864–1931] (3) | 1899 [1883–1951] (3) |
| `G4cpr` | 0 [0–0] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 895 [846–908] (3) | 0 [0–0] (3) | 896 [847–909] (3) | 900 [851–913] (3) |
| `G4o` | 877 [850–916] (3) | 109 [106–117] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 9542 [9391–9636] (3) | 1 [1–1] (3) | 10499 [10379–10671] (3) | 10512 [10388–10685] (3) |
| `G4ob` | 895 [860–914] (3) | 121 [115–141] (3) | 0 [0–0] (3) | 0 [0–0] (3) | 9670 [9417–10867] (3) | 1 [1–1] (3) | 10706 [10393–11905] (3) | 10720 [10406–11918] (3) |
| `G4best` | 0 [0–0] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 771 [667–819] (8) | 0 [0–1] (8) | 771 [668–820] (8) | 776 [673–825] (8) |
| `C3best` | 0 [0–0] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 0 [0–0] (8) | 136 [131–148] (8) | 1 [1–1] (8) | 137 [132–149] (8) | 142 [136–154] (8) |

- `G4`'s 10.69 s is the compile (9.69 s); loading and quantizing take 0.84 s and building 0.10 s. In
  the first window's `G4` run the compile logged `Initializing WebGPU-based API from graph.` once per
  signature, 0.3 s, 4.5 s and 8.7 s into it, and ended 0.8 s after the third; with shared constants
  (`G4c`) the same three lines came within 1.5 s.
- Shared constants (`G4c`) bring the compile to 1.91 s; the program cache (`G4cp`) to 0.84 s, and
  every compile then logs `Initializing WebGPU-based API from serialized data.`; the reused model file
  (`G4cpr`, `G4best`) drops the loading and building, and the executor's init is the compile alone
  (0.90 and 0.77 s).
- On the CPU the XNNPACK weight cache brings the compile from 0.86 to 0.12 s, and with the reused model
  file the init is 0.14 s.

## The published file's graph

Read from the TFLite section of the published `.litertlm` (two signatures; the other 223 subgraphs
are composite decompositions). Its decode signature has 1306 operators: 197 FULLY_CONNECTED with
int8 weights, 113 `odml.rms_norm` composites (every RMSNorm), 56 BATCH_MATMUL, 56
DYNAMIC_UPDATE_SLICE, 28 SOFTMAX, one EMBEDDING_LOOKUP with an int8 table, and no attention
composite. Its KV caches hold 4096 entries (keys `[1, 8, 4096, 128]`, values stored transposed,
`[1, 8, 128, 4096]`, fp32), and its only prefill signature takes 128 tokens. This repo's model has
5889 operators in its three signatures with the raw forms (6222 with the norm composite, 6299 with
the attention composite), all delegated to WebGPU in three partitions.

## Not measured

- Where the GPU time of a decode step goes inside the graph (per operator), and why LiteRT-LM's
  graph runs a step faster on the same accelerator; the graph forms tried here (composites) did not
  close the gap.
- One set of KV caches shared by all signatures; attention over the filled part of the cache only;
  sampling on the GPU; fp16 that passes the bar.
- `C5` (not timed, above); Metal; other prefill, decode and cache lengths; phones.
