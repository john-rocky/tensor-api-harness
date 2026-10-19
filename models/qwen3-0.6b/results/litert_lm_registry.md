# The graph as a LiteRT-LM executor backend: registration and output match

The prefill and decode graphs of `graph/` run inside LiteRT-LM as an executor registered with
LiteRT-LM's `ExecutorBackendRegistry` (code and build: `litert_lm/README.md`). One binary runs
this executor (`--backend=tensor_api_cpu`, `--backend=tensor_api_gpu`) and LiteRT-LM's own
executors (`--backend=cpu`, `--backend=gpu`) on the published `.litertlm` file.

## Versions and files

- LiteRT-LM: `google-ai-edge/LiteRT-LM` main at `d78488c4f21ada15b71a7d8f5e49f2d98b03ec33`
  (2026-10-18), no file of the checkout changed; `litert_lm/overlay.sh` links this repo's files in
  as `tensor_api/qwen3-0.6b`.
- LiteRT: `9b8c7f77e7a29e6073e614a05e1e38be7b761229`, the commit LiteRT-LM builds with
  (`LITERT_REF` in its `WORKSPACE`). The graph code builds against it unchanged (the earlier
  iterations used `bea3bc69`).
- Binary: `bazel build //tensor_api/qwen3-0.6b:litert_lm_advanced_main_tensor_api` (Bazel 7.6.1, the
  checkout's `.bazelversion`; the checkout's `.bazelrc` sets `-c opt`), 23,856,848 bytes, sha256
  `0195ed42ec32f072f7a1903ee88f58d00027827260946bf61318278a44861860`. Every run below used this
  one file.
- Published model: `litert-community/Qwen3-0.6B` at revision
  `e6687177e00897731e7716267eebd4eba0dde7c2`, `Qwen3-0.6B.litertlm`, 613,722,264 bytes, sha256
  `c5f39b4191926786c3e6668b62af4d6eb15248dfc7a5348252a9b406f33f2108`. Its LLM metadata: stop token
  `<|im_end|>`, no start token, sampler TOP_P with k 20, p 0.95, temperature 0.6, `max_num_tokens`
  4096; its TFLite section carries no backend constraint.
- Weights of the Tensor API executor: `Qwen/Qwen3-0.6B`, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`, `model.safetensors` (as in `ENVIRONMENT.md`).
- GPU: on this macOS build both GPU paths load LiteRT's WebGPU accelerator,
  `libLiteRtWebGpuAccelerator.dylib` from the checkout's `prebuilt/macos_arm64` (sha256
  `b7b7dd7bd6c90a41f6dee7d1f5b2ff76de2df39921f68f1c7e6d525f5136c4cf`, with `libwebgpu_dawn.dylib`
  sha256 `5541ab53c035d9c054b9cdacf2c6e5a0191618732c820a2f08fb0ea4aa9e97bc`). The earlier
  iterations used LiteRT's Metal accelerator.

## How the registration works

- `litert_lm/tensor_api_backend.cc` registers the names `tensor_api_cpu` and `tensor_api_gpu`
  with `LITERT_LM_REGISTER_EXECUTOR_BACKEND`. `--backend=<name>` resolves to the registered
  backend, the engine sends registered backends to its advanced implementation, and
  `EngineAdvancedImpl::Create` calls the registered creator instead of its own executor.
- The `.litertlm` file is loaded by LiteRT-LM as usual. The tokenizer, the prompt handling, the
  stop tokens and the sampler stay in LiteRT-LM; the creator gets the executor settings and these
  model resources, and the executor returns logits (`BackendTraits` defaults: LiteRT-LM's CPU
  sampler).
- The weights path is a flag of the registration library (`--tensor_api_weights`), parsed by the
  main's `absl::ParseCommandLine` with LiteRT-LM's own flags.
- One name per accelerator: LiteRT-LM's `Backend` value names the hardware and the executor at
  once (a note in `runtime/executor/executor_settings_base.h`).
- The creator receives no `litert::Environment`; the executor creates its own.
- Log lines of one run (`--backend=tensor_api_cpu`, prompt p0), in order:
  - `Choose backend: tensor_api_cpu` and `executor_settings: backend: TENSOR_API_CPU` (LiteRT-LM);
  - `tensor_api: ExecutorBackendRegistry creator called for TENSOR_API_CPU` (the creator);
  - `tensor_api prefill: 5 ids [785 6722 315 9625 374], runs 4 from position 0, pending 374`
    (the executor).

## Output match

Greedy decode of 16 tokens for the three prompts of `reference/prompts.json`, compared with
`reference/ref_ids.json` and counted as in `quantization.md`: the generated ids equal to the
reference before each prompt's first mismatch (3 × 16 = 48). The bar for the Tensor API executor
at fp32, written before the runs: at least 44 of 48. LiteRT-LM's own executors on the published
file are recorded without a bar.

Common flags: `--use_session --async=false --max_output_tokens=16 --max_num_tokens=2048` (KV cache
2048), one process per prompt. `--use_session` sends the prompt text without the chat template, and
the file has no start token, so the ids that enter prefill are the tokenizer's ids for the text.
The model path is a link named `Qwen3-0.6B.greedy.litertlm` to the published file: through
`litert_lm/greedy_litertlm_resources.cc`, LiteRT-LM loads the file with its own `.litertlm`
resources class but with greedy sampler parameters (TOP_P with k 1, what LiteRT-LM uses for CPU and
GPU when a file has none; the file's own parameters sample at temperature 0.6), and with a tokenizer
that logs the ids it encodes and decodes. The ids below come from those log lines; for the Tensor
API executor they also come from the executor's own log, and the two agree.

| form | `--backend` | other flags | prompt ids = `prompts.json` | ids equal /48 (p0, p1, p2) | bar ≥ 44 | text match |
|---|---|---|---|---|---|---|
| Tensor API, CPU fp32 | `tensor_api_cpu` | — | 3/3 | 48 (16, 16, 16) | passes | `text_match_litert_lm_tapi_cpu.txt` |
| Tensor API, GPU fp32 | `tensor_api_gpu` | — | 3/3 | 48 (16, 16, 16) | passes | `text_match_litert_lm_tapi_gpu.txt` |
| Tensor API, CPU int8 + lm_head int8 | `tensor_api_cpu` | `--tensor_api_weights_quant=int8 --tensor_api_lm_head_int8` | 3/3 | 48 (16, 16, 16) | no bar | `text_match_litert_lm_tapi_cpu_int8.txt` |
| Tensor API, GPU int8 + lm_head int8 | `tensor_api_gpu` | same | 3/3 | 48 (16, 16, 16) | no bar | `text_match_litert_lm_tapi_gpu_int8.txt` |
| Tensor API, CPU fp32, LiteRT-LM's XNNPACK flags | `tensor_api_cpu` | `--tensor_api_litert_lm_xnnpack_flags` | 3/3 | 48 (16, 16, 16) | no bar | `text_match_litert_lm_tapi_cpu_xf.txt` |
| Tensor API, CPU int8 + lm_head int8, LiteRT-LM's XNNPACK flags | `tensor_api_cpu` | int8 flags and `--tensor_api_litert_lm_xnnpack_flags` | 3/3 | 48 (16, 16, 16) | no bar | `text_match_litert_lm_tapi_cpu_int8_xf.txt` |
| LiteRT-LM CPU, published file | `cpu` | — | 3/3 | 31 (5, 16, 10) | no bar | `text_match_litert_lm_lm_cpu.txt` |
| LiteRT-LM GPU, published file | `gpu` | — | 3/3 | 48 (16, 16, 16) | no bar | `text_match_litert_lm_lm_gpu.txt` |

The ids of every form are in `litert_lm_ids_<form>.json` ({"prompts": [{"id", "ids"}]}, the prompt
ids followed by the 16 generated ids).

Notes:

- The published file is not an fp32 model: it is 613,722,264 bytes, against 2,384,789,652 bytes for
  the fp32 graph of this repo with both signatures (`quantization.md`). How its weights are stored
  was not inspected. Its CPU executor leaves the reference at the 6th generated token of p0
  (". The capital of France is also ...") and at the 11th of p2.
- The Tensor API executor gives the same 48 of 48 as the harness's own host loop at CPU fp32 and
  Metal fp32 (`text_match.txt`, `text_match_metal_fp32.txt`) and at CPU and Metal int8 with the int8
  lm_head (`quantization.md`).
- An earlier build of the binary (without the GPU prefill wait of `litert_lm/README.md` and
  without `--tensor_api_litert_lm_xnnpack_flags`) ran the six other forms with the same flags and
  the same cache directory. The ids of its four Tensor API forms and of LiteRT-LM's GPU executor
  are byte-identical to the ones above. LiteRT-LM's CPU executor gave the same ids for p0 and p1,
  and for p2 the same ids up to the 13th generated token, then different ones (`572` there, `389`
  here); both leave the reference at p2's 11th generated token, so the count is 31 either way. The
  cause was not investigated.
- With `--tensor_api_litert_lm_xnnpack_flags` the CPU delegation was the same (5718 of 5889 nodes,
  85 partitions) and so were the ids.

## Benchmark entry

`--benchmark --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048`
with the plain published file (no `.greedy` link) ran to the end for `--backend=cpu`, `gpu`,
`tensor_api_cpu` and `tensor_api_gpu` (fp32): each printed its BenchmarkInfo with
`Prefill Turn 1: Processed 1024 tokens` and `Decode Turn 1: Processed 256 tokens`. These runs were
not a timing plan, and no time is reported here.

## Delegation

Reported by the Tensor API executor (`--tensor_api_delegation_report`) for its model with three
signatures (prefill 128, prefill 1024, decode):

| accelerator | total nodes | delegated nodes | partitions |
|---|---|---|---|
| CPU (XNNPACK) | 5889 | 5718 | 85 |
| GPU (WebGPU) | 5889 | 5889 | 3 |

## The previous runner

The graph files gained one switch, off by default (`omit_logits`, used by the executor's prefill
signatures). With its default flags, the previous runner (`graph/qwen3_main`, LiteRT `bea3bc69`)
built from the changed files writes the same model as the binary built before the change
(`--serialize_only --prefill_len=32`: 2,384,789,652 bytes, identical), and its CPU fp32 greedy run
gives `TEXT MATCH PASS 3/3`.
