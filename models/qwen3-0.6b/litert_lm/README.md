# Qwen3-0.6B on the Tensor API as a LiteRT-LM executor backend

The prefill and decode graphs of `../graph` (written with the LiteRT Tensor API) run inside
LiteRT-LM as an executor backend, registered through LiteRT-LM's `ExecutorBackendRegistry`. One
binary runs both this backend and LiteRT-LM's own CPU and GPU executors with a published
`.litertlm` file, so the two can be compared with the same command line.

## How it plugs in

- `tensor_api_backend.cc` registers two backend names with
  `LITERT_LM_REGISTER_EXECUTOR_BACKEND`: `tensor_api_cpu` (LiteRT CPU, XNNPACK) and
  `tensor_api_gpu` (the LiteRT GPU accelerator). LiteRT-LM's `Backend` value names the
  hardware and the executor at once, so the accelerator is part of the name.
- The model file stays the published `.litertlm`. LiteRT-LM loads it as usual and keeps the
  tokenizer, the prompt handling, the stop tokens and the sampler; the registered creator gets
  the executor settings and these model resources.
- The executor (`tensor_api_executor.cc`) reads `model.safetensors` when the engine is created,
  builds one prefill signature per prefill length (without the lm_head) and one decode signature
  (with the logits) in memory, and compiles them with LiteRT. It owns its own
  `litert::Environment`; the creator does not receive the engine's.
- Token bookkeeping follows LiteRT-LM's compiled-model executor: `Prefill` runs every given token
  but the last, which stays pending; `DecodeLogits` runs one token and returns the logits. The KV
  caches are signature inputs and outputs of length `max_num_tokens`; each call binds the
  previous call's cache outputs as its inputs without copying them.
- On the GPU, `Run` can return once the work is enqueued. When the session asks a prefill to
  complete (it does for every prefill), the executor locks one cache output before it returns;
  a decode call ends with reading the logits, which waits as well.
- `greedy_litertlm_resources.cc` is for the output match only. A model path ending in
  `.greedy.litertlm` (a link to a `.litertlm` file) is loaded through `ModelResourcesRegistry`
  with LiteRT-LM's own `.litertlm` resources class, with the sampler parameters set to greedy
  (TOP_P with k = 1, which is what LiteRT-LM uses for CPU and GPU when a file has none) and a
  tokenizer that logs the ids it encodes and decodes. Other paths are not affected.

## Build

From a LiteRT-LM checkout (the build was checked at commit
`d78488c4f21ada15b71a7d8f5e49f2d98b03ec33`, which builds with LiteRT at
`9b8c7f77e7a29e6073e614a05e1e38be7b761229`):

```
<this repo>/models/qwen3-0.6b/litert_lm/overlay.sh <LiteRT-LM checkout>
cd <LiteRT-LM checkout>
bazel build //tensor_api/qwen3-0.6b:litert_lm_advanced_main_tensor_api
```

`overlay.sh` links the files of this directory and the graph sources into the checkout as
`tensor_api/qwen3-0.6b`; no file of the checkout changes. The binary is
`//runtime/engine:litert_lm_advanced_main` (same source, link options and dependencies) plus the
two registration libraries.

## Run

The backend is chosen with `--backend`; everything else can stay the same. On macOS the GPU
accelerators are found through `DYLD_LIBRARY_PATH` (LiteRT-LM's `prebuilt/macos_arm64`).

```
BIN=bazel-bin/tensor_api/qwen3-0.6b/litert_lm_advanced_main_tensor_api
DYLD_LIBRARY_PATH=prebuilt/macos_arm64 $BIN --backend=tensor_api_cpu \
  --model_path=<Qwen3-0.6B.litertlm> \
  --tensor_api_weights=<HF cache>/models--Qwen--Qwen3-0.6B/snapshots/<rev>/model.safetensors \
  --input_prompt="The capital of France is" --use_session --async=false \
  --max_output_tokens=16 --max_num_tokens=2048
```

`--backend=cpu` and `--backend=gpu` run LiteRT-LM's own executors on the same `.litertlm`;
`--tensor_api_*` flags are ignored by them.

Flags of this backend:

| flag | default | meaning |
|---|---|---|
| `--tensor_api_weights` | (required) | `model.safetensors` of `Qwen/Qwen3-0.6B` |
| `--tensor_api_prefill_lengths` | `128,1024` | lengths of the prefill signatures; a prefill runs in chunks, each with the shortest signature that holds it, padded on the right |
| `--tensor_api_weights_quant` | `none` | `none`, `int8` (per output channel) or `int4` (blockwise 32) for the decoder matmul weights, quantized while loading |
| `--tensor_api_lm_head_int8` | `false` | an int8 per-channel copy of the embedding table for the lm_head |
| `--tensor_api_gpu_precision` | `fp32` | GPU calculation precision (`fp32` or `fp16`) |
| `--tensor_api_gpu_buffer_storage` | `buffer` | GPU tensor storage (`default`, `buffer` or `texture2d`) |
| `--tensor_api_num_threads` | `0` | CPU threads; 0 takes `number_of_threads` from the executor settings |
| `--tensor_api_litert_lm_xnnpack_flags` | `false` | CPU: the XNNPACK flags of LiteRT-LM's own CPU executor (the default flags plus dynamic fully-connected and the latest operators) instead of LiteRT's defaults |
| `--tensor_api_log_ids` | `true` | log every token id the executor receives |
| `--tensor_api_delegation_report` | `false` | compile once more and log the delegation metrics |

The KV cache length is LiteRT-LM's `max_num_tokens` (`--max_num_tokens`, or the file's value).

## Limits

- One sequence (one output candidate), text input only.
- A saved context can be restored only while its tokens are the ones in the caches; the caches
  are not part of the context.
- Prefill logits are not computed: LiteRT-LM takes the first generated token from the decode call
  on the last prompt token, as with its own executors.
