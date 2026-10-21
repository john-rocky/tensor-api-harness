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
| `--tensor_api_delegation_report` | `false` | compile once more and log the delegation metrics and the buffer type the compiled model gives the `attention_mask`, `param_tensor` and `key_cache_0` inputs of each signature |
| `--tensor_api_attention` | `raw` | `sdpa`: the attention core as the `odml.scaled_dot_product_attention` composite (the graph option of iteration 3); `sdpa_transposed`: as the `odml.sdpa_transposed` composite, with the value caches stored transposed; `sdpa_transposed_param`: the same with the `param_tensor` input and a bool mask; `raw_vt`: the raw ops with the value caches stored transposed (iteration 11) |
| `--tensor_api_norms` | `raw` | `composite`: every RMSNorm as the `odml.rms_norm` composite (iteration 3) |
| `--tensor_api_gpu_share_constants` | `false` | GPU: share the constant tensors (weights) between the signatures |
| `--tensor_api_gpu_program_cache` | `false` | GPU: serialize the compiled programs (and, with shared constants, the converted weights) to `--cache_dir` |
| `--tensor_api_gpu_external_caches` | `off` | GPU: the KV caches as external tensor patterns, with the external tensors mode off (`stock`, as LiteRT-LM's GPU executor) or on (`runner`, as `LitertDynamicRunner`'s feedback loops) |
| `--tensor_api_gpu_litert_lm_options` | `false` | GPU: the other compile options of LiteRT-LM's GPU executor (see below) |
| `--tensor_api_gpu_command_buffer_steps` | `-1` | GPU: steps of command buffer preparation in advance (`-1` leaves the option unset) |
| `--tensor_api_xnnpack_weight_cache` | `false` | CPU: an XNNPACK weight cache file in `--cache_dir` |
| `--tensor_api_model_file` | `off` | `fresh`: write the serialized model to `--cache_dir` and compile the mapped file (the graph, its weight tensors and the in-memory model are released before the compile); `reuse`: as `fresh`, and a file of the same key is compiled without loading the checkpoint or building the graph |
| `--tensor_api_report` | `false` | log the init phases, the memory footprint at each stage and the time of each part of a decode step |
| `--tensor_api_report_hold_s` | `0` | with the report: sleep this many seconds after the decode steps are logged (to inspect the process from outside) |

The KV cache length is LiteRT-LM's `max_num_tokens` (`--max_num_tokens`, or the file's value).

`--tensor_api_gpu_litert_lm_options` sets the options LiteRT-LM's GPU executor passes in
`runtime/executor/llm_executor_settings_utils.cc` and `litert_compiled_model_executor_utils.cc`
beyond precision, shared constants, caches and the KV cache patterns: the WebGPU backend, infinite
float capping, buffer weights and Metal argument buffers on Apple, madvise of the original shared
tensors, weight conversion on the GPU, the fully-delegated hint, source-quantized FC / convolution
ops, shader optimization, 2 upload threads and 1 compile thread, and, when LiteRT-LM runs its
benchmark, the active wait for synchronous execution and the wait for the weight conversion.

The cache files (`--tensor_api_model_file`, `--tensor_api_xnnpack_weight_cache`,
`--tensor_api_gpu_program_cache`) carry a key made of the checkpoint's size and modification time,
the weight quantization, the lm_head copy, the graph forms, the prefill lengths and the KV cache
length (and, for the compiled caches, the accelerator options).

With `--tensor_api_report` the executor logs lines starting with `tensor_api report:`: the time of
each init phase (checkpoint loading and quantization, graph building, the model file, the
environment and options, the compile, the rest), the process's physical footprint, its peak so far
and its graphics-tagged part (macOS `task_info`) after each stage, and for every decode turn the
median, minimum and maximum over its steps of: writing the step inputs and binding the caches,
`Run`, taking the cache outputs, reading the logits back (`GetOutput`, `Lock`, copy), copying
them to the returned buffer, the whole call, and the time from the previous call's return
(LiteRT-LM's sampler and session).

## Limits

- One sequence (one output candidate), text input only.
- A saved context can be restored only while its tokens are the ones in the caches; the caches
  are not part of the context.
- Prefill logits are not computed: LiteRT-LM takes the first generated token from the decode call
  on the last prompt token, as with its own executors.
