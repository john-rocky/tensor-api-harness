# Inside LiteRT-LM

The skill table is [`skill_table.md`](../skill_table.md). An iteration is one run of the coding agent on one instruction file.

LiteRT-LM main has an executor registry (`runtime/executor/executor_backend_registry.h`). Iteration 7
registers this repo's Qwen3-0.6B graph through it
([`models/qwen3-0.6b/litert_lm/`](../models/qwen3-0.6b/litert_lm/)): a library that adds two backends,
`tensor_api_cpu` and `tensor_api_gpu`, and builds the graph from the checkpoint when the engine is
created. The tokenizer, prompt handling, stop tokens and sampling stay with LiteRT-LM, which reads
them from the published `Qwen3-0.6B.litertlm`. One binary runs the graph with
`--backend=tensor_api_cpu` or `tensor_api_gpu` and the published `.litertlm` with `--backend=cpu` or
`gpu` (`litert_lm_advanced_main`'s sources and dependencies with that library and the
greedy-sampler resources of the output match linked in, built from LiteRT-LM main without changing
any of its files).

- Output match (`results/litert_lm_registry.md`): through LiteRT-LM, greedy decode of the graph
  reproduces the PyTorch reference ids, 48 of 48, on the CPU and on the GPU, with fp32 and with int8
  weights. The published `.litertlm` through the same binary gives 48 of 48 on the GPU and 31 of 48
  on the CPU.
- Speed and memory (`results/litert_lm_speed.md`, `results/litert_lm_levers.md`): Apple M4 Max,
  LiteRT-LM's benchmark entry with forced prefill 1024, decode 256 and context 2048, int8 matmul
  weights on both sides (the 197 FULLY_CONNECTED ops of the published file's decode signature carry
  int8 weights, read from its TFLite section), the GPU through the WebGPU accelerator that LiteRT-LM's macOS build loads, medians
  of one quiet window, MB = 10^6 bytes:

| | published `.litertlm` | Tensor API graph |
|---|---|---|
| CPU decode, tok/s | 31.48 | 46.25 |
| CPU prefill, tok/s | 769.43 | 1226.38 |
| CPU peak memory footprint, MB | 1195 | 1966 |
| GPU decode, tok/s (fp32 activations) | 102.07 | 77.75 |
| GPU prefill, tok/s (fp32 activations) | 2226.41 | 3804.39 |
| GPU peak memory footprint, MB | 2421 | 8083 |

  The Tensor API columns are the backend's forms with its cache, memory and GPU options on (`C3best`: the
  XNNPACK weight cache and a reused serialized model file; `G4best`: constants shared between the
  signatures, the GPU program cache, a reused serialized model file and LiteRT-LM's GPU options; the
  flags are in `results/litert_lm_levers.md`); without them the process peaked at 4766 MB on the CPU
  and 18190 MB on the GPU and took 1.84 s and 10.69 s to start. At its default fp16 activations
  LiteRT-LM's GPU executor decodes at 141.52 tok/s in 1436 MB; this graph does not reproduce the
  model at fp16 (0 of 48 ids), on WebGPU as on Metal. A GPU decode step of the graph is about 12.4 ms
  of GPU work and 0.3 ms of logits readback (LiteRT-LM's step on the published file: 9.8 ms at fp32).
  The published file's decode graph keeps a 4096-entry cache (V stored transposed), twice this graph's
  2048, and decodes faster on the GPU; its only prefill signature takes 128 tokens, where this
  graph's executor also has a 1024-token one. The published graph runs every RMSNorm as the
  `odml.rms_norm` composite.
  Where the remaining GPU decode difference comes from was not established.
- Two things the registry leaves to the backend: the creator gets no `litert::Environment`, so the
  backend makes its own, and a backend name stands for both the executor and the hardware, so the
  graph registers one name per accelerator.
- The attention core as the `odml.sdpa_transposed` composite (iteration 11,
  `results/attention_forms.md`), at the composite boundary LiteRT-Torch exports for it: the graph matches the
  reference on the CPU and, through LiteRT-LM, on WebGPU, where it decodes at 87.82 tok/s against
  79.29 for the raw graph. A control with the raw ops and the same transposed value caches decodes
  at 90.99 against 80.77 in a second window, so that gain is the cache layout. On Metal the
  composite gives wrong ids from the first generated token, and so does the form with LiteRT-Torch's
  param tensor on both GPUs; the raw ops with the transposed caches match on Metal (8.057 against
  8.113 ms/token for the raw graph). A model with only the composite gives the Metal failure
  whenever the query heads are grouped over two or more key/value heads
  (`tools/repro_sdpa_transposed_metal/`). The param-tensor form sets `from_cache_update` while this
  graph writes the caches with `DYNAMIC_UPDATE_SLICE`, not `odml.cache_update`.
- Hours, from the `Time` sections of iterations 7 to 11 (minutes; 7 and 8 ran at the same time):

| part | iteration 7 | 8 | 9 | 10 | 11 | total |
|---|---|---|---|---|---|---|
| coding and reading | 175 | 91 | 36 | 69 | 100 | 471 |
| builds | 4 | 3 | 0 | 0 | 4 | 11 |
| waiting for the machine (the measurement lock shared with other jobs) | 321 | 50 | 0 | 286 | 542 | 1199 |
| waiting for a decision | 0 | 74 | 0 | 0 | 0 | 74 |
| runs and checks | 7 | 6 | 18 | 34 | 37 | 102 |
| rework after a failure | 5 | 1 | 3 | 0 | 5 | 14 |

  Iteration 10 also has 309 minutes in which the agent's session was not running; its second window
  waited and ran in that time through a script.
