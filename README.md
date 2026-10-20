# Tensor API bring-up harness

A harness for bringing a model up on the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
with a coding agent. The agent works through a fixed skill table: one row per option to try or
check to pass. It marks each row `tried`, `not applicable`, `blocked` or `not yet`, with the
evidence path next to it, and a script refuses "done" while any row is still `not yet`. The
expertise sits in the table, not in the prompt, so the next model starts from the same table.

## What is in it

- `skill_table.md` — the skill table for the Tensor API: architecture and op inventory, the
  PyTorch reference forward, the graph skeleton, weight loading, per-block output match against
  PyTorch, output text match, state and KV cache, attention form, which ops go to which
  accelerator, GPU precision, buffer storage, quantization, speed floor, the phone run, and the
  environment record. Each row says how to do it on the Tensor API and what counts as a pass.
- `checks/` — pass/fail checks with the bar written down before the run: `block_match.py`,
  `text_match.py`, `speed_floor.py`. They read dumps and JSON files, so they do not depend on the
  model's binary. `checks/fixtures/` holds the small files `tools/selftest.sh` runs them on,
  failing paths included.
- `tools/done_check.py` — reads a filled skill table. Exit 0 when every row is `tried` or
  `not applicable`. Exit 1 with the list of rows still `not yet`. Exit 3 when only `blocked` rows
  remain, for a person to decide. A `tried` row needs an evidence file that exists; a `blocked`
  row needs its cause.
- `tools/iteration_close_check.py` — refuses an iteration record that is missing one of the five
  close sections (corrections, dead ends, op and delegate observations, environment, files left).
  What one iteration learned reaches the next one through the file.
- `iterations/` — the instruction template for one iteration (one goal, acceptance numbers, stop
  conditions, scope) and the close form.
- `models/<name>/` — the record of one model going through the table: the filled table, the
  iteration records, the graph code, the reference-dump script, the results. The first one is
  listed below.

## How to run it on one model

1. Prerequisites: a LiteRT checkout, Bazel through bazelisk, Python 3 with numpy (torch and
   transformers for the reference dumps), and the model's checkpoint from Hugging Face. Weights
   and generated graphs are never committed.
2. `cp skill_table.md models/<name>/skill_table.md`, then write
   `models/<name>/iterations/ITERATION_1.md` from `iterations/ITERATION_TEMPLATE.md`: the rows
   this iteration takes, the output-match bar, the device.
3. Start the coding agent on that iteration file. We run it with Claude Code; the files are plain
   text, so any agent that reads files and runs commands can take them.
4. `python3 tools/done_check.py models/<name>/skill_table.md`. Exit 1 prints the rows still
   `not yet`; the next iteration starts from them. Exit 3 prints the `blocked` rows.
5. Repeat 3 and 4 until exit 0.

## The graph code

A model's graphs are authored in C++ as a LiteRT `tensor/examples/` directory.
`models/<name>/graph/` is such a directory; `tools/overlay.sh <LiteRT checkout> <name>` links it
into the checkout and prints the Bazel build line and the run line. Reference implementations in
the same form:

- LiteRT's own `tensor/examples/gemma3`: prefill and decode signatures, the KV cache as signature
  input and output, safetensors loading, the Metal accelerator prebuilt on macOS.
- Three merged litert-samples implementations under `models/<family>/<model>/tensor_api/`:
  Qwen3-TTS ([#241](https://github.com/google-ai-edge/litert-samples/pull/241)), LLaDA-8B
  ([#242](https://github.com/google-ai-edge/litert-samples/pull/242)) and SAM 2.1 video
  ([#301](https://github.com/google-ai-edge/litert-samples/pull/301)), each with its verification
  suite against an independent reference.

## First model

[Qwen3-0.6B](models/qwen3-0.6b/) went through all 15 rows in six iterations
(`models/qwen3-0.6b/iterations/`), on an Apple M4 Max and on a Galaxy S26. The graphs are the C++ in
`models/qwen3-0.6b/graph/`: prefill and decode signatures, the embedding lookup in the graph, the KV
cache written at a runtime position, the tied lm_head over the embedding table.

- Output match (`results/block_match_p*.txt`, `results/text_match.txt`, `results/kv_chain*.txt`): on
  the Mac CPU in fp32, the embedding output, all 28 block outputs and the last-position logits match
  the PyTorch reference on three prompts (largest block difference 1.5e-3), greedy decode reproduces
  the reference ids for 16 tokens on all three prompts, and 15 chained decode steps through the KV
  cache match a stateless reference at every step, on the CPU and on Metal.
- Accelerators (`results/delegation_metal.md`): on Metal every one of the graph's 3994 nodes is
  delegated, in one partition per signature, the embedding lookup and the cache writes included.
  Metal fp32 passes the per-block bar except for one exported tensor: the final norm reads back wrong
  while the logits computed from it are right (cause not established; recorded in iterations 2 and 3).
  Metal fp16 does not reproduce this model's text: the residual stream reaches about 6,500 and the
  per-block difference grows to `nan` by the last block.
- Forms (`results/attention_forms.md`, `results/quantization.md`): the `odml.rms_norm` composite is
  byte-identical to the raw norm on the CPU and 2.5% faster on Metal. The
  `odml.scaled_dot_product_attention` composite runs fully delegated on Metal but slower than the raw
  ops at these shapes, and fails to prepare on the CPU. int8 per-channel weights reproduce the
  reference text on Metal (48 of 48 ids) and on the CPU once the lm_head is int8 too; int4
  blockwise-32 does not reach the bar on the CPU and crashes the Metal delegate at compile. With the
  fp32 lm_head the CPU matches 41 of 48 ids; the difference from Metal comes from XNNPACK: it
  quantizes the input of every FULLY_CONNECTED with int8 weights to int8, per row, before the matmul.
  The same rounded weights stored as fp32 give 48 of 48 on the CPU too, at the fp32 speed (iteration
  8, `results/xnnpack_int8_fc.md`).
- Speed (`results/speed.md`, `results/speed_floor.txt`): against the CPU fp32 floor of 18.12 ms per
  decode token, the fastest form that still matches the reference is Metal with int8 weights and an
  int8 lm_head, 4.72 ms per token.
- Phone (`results/phone_s26.md`): on a Galaxy S26 the CPU fp32 run gives the same ids as the Mac and
  its chained decode matches the stateless reference at every step. On a lab unit (SM-S942U1, reached
  through Google's Developer Device Platform) decode takes 39.5 ms per token in fp32 and 11.7 with int8
  weights and an int8 lm_head, with the unit's CPU frequencies capped for the whole job; its free
  memory did not allow the OpenCL form under the job's memory rule. On a unit attached over USB
  (SM-S942Q) the OpenCL form with int8 weights runs fully delegated (3994 of 3994 nodes) and
  reproduces the reference ids (48 of 48); its decode takes 26.7 ms per token, against 38.1 for CPU
  fp32 and 13.0 for CPU int8 with an int8 lm_head on the same unit. On this OpenCL path `Run` returns
  before the GPU has finished (measured that way, the same form shows 3.3 ms per token), so the phone
  times are taken through the readback of the logits; on the Mac the two definitions differ by 0.5%
  or less.
- Hours, from the `Time` sections of the six iteration records (minutes; the harness skeleton
  itself is not counted):

| part | iteration 1 | 2 | 3 | 4 | 5 | 6 | total |
|---|---|---|---|---|---|---|---|
| coding and reading | 7 | 8 | 33 | 30 | 8 | 7 | 93 |
| builds | 0 | 0 | 0 | 16 | 0 | 0 | 16 |
| waiting for the machine (another measurement running) | 86 | 0 | 32 | 386 | 0 | 0 | 504 |
| waiting for a decision or for the phone | 0 | 0 | 0 | 1137 | 11 | 0 | 1148 |
| runs and checks | 1 | 3 | 5 | 3 | 3 | 4 | 19 |
| rework after a failure | 0 | 0 | 130 | 17 | 2 | 0 | 149 |
| phone in use | 0 | 0 | 0 | 2 | 6 | 4 | 12 |

  The first four iterations ran from one start to the fourth's close in 31 h 45 min; iterations 5
  and 6 (the OpenCL rows on the attached phone, and the timing definition) took 23 and 12 minutes
  the next night. The agent's own work in all six is about 2 h 30 min (coding and reading, builds,
  runs and checks, and 19 minutes of rework); the rest is waiting. The 130 minutes of rework in
  iteration 3 is an idle wait on a script that had stopped at its start; the 1137 minutes in
  iteration 4 are a wait for the go and for a phone that turned out to be disconnected, before the
  lab phone was used instead. Builds are the minutes the build ran: the Android cross-compile and
  five builds under twelve seconds each.

## Inside LiteRT-LM

LiteRT-LM main has an executor registry (`runtime/executor/executor_backend_registry.h`). Iteration 7
registers this repo's Qwen3-0.6B graph through it
([`models/qwen3-0.6b/litert_lm/`](models/qwen3-0.6b/litert_lm/)): a library that adds two backends,
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
- Hours, from the `Time` sections of iterations 7 to 10 (minutes; 7 and 8 ran at the same time):

| part | iteration 7 | 8 | 9 | 10 | total |
|---|---|---|---|---|---|
| coding and reading | 175 | 91 | 36 | 69 | 371 |
| builds | 4 | 3 | 0 | 0 | 7 |
| waiting for the machine (the measurement lock shared with other jobs) | 321 | 50 | 0 | 286 | 657 |
| waiting for a decision | 0 | 74 | 0 | 0 | 74 |
| runs and checks | 7 | 6 | 18 | 34 | 65 |
| rework after a failure | 5 | 1 | 3 | 0 | 9 |

  Iteration 10 also has 309 minutes in which the agent's session was not running; its second window
  waited and ran in that time through a script.

## Environment

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

## License

Apache 2.0. The graph code follows the LiteRT examples' layout so it can be moved upstream.
