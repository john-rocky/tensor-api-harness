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

[Qwen3-0.6B](models/qwen3-0.6b/) went through all 15 rows in four iterations
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
  blockwise-32 does not reach the bar on the CPU and crashes the Metal delegate at compile.
- Speed (`results/speed.md`, `results/speed_floor.txt`): against the CPU fp32 floor of 18.12 ms per
  decode token, the fastest form that still matches the reference is Metal with int8 weights and an
  int8 lm_head, 4.72 ms per token.
- Phone (`results/phone_s26.md`): on a Galaxy S26 (SM-S942U1), a lab unit reached through Google's
  Developer Device Platform, the CPU fp32 run gives the same ids as the Mac and its chained decode
  matches the stateless reference at every step; decode takes 39.5 ms per token in fp32 and 11.7 with
  int8 weights and an int8 lm_head, with the unit's CPU frequencies capped for the whole job. The
  OpenCL form was not run: the unit had 6.5 GB free and the job's memory rule (six times the file
  plus 1 GB, after a delegate initialization that took that much on another phone) asked for 8.3 GB.
- Hours, from the `Time` sections of the four iteration records (minutes; the harness skeleton
  itself is not counted):

| part | iteration 1 | 2 | 3 | 4 | total |
|---|---|---|---|---|---|
| coding and reading | 7 | 8 | 33 | 30 | 78 |
| builds | 0 | 0 | 0 | 16 | 16 |
| waiting for the machine (another measurement running) | 86 | 0 | 32 | 386 | 504 |
| waiting for a decision or for the phone | 0 | 0 | 0 | 1137 | 1137 |
| runs and checks | 1 | 3 | 5 | 3 | 12 |
| rework after a failure | 0 | 0 | 130 | 17 | 147 |
| phone in use | 0 | 0 | 0 | 2 | 2 |

  Wall-clock from the first iteration's start to the fourth's close: 31 h 45 min. The agent's own
  work in it is about 2 h (coding and reading, builds, runs and checks, and 17 minutes of rework);
  the rest is waiting. The 130 minutes of rework in iteration 3 is an idle wait on a script that
  had stopped at its start; the 1137 minutes in iteration 4 are a wait for the go and for a phone
  that turned out to be disconnected, before the lab phone was used instead. Builds are the
  minutes the build ran: three builds under ten seconds each, then the Android cross-compile.

## Environment

- LiteRT: `google-ai-edge/LiteRT` main at `bea3bc69df855f80e062ea5a7fb72b5445d2a64c` (2026-10-14). The
  Metal accelerator is the `latest` prebuilt the build fetches (its sha256 is in
  `models/qwen3-0.6b/ENVIRONMENT.md`); the Android GPU accelerator is the prebuilt
  `libLiteRtClGlAccelerator.so` from the same fetch.
- Bazel 7.7.0 through bazelisk 1.29.0. Xcode 27.0 (27A266a) on macOS 27.0. Android NDK 28.2.13676358
  for the arm64 cross-compile (`bazel build -c opt --config=android_arm64`).
- Python 3.12.11 with torch 2.14.1 and transformers 5.19.0 for the reference dumps; Python 3.14.6
  with numpy 2.4.4 for the checks.
- Mac: Apple M4 Max, 128 GB. Phone: Galaxy S26 (SM-S942U1), Android 16, a lab unit reached through
  Google's Developer Device Platform: the job pushes the binary and the saved graphs, runs a shell
  script, and the outputs are pulled back; no adb.

## License

Apache 2.0. The graph code follows the LiteRT examples' layout so it can be moved upstream.
