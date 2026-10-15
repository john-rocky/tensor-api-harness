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

<!-- fill before push: models/<name>/ summary — which rows were tried, the output-match result, the fastest form, the phone result, the iteration count -->

## Environment

<!-- fill before push: LiteRT commit, Bazel version, Xcode / NDK versions, Python versions, the Mac and the phone -->

## License

Apache 2.0. The graph code follows the LiteRT examples' layout so it can be moved upstream.
