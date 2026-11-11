# What each file does

The skill table is [`skill_table.md`](../skill_table.md). An iteration is one run of the coding agent on one instruction file.

A harness for bringing a model up on the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor)
with a coding agent. The agent works through a fixed skill table: one row per option to try or
check to pass. It marks each row `tried`, `not applicable`, `blocked` or `not yet`, with the
evidence path next to it, and a script refuses "done" while any row is still `not yet`. The
expertise sits in the table, not in the prompt, so the next model starts from the same table.

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
- `tools/repro_sdpa_transposed_metal/` — a model with only the `odml.sdpa_transposed` composite,
  run on the CPU and on the GPU accelerator from one `.tflite` file and compared with a float64
  computation. `run.sh` builds it in a LiteRT checkout and prints one PASS / FAIL line per case.
- `models/<name>/` — the record of one model going through the table: the filled table, the
  iteration records, the graph code, the reference-dump script, the results. The first one is
  [Qwen3-0.6B](qwen3-0.6b.md).

## How to run it on one model

The steps are in [the README](../README.md#run-it-on-one-model). Weights and generated graphs are never committed. `tools/done_check.py` prints the rows still `not yet` with exit 1, and the `blocked` rows with exit 3.
