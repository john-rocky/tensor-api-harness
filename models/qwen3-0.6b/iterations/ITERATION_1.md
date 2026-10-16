# Iteration 1 — Qwen3-0.6B: the graph, the weights and output match on the Mac CPU

## Read first
- `skill_table.md` at the repo root, then `models/qwen3-0.6b/skill_table.md`.
- No earlier iteration for this model.
- LiteRT's `tensor/examples/gemma3` (embedding table, tied lm_head, the safetensors loader and its
  Gemma-only +1.0 on norm weights) and the Qwen3-TTS talker graph in litert-samples (a Qwen3-shaped
  decoder: q_norm / k_norm, rotate-half RoPE, grouped-query attention, the KV cache written with
  `DynamicUpdateSlice` at a runtime position).

## Goal (one)
Rows 1–6 and 15 on the Mac CPU (fp32, XNNPACK): the op inventory, the PyTorch reference dumps, the
prefill and decode graphs in C++, the checkpoint loader, per-block output match, greedy text match
and the environment record. Rows 7–14 stay `not yet`.

## Acceptance
- Reference: transformers, fp32, CPU, three fixed prompts (plain text, no chat template):
  `The capital of France is`, `def fibonacci(n):`,
  `In 1969, humans first landed on the Moon. The mission was called`.
- Row 5: for each prompt, the embedding output, the output of each of the 28 decoder blocks, the
  final norm output and the last position's logits pass `checks/block_match.py`. The bar is written
  into the manifests when the reference is dumped, before any graph run: cosine >= 0.9999 and
  max abs difference <= max(2e-2, 1e-3 × max|ref|) per tensor; logits max(5e-2, 1e-3 × max|ref|).
  All numbers are CPU fp32 against PyTorch fp32 (bf16 weights widened to fp32 on both sides).
- Row 6: greedy decode of 16 tokens per prompt; `checks/text_match.py` passes 3/3 (every id equal,
  prompt ids included).
- Rows 1–4 and 15: their evidence files exist; `tools/done_check.py` lists exactly rows 7–14.

## Stop conditions
- An op in a required block with no Tensor API equivalent and no rewrite (an embedding lookup moved to
  the host counts as a rewrite).
- The first decoder block does not match after three builds aimed at the same cause.
- Eight hours without row 5 passing: fill the table honestly and close.

## Scope
`models/qwen3-0.6b/` (new). Everything else in the repository is read only. Weights, generated
`.tflite` files and dumps are not committed.

## Do not
- Do not write `tried` without an evidence path that exists.
- Do not turn a `not yet` row into `not applicable` or `blocked` to make the done check pass; `blocked` needs the cause after the colon.
- Do not change the thresholds in `checks/` during the iteration; a new bar is a new iteration.
- Do not commit weights, generated graphs or dumps.
- No timing in this iteration: the speed rows come later, so no millisecond number is recorded.

## Progress
- Row 1: `OPS.md` maps every block to Tensor API ops. The two rewrites are RoPE (rotate-half with host
  cos/sin rows) and the token loop, sampling and tokenizer on the host. The embedding lookup stays in
  the graph (`EmbeddingLookup`, which emits the TFLite `EMBEDDING_LOOKUP` builtin,
  `tensor/backends/tflite/arithmetic_tflite.cc`).
- Row 2: `reference/dump_reference.py` checked the weights' sha256, then dumped the embedding output,
  the 28 block outputs, the final norm output and the last-position logits for the three prompts. It
  wrote one manifest per prompt with the bar fixed at dump time. The greedy 16 ids from a hand-written
  argmax loop equal `generate(do_sample=False)` for all three prompts (`generate_same` in
  `reference/ref_ids.json`).
- Row 3: with synthetic weights (seed 42) under the real names, both signatures built, serialized and
  ran on the CPU. Prefill outputs are [1, S, 1024] per block and [1, S, 151936] logits; decode
  outputs are [1, 1, 151936] logits and the KV caches as [1, 8, 64, 128] (the runner prints layer 0)
  (`results/synthetic_run.txt`). Each block output has the reference's [S, 1024] elements with a
  leading batch of 1.
- Row 4: 310 of the checkpoint's 311 tensors are bound, bf16 widened to fp32, with every shape checked
  against the config. `lm_head.weight` is not read because the lm_head reuses the embedding table
  (`results/weights_map.md`).
- Row 5: `BLOCK MATCH PASS 31/31` for p0, p1 and p2 on the first build (`results/block_match_p*.txt`).
  The largest block max abs difference is at block_27 for every prompt: 6.3e-4 (p0), 9.8e-4 (p1),
  1.5e-3 (p2). The embedding output is bit-equal. Every cosine prints as 1.00000000.
- Row 6: `TEXT MATCH PASS 3/3` (`results/text_match.txt`). The prefill and 15 chained decode steps per
  prompt use the production signatures (no block outputs). The decoded texts equal the reference's
  (`results/text_graph.txt`).
- Row 15: `ENVIRONMENT.md`.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 1):

```
models/qwen3-0.6b/skill_table.md: tried 7, not applicable 0, blocked 0, not yet 8
NOT DONE models/qwen3-0.6b/skill_table.md: 8 rows not yet
  #7 State and KV cache
  #8 Attention form
  #9 Which ops go to which accelerator
  #10 GPU precision
  #11 Buffer storage and large tensors
  #12 Quantization
  #13 Speed floor
  #14 Phone run
```

## Corrections
- One manifest per prompt (`reference/manifest_p0.json`, `_p1`, `_p2`), not a single manifest.
  `checks/block_match.py` reads fixed shapes from one manifest, and the prompts have 5, 4 and 18
  tokens.
- In transformers 5.19.0, the last entry of `output_hidden_states` is the final norm output, not the
  last decoder block's output. The reference therefore takes every block output with a forward hook.
  The script asserts the hooks against `output_hidden_states` (note in `reference/manifest_p0.json`).
- The checkpoint is tied but still stores `lm_head.weight` (311 tensors, not 310), with the same bytes
  as `model.embed_tokens.weight` (`results/weights_map.md`).
- The C++ namespace is `qwen3` because the model directory name `qwen3-0.6b` is not a valid
  identifier. The bazel package `//tensor/examples/qwen3-0.6b` accepts it (`graph/BUILD`).

## Dead ends
- None. The first build passed, and every check passed on its first run.
- One compile mistake was caught before building by a `clang++ -fsyntax-only` pass over `graph/`:
  `TensorHandle::GetName()` returns `absl::string_view`, so it needs an explicit `std::string` where
  a `std::string` is taken (`graph/qwen3_main.cc`).

## Op and delegate observations
- LiteRT `bea3bc69df855f80e062ea5a7fb72b5445d2a64c`, CPU only (the XNNPACK delegate is created; the
  partition count was not recorded: unverified). No GPU run in this iteration.
- Embedding: in-graph `EMBEDDING_LOOKUP` with rank-1 int32 ids [S] and the fp32 table
  [151936, 1024]. It is followed by a `Reshape` to [1, S, 1024]. Its output is bit-equal to the
  reference on all three prompts.
- Tied lm_head: the same table tensor feeds the lookup and the lm_head `FullyConnected`. The
  serialized graph is 2,384,789,652 bytes (prefill and decode, no block outputs). That is the size of
  596,049,920 fp32 parameters (2,384,199,680 bytes) plus about 0.6 MB, so the table is stored once. A
  second copy would add 622,329,856 bytes. The variant with block outputs is 2,384,791,316 bytes.
- Block max abs difference over the 28 blocks: median 4.9e-4 (p0), 9.8e-4 (p1), 9.8e-4 (p2); max at
  block_27 as above. Final norm: 6.9e-5, 7.6e-5, 8.8e-5. Last-position logits: 5.1e-5, 6.5e-5,
  3.1e-5. The per-tensor bars from block_02 on are large (about 6.5 at block_02) because some
  residual-stream entries reach several thousand in magnitude (`max_abs_ref` in the manifests).
- The KV caches are written with `DynamicUpdateSlice` at a runtime int32 position and rebound by
  handle between decode steps. Row 6 exercises this path, but row 7's comparison against a stateless
  reference is not done in this iteration.

## Environment
- `ENVIRONMENT.md`: LiteRT commit, Bazel 7.7.0 through bazelisk 1.29.0, Xcode 27.0, macOS 27.0, Apple M4
  Max (128 GB). Reference in Python 3.12.11, torch 2.14.1, transformers 5.19.0 (CPU, fp32, eager
  attention); checks in Python 3.14.6, numpy 2.4.4. Weights sha256 `f47f7117…6874b` checked before
  the reference run.
- No timing was taken. Runs started only once no other job was timing on the machine.

## Files left
- `graph/` (BUILD, `qwen3_config.h`, `qwen3_graph.{h,cc}`, `qwen3_weights.{h,cc}`, `qwen3_main.cc`):
  the graphs, the loader and the runner. Keep.
- `reference/dump_reference.py`, `prompts.json`, `ref_ids.json`, `manifest_p{0,1,2}.json`: the
  reference script and its committed outputs. Keep.
- `results/`: block match outputs, text match output, decoded texts, weights map, synthetic run log.
  Keep.
- `OPS.md`, `ENVIRONMENT.md`, `skill_table.md`, this file. Keep.
- Not committed and deletable: the per-block dumps (reference and graph, 1.1 to 2.7 MB per prompt
  and side) and the three serialized `.tflite` files (about 2.4 GB each), all outside the repository.

## Time
Start 15:35, end 17:13 (JST, 2026-10-15). Added after the iteration from its logs; the start is the
minute the iteration was launched.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 7 (15:35–15:43 the graph files, the reference script and the run script; 17:10–17:13 the results and this record) |
| builds | 1 | under 1 (8.2 s, with the skeleton iteration's cache warm) |
| waiting for the machine (another measurement was running) | 1 | 86 (5,132 s before the single run script) |
| runs and checks | 1 script | 1 (17:09:05–17:09:52: reference dump, build, synthetic run, dumps, block match ×3, greedy, text match) |
| rework after a failure | 0 | 0 (one build; every check passed on its first run) |
| phone in use | 0 | 0 |
