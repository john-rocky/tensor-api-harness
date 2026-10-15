# KV cache chain (row 7)

## What is compared

For each prompt, the graph runs prefill (the prompt, S tokens) and then 15 decode steps, one token
each, feeding back its own greedy pick. It dumps 16 logits rows (`--dump_chain`): `chain_00` is the
prefill output at the last prompt position, `chain_01` .. `chain_15` are the decode steps.

The reference is stateless: one PyTorch fp32 forward (eager attention, `use_cache=False`) over the
prompt and its 16 reference ids, with no cache. Row i of the reference is the logits at position
S - 1 + i, and its argmax is generated token i + 1 (asserted when the reference is dumped). So each
decode step's logits, computed from the cache the earlier calls wrote, are compared with logits
computed from scratch at the same position.

Bar, fixed in `reference/manifest_chain_p*.json` when the reference was dumped: cosine >= 0.9999
and max abs difference <= max(5e-2, 1e-3 × max|ref|), per row.

## Results, CPU fp32

| prompt | S | rows | result | largest max abs difference | median | argmax equal |
|---|---|---|---|---|---|---|
| p0 | 5 | 16 | `BLOCK MATCH PASS 16/16` (`kv_chain_p0.txt`) | 9.8e-5 | 4.1e-5 | 16/16 |
| p1 | 4 | 16 | `BLOCK MATCH PASS 16/16` (`kv_chain_p1.txt`) | 4.2e-5 | 3.4e-5 | 16/16 |
| p2 | 18 | 16 | `BLOCK MATCH PASS 16/16` (`kv_chain_p2.txt`) | 1.1e-4 | 6.1e-5 | 16/16 |

Over all 48 rows the largest max abs difference is 1.1e-4 and the median 4.0e-5. The greedy ids of
the same run equal the reference ids for all three prompts (`TEXT MATCH PASS 3/3`).

## How the cache moves between calls

- The caches are signature inputs and outputs: `key_cache_<i>` / `value_cache_<i>` in,
  `key_cache_<i>_new` / `value_cache_<i>_new` out, each [1, 8, 64, 128] fp32 (8 key/value heads,
  a fixed length of 64 positions, head dim 128).
- Each layer writes the new keys and values with `DynamicUpdateSlice` at a runtime start index:
  the int32 input `cache_position` = {0, 0, position, 0} (`graph/qwen3_graph.cc` lines 131–135;
  the host sets it in `graph/qwen3_main.cc` lines 336–338). Nothing in the graph is a constant
  start index.
- Rebind, by handle and without a copy (`graph/qwen3_main.cc`, the `rebind` lambda, lines 445–472;
  called after prefill at line 475 and after every decode step at line 497):
  `runner.GetOutput(sig, "<kind>_cache_<i>_new")` returns a `TensorHandle` whose buffer is a
  `LitertBuffer` holding `TensorBuffer::Duplicate()` of the output buffer
  (LiteRT `tensor/runners/litert/litert_dynamic_runner.h` lines 547–608). `runner.SetInput("decode",
  "<kind>_cache_<i>", handle)` sees the `LitertBuffer`, duplicates it again and calls
  `SetInputBuffer` (same header, lines 386–410), which "retains a TensorBuffer without copying its
  contents" (line 225). `Duplicate()` adds a reference to the same underlying buffer handle
  (`litert/cc/litert_tensor_buffer.h` lines 382–394), so no cache bytes are copied on the host.
- A consequence of this form: from the second decode step on, the decode signature's cache input
  and its cache output are the same buffer, so `DynamicUpdateSlice` reads and writes one buffer in
  place. The CPU results above are with that aliasing.
- `--rebind=copy` (a diagnostic) reads each cache output to the host and writes it into the next
  call's input instead. It was not needed: the handle form passed on the CPU and on Metal.

## Results, Metal fp32 (buffer storage)

The same chain on the Metal delegate at fp32, with the same handle rebind and the same bar:
`BLOCK MATCH PASS 16/16` for p0, p1 and p2 (`kv_chain_metal_fp32_p0.txt`, `_p1`, `_p2`). Over the
48 rows the largest max abs difference is 1.0e-4 and the median 4.6e-5; every row's argmax equals
the reference's. With buffer storage and with the default storage the 48 rows are byte-identical
(`delegation_metal.md`).
