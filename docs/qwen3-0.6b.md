# Qwen3-0.6B

The skill table is [`skill_table.md`](../skill_table.md). An iteration is one run of the coding agent on one instruction file.

[Qwen3-0.6B](../models/qwen3-0.6b/) went through all 15 rows in six iterations
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
  times are taken through the readback of the logits; on the Mac the two definitions differ by 2.4%
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
