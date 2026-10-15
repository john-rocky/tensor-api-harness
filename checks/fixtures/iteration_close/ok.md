# Iteration 1 — example-model: rows 1 and 2 (fixture, expect exit 0)

## Progress
- Op inventory written, reference dumps written.

## Corrections
- The instruction assumed tied embeddings; the checkpoint has a separate lm_head (config.json).

## Dead ends
none

## 3. Op and delegate observations
- CPU only in this iteration; no delegate run.

## Environment
- Python 3.12, numpy 2.x; no other job held the machine.

## Files left
- reference/dump_reference.py — the reference dump script.
