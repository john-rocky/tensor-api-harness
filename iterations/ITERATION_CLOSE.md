# Iteration close — the five sections every iteration record ends with

Write these five headings, in this order, at the end of `models/<model>/iterations/ITERATION_<N>.md`.
Each section has at least one line; write "none" when there is nothing. `tools/iteration_close_check.py`
refuses a record with a missing or empty section. The point is that what one iteration learned reaches
the next one through the file, not through memory.

## Corrections
What the instruction assumed and what was actually there. One line each, with the path that shows it.

## Dead ends
What was tried and did not work, and why (the flag, the version, the environment, the time it took).
Written so that the next person avoids it in one line.

## Op and delegate observations
Ops that fell back from the GPU (name, rank, dtype), the number of partitions, the fp16 versus fp32
difference (max abs difference, whether the argmax changed), compile and runtime failure text, the
LiteRT commit. Measured things only; mark a guess as "unverified".

## Environment
Python environment and the versions that matter (torch, transformers), the device id and OS, the
toolchain versions, whether another job held the machine during timing.

## Files left
Every file this iteration created and what it is for. Mark the ones that can be deleted.
