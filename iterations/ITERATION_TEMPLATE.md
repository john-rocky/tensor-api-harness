# Iteration <N> — <model>: <one goal>

This file is the instruction the agent gets for one iteration. One iteration has one goal.
Fill in every angle-bracket field. Keep the "Close" section as it is; the agent writes it at the end.

## Read first
- `skill_table.md` at the repo root (the rows and the status words), then `models/<model>/skill_table.md` (the filled rows so far).
- `models/<model>/iterations/ITERATION_<N-1>.md` and earlier: the corrections and dead ends of the previous iterations. Do not retry a dead end without a new reason.
- The reference implementations named in `README.md`.

## Goal (one)
<the rows of the skill table this iteration takes, by number, and the end state in one sentence>

## Acceptance
<the numbers: the output-match bar (max abs difference, cosine), the token count for the text match,
the device and the backend, the precision label on every number>

## Stop conditions
<what makes the agent stop and ask a person instead of working around it: an op with no Tensor API
equivalent in a required block, a runtime failure that does not reproduce in a minimal graph after one
hour, a device that is not reachable>

## Scope
<the files the agent may change; everything else is read only>

## Do not
- Do not write `tried` without an evidence path that exists.
- Do not turn a `not yet` row into `not applicable` or `blocked` to make the done check pass; `blocked` needs the cause after the colon.
- Do not change the thresholds in `checks/` during the iteration; a new bar is a new iteration.
- Do not commit weights, generated graphs or dumps.

## Report
At the end, run `python3 tools/done_check.py models/<model>/skill_table.md` and paste its output
into the close, then run `python3 tools/iteration_close_check.py models/<model>/iterations/ITERATION_<N>.md`.

## Close (the agent writes this, in `models/<model>/iterations/ITERATION_<N>.md`)
The five sections of `iterations/ITERATION_CLOSE.md`, each with at least one line ("none" counts).
