# Fixture: wrong status words (expect exit 2, rows 2 and 3 named)

| # | row | how, on the Tensor API | pass when | status | evidence |
|---|---|---|---|---|---|
| 1 | Architecture and op inventory | list the ops | every op mapped | tried | evidence/ops.md |
| 2 | Reference forward | dump the blocks | dumps exist | done | evidence/manifest.json |
| 3 | Phone run | run on the phone GPU | match and time | blocked: later | |
