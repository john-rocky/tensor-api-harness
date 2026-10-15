# Fixture: `tried` without an evidence file that exists (expect exit 2, rows 1 and 2 named)

| # | row | how, on the Tensor API | pass when | status | evidence |
|---|---|---|---|---|---|
| 1 | Architecture and op inventory | list the ops | every op mapped | tried | |
| 2 | Reference forward | dump the blocks | dumps exist | tried | evidence/does_not_exist.json |
| 3 | Environment record | record the commit | recorded | tried | evidence/env.md |
