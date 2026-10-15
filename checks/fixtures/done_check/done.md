# Fixture: every row tried or not applicable (expect exit 0)

The table below is inside a code fence and must be ignored:

```
| # | row | how | pass when | status | evidence |
|---|---|---|---|---|---|
| 9 | Fenced row | - | - | not yet | |
```

| # | row | how, on the Tensor API | pass when | status | evidence |
|---|---|---|---|---|---|
| 1 | Architecture and op inventory | list the ops | every op mapped | tried | `evidence/ops.md` |
| 2 | Reference forward | dump the blocks | dumps exist | tried | [the manifest](evidence/manifest.json), see also notes |
| 3 | State and KV cache | chain decode steps | N steps match | not applicable: single-forward model, no state between calls | |
| 4 | Environment record | record the commit | recorded | tried | checks/fixtures/done_check/evidence/env.md |
