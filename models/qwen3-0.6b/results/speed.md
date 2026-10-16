# Row 13 — decode time per form against the CPU fp32 floor

Apple M4 Max, macOS 27.0, measured 2026-10-15 between 18:32:53 and 18:34:32 JST while no other
measurement ran on the machine (one window, 100 s, all forms in one script; nothing else was built or
run inside it). Every form ran its three prompts once: prefill, then 15 decode steps (greedy 16), on
the production signatures (no dumps), with the KV caches rebound by handle. Each `runner.Run` call is
timed with `steady_clock`; setting the inputs, the rebind and reading the logits are outside the timed
interval. "decode ms/token median" is the median of the three per-prompt medians over the 15 steps;
min and max are over all 45 steps. The first decode call of each prompt is the slow one (the max
column). Prefill is one call per prompt, the first call of the compiled model.

Metal forms run with buffer storage; "metal int8" is fp32 delegate precision with int8 weights;
"metal fp16" is fp16 delegate precision with fp32 weights. CPU forms use the XNNPACK delegate at the
runtime's default thread count. "match" is the greedy ids equal to the reference before the first
mismatch, out of 48, with the bar of at least 44 (`quantization.md`, `attention_forms.md`); the timed
run's ids gave the same counts as the match runs for every form. The three forms without a number did
not run (CPU sdpa: tensor allocation failed; Metal int4: segmentation fault at delegate init).
`timing.json` holds these numbers in the form `checks/speed_floor.py` reads, and `speed_floor.txt`
is its output.

| form | prefill ms p0 (S 5) | prefill ms p1 (S 4) | prefill ms p2 (S 18) | decode ms/token median | min | max | per-prompt medians | match (ids /48) | serialized bytes | note |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu fp32 raw | 42.8 | 25.4 | 47.6 | 18.12 | 17.80 | 47.95 | 17.98 / 18.12 / 18.54 | PASS (48/48) | 2384789652 |  |
| cpu fp32 rms | 38.1 | 26.2 | 48.1 | 18.10 | 17.90 | 56.30 | 18.19 / 18.07 / 18.10 | PASS (48/48) | 2384886420 |  |
| cpu fp32 sdpa | — | — | — | — | — | — | — | FAIL (no ids (run failed)) | 2384870356 | not timed (no timing file: the run failed) |
| cpu int8 | 18.0 | 14.6 | 28.8 | 9.23 | 9.00 | 27.25 | 9.23 / 9.26 / 9.11 | FAIL (41/48) | 1071852948 |  |
| cpu int4 | 33.3 | 19.5 | 75.9 | 11.39 | 11.17 | 23.56 | 11.47 / 11.39 / 11.36 | FAIL (6/48) | 898490644 |  |
| cpu int8 + lm_head int8 | 12.8 | 10.7 | 21.3 | 6.06 | 5.88 | 8.47 | 6.06 / 6.00 / 6.06 | PASS (48/48) | 1231082068 |  |
| cpu int4 + lm_head int8 | 30.1 | 16.2 | 69.6 | 8.32 | 8.22 | 9.58 | 8.32 / 8.32 / 8.33 | FAIL (20/48) | 1057719764 |  |
| metal fp32 raw | 28.8 | 24.8 | 32.1 | 8.06 | 7.87 | 24.48 | 8.21 / 8.06 / 7.94 | PASS (48/48) | 2384789652 |  |
| metal fp32 rms | 26.9 | 26.5 | 29.8 | 7.86 | 7.62 | 23.73 | 7.84 / 7.88 / 7.86 | PASS (48/48) | 2384886420 |  |
| metal fp32 sdpa | 26.2 | 24.9 | 33.0 | 8.60 | 8.33 | 24.42 | 8.46 / 8.64 / 8.60 | PASS (48/48) | 2384870356 |  |
| metal fp32 rms+sdpa | 27.5 | 26.2 | 30.4 | 8.30 | 8.13 | 26.96 | 8.43 / 8.27 / 8.30 | PASS (48/48) | 2384967060 |  |
| metal fp16 raw | 17.4 | 15.4 | 23.6 | 5.76 | 5.71 | 13.34 | 5.76 / 5.95 / 5.76 | FAIL (0/48) | 2384789652 |  |
| metal int8 | 19.7 | 16.7 | 32.0 | 5.88 | 5.60 | 14.81 | 5.88 / 5.94 / 5.63 | PASS (48/48) | 1071852948 |  |
| metal int4 | — | — | — | — | — | — | — | FAIL (no ids (run failed)) | 898490644 | not timed (no timing file: the run failed) |
| metal int8 + lm_head int8 | 16.0 | 13.5 | 32.2 | 4.72 | 4.67 | 11.05 | 4.72 / 4.79 / 4.72 | PASS (48/48) | 1231082068 |  |
| metal int4 + lm_head int8 | — | — | — | — | — | — | — | FAIL (no ids (run failed)) | 1057719764 | not timed (no timing file: the run failed) |

## Timing definition (iteration 6)

The table above times `runner.Run` alone. On the phone's OpenCL, `Run` returns before the GPU has
finished, so iteration 6 added `--time_readback`. With it, the timed interval ends after the logits
output buffer is locked and one value read (`phone_s26.md`, "Time"). Three forms of this table were
measured again on the same Mac with both definitions, back to back in one 52 s window on 2026-10-17
(02:21:09 to 02:22:01 JST). The difference is 5% or less for each form, so the table above stands
with the run-only definition. The forms, decode ms/token run only → run + readback:

- metal fp32 raw: 8.117 → 8.159 (+0.5%)
- metal int8 + lm_head int8: 4.780 → 4.782 (+0.0%)
- cpu fp32 raw: 18.701 → 18.251 (−2.4%)

The prefill times moved by 1.7 ms or less on Metal and by up to 4.1 ms on the CPU (p0 29.6 → 25.5
ms), in both directions. `timing.json` and `speed_floor.txt` are unchanged.
