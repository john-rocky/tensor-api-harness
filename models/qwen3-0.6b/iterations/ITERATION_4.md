# Iteration 4 — Qwen3-0.6B: the same graph on a Galaxy S26, CPU and OpenCL

## Read first
- `skill_table.md` at the repo root, then `models/qwen3-0.6b/skill_table.md`.
- `iterations/ITERATION_1.md` to `iterations/ITERATION_3.md` (this model): the graphs, the bars, the
  KV chain check, the Metal delegation and precision results, the int8 and int4 forms and the speed
  floor on the Mac.
- LiteRT's `tensor/examples/gemma3/README.md` (Android cross-compilation, the OpenCL/GL accelerator
  prebuilt, the push and run form), `WORKSPACE` (the Android NDK) and `.bazelrc`
  (`--config=android_arm64`).

## Goal (one)
Row 14 on a Galaxy S26 (SM-S942Q): build the same graph for Android arm64, run it on the phone's CPU
(XNNPACK) and GPU (OpenCL), compare the generated ids and the chained decode logits with the reference
(and the phone CPU fp32 logits with the Mac CPU fp32 logits), and time decode on the phone. The phone
runs saved models built on the Mac with one fixed prefill length (32, prompts padded on the right), so
one file per form is pushed: fp32, int8 + int8 lm_head (the fastest passing form on the Mac) and int4 +
int8 lm_head (for the phone's time only; it fails the match on the Mac CPU).

## Acceptance
- Row 14: the phone CPU fp32 chain passes the KV chain bar (`manifest_chain`) with 16/16 rows for each
  of the 3 prompts; every form that ran on the phone's GPU has its count of generated ids equal to the
  reference out of 48 and its delegation numbers (total nodes, delegated nodes, partitions); a form
  that did not run on the GPU has the reason with its numbers; every form that ran has its decode ms
  per token measured on the phone inside the device slot. `results/phone_s26.md`,
  `results/timing_s26.json` and `results/speed_floor_s26.txt` (with its exit code) exist.
- The padded prefill on the Mac CPU passes the KV chain bar 16/16 for each prompt and the text match
  3/3 before anything goes to the phone.
- `tools/done_check.py` prints `DONE` with exit code 0 (15 of 15 rows); `tools/iteration_close_check.py`
  passes; this file has a `## Time` section with the minutes the phone was in use.
- The bars are the ones written before iteration 2 and are not changed.

## Stop conditions
- The Android cross-compilation fails with NDK 28 and with NDK 29.
- The padded prefill does not match the reference on the Mac CPU after one fix.
- The phone is not available within six hours of waiting, is not connected, is locked, runs hot
  (above 45 °C) or runs another measurement: no slot is taken.
- The phone restarts during the slot: no further GPU form; close with the CPU results.

## Scope
`models/qwen3-0.6b/` only: `graph/qwen3_main.cc` (fixed prefill length, save only, run a saved
model), new files in `results/`, this file, row 14 of `skill_table.md` and an added section in
`ENVIRONMENT.md`. Everything else in the repository is read only. Weights, generated `.tflite` files
and dumps are not committed. With every flag at its default, the graph is the one iteration 2 ran.

## Do not
- Do not write `tried` without an evidence path that exists.
- Do not turn a `not yet` row into `not applicable` or `blocked` to make the done check pass; `blocked` needs the cause after the colon.
- Do not change the thresholds in `checks/` during the iteration; a new bar is a new iteration.
- Do not commit weights, generated graphs or dumps.
- Timing only inside the machine's quiet window; report the median, keep every measured form.
- No adb before the go for the phone; one device slot of at most 40 minutes; delete the device files at
  the end.
- Do not run a GPU form whose model needs more memory than the phone has free: a form runs on the GPU
  only when the free memory is at least six times its file size plus 1 GB (the OpenCL delegate's
  initialization has taken five to six times the model's bytes).

## Progress
- `qwen3_main` has three new flags: `--prefill_len`, `--serialize_only` and `--load_tflite`
  (`ENVIRONMENT.md`, "Iteration 4").
- Padded prefill on the Mac CPU, fp32, length 32:
  - KV chain 16/16 for each prompt, text match 3/3
    (`results/kv_chain_mac_cpu_fp32_prefill32_p*.txt`, `results/text_match_mac_cpu_fp32_prefill32.txt`).
  - The chain rows and ids are byte-identical to iteration 2's prompt-length chain.
  - The saved file, run with `--load_tflite`, gives byte-identical chain rows.
  - The saved int8 + int8 lm_head file passes 48/48. The int4 + int8 lm_head file gives 20/48, as in
    iteration 3. Their ids equal iteration 3's.
  - The `std::lround` change of iteration 3, rebuilt and rerun, gives ids, chain bytes and quantized
    weight bytes identical to iteration 3's.
- The Android arm64 build passed with NDK 28 on the first NDK tried.
- The planned phone, a Galaxy S26 (SM-S942Q) on this machine's USB, was not connected when the device
  slot came: the hold was taken and given back in the same second, and nothing was run.
- Row 14 was then run on a Galaxy S26 (SM-S942U1) in Google's Developer Device Platform: one session,
  one job, no adb (`results/phone_s26.md`).
  - CPU fp32: KV chain 16/16 for each prompt (`results/kv_chain_s26_cpu_fp32_p*.txt`), text match 3/3
    (48/48), decode 39.468 ms/token. The ids are identical to the Mac CPU's; 1 of 48 chain files is
    byte-identical; the largest max abs difference is 9.20e-05.
  - CPU int8 + int8 lm_head: 48/48, decode 11.665 ms/token.
  - OpenCL int8 + int8 lm_head was not run because of the memory rule: MemAvailable was 6,553,616 kB,
    and the rule needs 8,269,072 kB (six times the file plus 1 GiB).
  - `checks/speed_floor.py` exits 0 (`results/speed_floor_s26.txt`).

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Row 14:**
  - The phone CPU fp32 chain passes 16/16 for each prompt.
  - No form ran on the phone's GPU. The reason and its numbers are in `results/phone_s26.md`.
  - Both forms that ran were timed on the phone.
  - `results/phone_s26.md`, `results/timing_s26.json` and `results/speed_floor_s26.txt` (exit code
    0) exist.
- **Mac padded prefill:** met.
- **Done check:** met.
- **Bars:** unchanged.

## Corrections
- The phone is a Galaxy S26 SM-S942U1 reached through Google's Developer Device Platform, not the
  SM-S942Q on this machine's USB that the goal names. The planned phone was not connected when its
  slot came.
- The lab job ran once, with no device slot or hold. The phone was in use for the job's execution:
  115.3 s.
- The int4 + int8 lm_head file was not pushed. It fails the match on the Mac CPU, so it would have
  been timed only, and the session's push was kept to the two forms that pass.
- The lab phone's MemAvailable, about 6.5 GB of 11.4 GB, was below the memory rule's threshold for
  every form, so the OpenCL half of row 14 has no measurement.

## Dead ends
- **OpenCL on the lab phone.** No form fits the memory rule there: the smallest, int4 + int8 lm_head
  at 1,057,719,764 bytes, needs 7,246,153 kB, against the 6,553,616 kB the phone had at the start.
  Not run.
- **The Android build under quiet windows.** It was stopped at each measurement window and started
  again after it: 16 attempts, 936 s of build time, 6 h 39 min of wall time. The window check
  before a command only looks once, at the start, so a long build has to watch for windows itself.

## Op and delegate observations
- Phone CPU (XNNPACK): 3880 of 3994 nodes, 58 partitions, for both fp32 and int8 + int8 lm_head. The
  Mac CPU has the same numbers.
- Phone CPU fp32 against the Mac CPU fp32, same file:
  - The ids are identical.
  - 1 of 48 chain files is byte-identical; the largest max abs difference is 9.20e-05.
  - Against the reference, the largest chain max abs is 1.04e-04 (bar 0.05, cosine 1.00000000).
- Phone CPU int8 + int8 lm_head against the Mac, same file:
  - The ids are identical.
  - 25 of 48 chain files are byte-identical.
  - The largest max abs difference is 1.47 (p0, step 14).
- OpenCL: not run (memory rule), so there are no OpenCL partitions or compile messages.
- The runtime prints `NPU accelerator could not be loaded and registered: kLiteRtStatusErrorNotFound`
  at start on the phone; the CPU runs go on.
- CPU frequency ceilings were capped during the whole job:
  - at the start: policy0 3,187,200 of 3,628,800 kHz, policy6 3,398,400 of 4,742,400 kHz;
  - from the end of the fp32 run: 2,899,200 and 2,880,000 kHz.
  - Whether the times on an uncapped phone would differ: unverified.
- The phone did not restart (uptime 2,184 s, then 2,207 s). Thermal status 0 at every read.
- The XNNPACK thread count is not printed by the runtime: unverified.

## Environment
- `ENVIRONMENT.md`, "Iteration 4":
  - the new flags;
  - the padded and saved forms;
  - the phone file sizes;
  - the NDK, the build command and the hashes of the binary and the accelerator `.so`;
  - the phone, the lab route, the job script's steps and each run's command.
- The match runs on the Mac waited 0 s for a quiet window.
- The upload took 219.7 s for 3,637,255,206 bytes (16.6 MB/s).
- The lab session took 284 s.

## Files left
- `graph/qwen3_main.cc`: the new flags. Keep.
- `results/`, new files. Keep:
  - `phone_s26.md`, `timing_s26.json`, `speed_floor_s26.txt`
  - `kv_chain_s26_cpu_fp32_p*.txt`
  - `text_match_s26_cpu_fp32.txt`, `text_match_s26_cpu_int8_lmh8.txt`
  - `kv_chain_mac_cpu_fp32_prefill32_p*.txt`, `text_match_mac_cpu_fp32_prefill32.txt`
- Not committed, outside the repository:
  - the saved `.tflite` files (deleted at close);
  - the Android binary and accelerator;
  - the dumps and the pulled lab output.

## Time
Start 21:00 (2026-10-15), end 23:20 (2026-10-16), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 30 (21:00–21:10 code and scripts; 22:40–22:57 the lab route's script and driver; 23:05–23:20 closing) |
| builds | 1 host, 16 Android attempts | 16 (host 5.3 s; Android 936 s of build time) |
| waiting for the machine: quiet windows during the Android build | 15 | 386 (23,167 s, 21:10–03:49) |
| waiting for a decision on the phone | 2 | 1130 (03:50–18:21 for the go; 18:21 to about 22:40 after the planned phone was found missing) |
| waiting for the phone: the queue | 1 | 0 (the hold was free; given back the same second, phone missing) |
| waiting for the phone: upload to the lab | 1 | 4 (219.7 s) |
| waiting for the phone: allocation and push in the lab | 1 | 3 (154.6 s) |
| runs and checks | Mac matches 1, lab job 1, checks 1 | 3 (21:03:44–21:04:09 Mac; 115.3 s lab execution; 2 s checks) |
| rework after a failure | 1 | 17 (22:40–22:57: the lab route after the planned phone was not connected) |
| phone in use | 1 | 2 (115.3 s, the lab execution) |
