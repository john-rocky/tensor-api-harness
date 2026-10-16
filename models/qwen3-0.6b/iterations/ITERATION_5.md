# Iteration 5 — Qwen3-0.6B: OpenCL on an attached Galaxy S26

## Read first
- `iterations/ITERATION_4.md` (this model): the padded prefill, the saved models, the Android build,
  the lab run and why its OpenCL half was not run.
- `results/phone_s26.md`.

## Goal (one)
The OpenCL half of row 14 on a Galaxy S26 (SM-S942Q) attached over USB. Run the int8 forms on the
phone's GPU and record:
- the delegation (total nodes, delegated nodes, partitions);
- the count of generated ids equal to the reference out of 48;
- the KV chain against the reference (fp32 bar, a record for int8);
- decode ms per token.

The same phone's CPU fp32 run is the floor for `checks/speed_floor.py`.

## Acceptance
- At least one of opencl int8 and opencl int8 + int8 lm_head ran, with its delegation numbers, its
  match count out of 48 and its decode ms per token. A form that did not run has the reason with its
  numbers.
- The phone CPU fp32 chain passes the KV chain bar 16/16 for each prompt.
- `results/timing_s26_attached.json` and `results/speed_floor_s26_attached.txt` (with its exit code)
  exist.
- `tools/done_check.py` prints `DONE` with exit code 0 (15 of 15 rows); `tools/iteration_close_check.py`
  passes; this file has a `## Time` section.
- The bars are unchanged.

## Stop conditions
- The phone is not connected or is locked.
- The phone is above 45 °C: nothing runs until the AP and SKIN sensors are at or below 45 °C and
  every sensor's status is at most 1. After 15 minutes the runs start anyway, and the temperatures
  at each run's start are recorded with the times.
- The phone restarts during the slot: no further GPU form.
- The device slot would pass 40 minutes: the later forms are dropped.

## Scope
New files in `results/`, an added section in `results/phone_s26.md`, this file, the evidence of row 14
in `skill_table.md` and a paragraph in `ENVIRONMENT.md`. No code changes.

## Do not
- Do not change the thresholds in `checks/`.
- Do not commit weights, generated graphs or dumps.
- One device slot of at most 40 minutes; delete the device files at the end.
- Run a GPU form only when MemAvailable, read right before the run, is at least five times its file
  size in kB plus 512 MiB. This is a condition for running a form, not a bar.

## Progress
- The three model files were saved again on the Mac (fp32, int8 with the fp32 lm_head, int8 + int8
  lm_head). The fp32 and int8 + int8 lm_head sizes equal iteration 4's.
- First slot: the phone was hot.
  - Readings: Thermal Status 4, AP 50.8 °C, SKIN 45.6 °C.
  - It was given back after 3 s, and nothing was pushed.
- Second slot: 6 min 9 s.
  - The first 125 s were cooling with nothing running. They ended when the AP and SKIN sensors were
    at or below 45 °C (41.3 and 39.8 °C) and every sensor's status was at most 1.
  - Then the push and four runs. All forms ran:
    - cpu fp32: KV chain 16/16 for each prompt, text match 48/48, decode 37.335 ms/token.
    - opencl int8: GPU 3994 of 3994 nodes, 2 partitions; 48/48; decode 3.297 ms/token.
    - opencl int8 + int8 lm_head: GPU 3994 of 3994 nodes, 2 partitions; 48/48; 3.314 ms/token.
    - cpu int8 + int8 lm_head: 48/48, decode 12.729 ms/token.
  - The OpenCL chains fail the fp32 bar on every row (recorded): largest max abs 1.10, smallest
    cosine 0.99880.
  - The OpenCL ids are identical to the Mac Metal ids of the same quantization.
  - The phone did not restart. The AP sensor read 42.3 to 48.7 °C at the run starts.
- `checks/speed_floor.py` exits 0: opencl int8 at 3.297 ms/token against the floor of 37.335.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **OpenCL:** both int8 forms ran, with delegation numbers, 48/48 and their decode times.
- **Phone CPU fp32 chain:** 16/16 for each prompt.
- **Files:** `timing_s26_attached.json` and `speed_floor_s26_attached.txt` (exit code 0) exist.
- **Done check:** met.
- **Bars:** unchanged.

## Corrections
- The first slot was not used: the phone was above 45 °C. The second slot cooled the phone first,
  inside the slot, and its 125 s count toward the slot.
- After the 125 s of cooling, the AP sensor rose above 45 °C again during the runs (up to 48.7 °C at a
  run start). The runs went on, and the temperatures are recorded with the times.
- The memory rule's margin was too small for opencl int8. MemAvailable fell by 6.85 times the file
  size during that run, more than five times the file plus 512 MiB. The lowest read was 687,616 kB.

## Dead ends
- **Running straight after another job.** The phone came back at 50.8 °C (AP). Waiting outside the
  slot would have let other jobs take the phone again, so the next slot cooled the phone inside it:
  125 s.

## Op and delegate observations
- OpenCL, both int8 forms:
  - 3994 of 3994 nodes delegated, 2 partitions, fully accelerated;
  - no CPU fallback, no compile or runtime error.
  - Precision fp32, buffer storage.
- OpenCL setup: the delegation report came after 24 s (int8) and 29 s (int8 + int8 lm_head). The
  runner was ready after 59 s and 55 s. The program compiles twice.
- MemAvailable during the OpenCL runs, read every 3 s:
  - int8: 7,853,188 kB down to 687,616 kB, a drop of 6.85 times the file size;
  - int8 + int8 lm_head: 7,730,216 kB down to 1,841,156 kB, a drop of 4.90 times;
  - the smaller file took more memory.
  - Why: unverified. The difference between the two files is the lm_head: in the int8 form its fully
    connected layer reads the fp32 table (151,936 × 1024), in the other an int8 copy.
- Generated ids:
  - Every form on this phone gives the same ids, 48 of 48 equal to the reference.
  - opencl int8 equals Mac Metal int8; opencl int8 + int8 lm_head equals Mac Metal int8 + int8 lm_head.
  - On the Mac CPU, int8 with an fp32 lm_head gave 41 of 48.
- Chains against the reference:
  - opencl int8: largest max abs 1.10, smallest cosine 0.99885;
  - opencl int8 + int8 lm_head: 1.08 and 0.99880;
  - argmax agrees on all 48 rows for both.
- This phone's CPU against the lab phone's CPU, same files: ids identical and 48 of 48 chain files
  byte-identical, for both fp32 and int8 + int8 lm_head.
- The phone did not restart (uptime 97,951 to 98,316 s).
- Thermal Status 1 at every run start; 2 at the slot start.
- Runtime: the same LiteRT commit and the same OpenCL/GL accelerator prebuilt as iteration 4 (sha256
  `6ec1f5d5…6d2d34`).
- Unverified:
  - whether OpenCL `runner.Run` returns after the GPU finishes. The 32-position prefill timed 2.2 to
    3.2 ms, less than one decode step, so the OpenCL times may leave out GPU time;
  - this unit's CPU frequency ceilings (not read).

## Environment
- `ENVIRONMENT.md`, "Iteration 4", the paragraph "Attached unit (iteration 5)".
- Mac: the host build of `qwen3_main` was redone (11.2 s), then the three files were saved (4 s).
- Push: 88 s for 4,709,100,242 bytes over USB.

## Files left
- `results/`, new files. Keep:
  - `timing_s26_attached.json`, `speed_floor_s26_attached.txt`
  - `kv_chain_s26_attached_cpu_fp32_p*.txt`, `kv_chain_s26_attached_opencl_int8_p*.txt`,
    `kv_chain_s26_attached_opencl_int8_lm_head_int8_p*.txt`
  - `text_match_s26_attached_cpu_fp32.txt`, `text_match_s26_attached_opencl_int8.txt`,
    `text_match_s26_attached_opencl_int8_lm_head_int8.txt`,
    `text_match_s26_attached_cpu_int8_lm_head_int8.txt`
- `results/phone_s26.md`: a new section "Attached unit (SM-S942Q), OpenCL". Keep.
- Not committed, outside the repository:
  - the saved `.tflite` files (deleted at close);
  - the pulled dumps.

## Time
Start 01:47, end 02:10 (2026-10-17), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 8 (01:47–01:51 reading and scripts; 02:06:40–02:10 closing) |
| builds | 1 host | 0.2 (11.2 s) |
| waiting for the machine: quiet windows | 3 | 0 (0 s each) |
| waiting for a decision | 1 | 2 (01:58:18–02:00, after the first slot) |
| waiting for the phone: the queue | 2 | 7 (435 s, 01:51:00–01:58:15; 0 s the second time) |
| waiting for the phone: cooling inside the slot | 1 | 2 (125 s) |
| runs and checks | Mac saves 1, phone runs 4, checks 1 | 3 (4 s Mac; 147 s phone; 3 s checks) |
| rework after a failure | 1 | 2 (the cooling step after the first slot) |
| phone in use | 2 | 6 (3 s first slot; 369 s second slot) |
