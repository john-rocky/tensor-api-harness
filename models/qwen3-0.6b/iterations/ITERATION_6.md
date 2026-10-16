# Iteration 6 — Qwen3-0.6B: the timing definition

## Read first
- `iterations/ITERATION_5.md` and `results/phone_s26.md` (section "Attached unit (SM-S942Q),
  OpenCL").
- LiteRT's `litert/c/options/litert_gpu_options.h`, `LrtSetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion`.

## Goal (one)
The timing definition. Until now every time was the wall time of `runner.Run` alone. On the phone's
OpenCL, the 32-position prefill timed shorter than one decode step, which suggests that `Run`
returns before the GPU finishes.

So the timed interval now ends after the logits output buffer is locked and one value read. With
that definition:
- time the phone's OpenCL int8 again, with the CPU floor timed the same way;
- compare both definitions on three Mac forms to decide whether the Mac table stands.

## Acceptance
- On the phone, opencl int8 has its decode ms/token and prefill ms with the new definition. The
  32-position prefill takes at least one decode step; if it does not, the cause is written down.
- The phone's cpu fp32 floor is timed with the same definition.
- The Mac difference between the two definitions is given in percent for each form, and the table's
  status follows from it: it stands at 5% or less, and is timed again otherwise.
- The phone results use the new definition. The run-only numbers stay, labeled as the dispatch time.
- `tools/done_check.py` exit code 0; `tools/iteration_close_check.py` passes; a `## Time` section.

## Stop conditions
- With the new definition the prefill is still shorter than one decode step.
- No device slot by 02:50 JST.
- The phone does not cool within 10 minutes and its Thermal Status is 3 or more.
- The phone restarts.

## Scope
`graph/qwen3_main.cc` (two flags; with the defaults the graph and the outputs are unchanged).
In `results/`:
- `timing_s26_attached.json` and `speed_floor_s26_attached.txt`, both overwritten;
- the attached-unit section of `phone_s26.md`;
- a note in `speed.md`.

Also a section in `ENVIRONMENT.md`, and this file. The bars are unchanged.

## Do not
- Do not change the thresholds in `checks/`.
- Do not delete the run-only numbers.
- Do not commit weights, generated graphs or dumps.
- One device slot of at most 20 minutes, cooling included; delete the device files at the end.
- Run a GPU form only when MemAvailable is at least seven times its file in kB plus 512 MiB.

## Progress
- `--time_readback` (default on) and `--gpu_wait_hint` in `qwen3_main`. On the Mac CPU, the saved int8
  + int8 lm_head file gives the same ids with both definitions: 6.508 ms/token run only, 6.422 with
  the readback.
- Phone, one slot, 3 min 31 s, no cooling needed (AP 34.2 °C at the start):
  - cpu fp32: 48/48, chain 16/16 for each prompt, decode 38.108 ms/token (37.335 run only).
  - opencl int8: 48/48, decode **26.730** ms/token, prefill 70.3 / 64.6 / 62.8 ms. Run only it was
    3.297 ms/token and 3.2 / 2.4 / 2.2 ms. The new prefill is longer than a decode step.
  - opencl int8 with `--gpu_wait_hint`, run only: 2.823 ms/token, prefill 1.8 to 2.6 ms. The hint
    does not make `Run` wait on this phone.
  - opencl int8 + int8 lm_head: not run. The memory rule asked for 8,939,891 kB, and MemAvailable was
    8,505,892 kB.
  - cpu int8 + int8 lm_head: 48/48, decode 12.991 ms/token (12.729 run only).
  - The ids and all 48 chain files of every form that ran are byte-identical to iteration 5's.
- `checks/speed_floor.py` exits 0: the fastest passing form on the phone is cpu int8 + int8 lm_head
  at 12.991 ms/token against the floor of 38.108; opencl int8 is at 26.730.
- Mac, the two definitions back to back: metal fp32 raw +0.5%, metal int8 + int8 lm_head +0.0%, cpu
  fp32 raw −2.4%. The Mac table stands, with a note in `results/speed.md`.

## Report
`python3 tools/done_check.py models/qwen3-0.6b/skill_table.md` (exit code 0):

```
models/qwen3-0.6b/skill_table.md: tried 15, not applicable 0, blocked 0, not yet 0
DONE models/qwen3-0.6b/skill_table.md
```

Against the acceptance list:
- **Phone OpenCL int8, new definition:** met. The prefill is longer than a decode step.
- **Phone CPU floor:** met.
- **Mac:** at most 2.4% for each form; the table stands.
- **Phone results:** use the new definition; the run-only numbers are kept as dispatch times.
- **Done check:** met.

## Corrections
- Iteration 5's OpenCL numbers were the dispatch time, not the time to compute a token. opencl
  int8 takes 26.730 ms per decode token, not 3.297, and 63 to 70 ms per prefill, not 2 to 3.
- Iteration 5's speed floor named opencl int8 as the fastest passing form. With the new definition
  it is cpu int8 + int8 lm_head, 12.991 ms/token.
- opencl int8 + int8 lm_head has no time with the new definition. Its iteration 5 numbers are
  dispatch times.

## Dead ends
- **`HintWaitingForCompletion` on this phone.** With the hint set, `Run` still returned in 2.8 ms per
  decode step. The header names Mali as an example of where the hint applies; this phone has an
  Adreno GPU. Reading the output is what makes the time include the GPU.

## Op and delegate observations
- OpenCL `runner.Run` does not wait for the GPU. Decode ms/token for opencl int8:
  - run only 3.297, with the readback 26.730 (8.1x);
  - with the wait hint, run only 2.823.
- Prefill for opencl int8: run only 2.2 to 3.2 ms, with the readback 62.8 to 70.3 ms.
- On the CPU (XNNPACK), the two definitions differ by 2% on the phone and by 2.4% or less on the Mac.
  Metal differs by 0.5% or less. That suggests Metal's `Run` already includes the GPU work; it was
  not checked further.
- Delegation as in iteration 5:
  - OpenCL: 3994 of 3994 nodes, 2 partitions, no fallback, no error.
  - CPU: 3880 of 3994, 58 partitions.
- Memory during opencl int8, every 3 s:
  - a drop of 7.06 times the file (lowest 1,296,916 kB);
  - with the hint, 7.20 times (lowest 878,704 kB).
- cpu fp32's p1 prefill took 935.1 ms (one slow call; its cause was not measured).
- The phone did not restart (uptime 98,975 to 99,183 s); Thermal Status 0 throughout. AP 34.2 to
  44.6 °C at the run starts.
- Unverified:
  - how much of the 26.730 ms is the logits copy rather than the compute. The lock reads
    151,936 floats.
  - the time of opencl int8 + int8 lm_head.

## Environment
- `ENVIRONMENT.md`, "Iteration 6".
- Mac window: 52 s for the six timing runs; the builds, the save and the checks waited 0 s.
- Phone: the attached Galaxy S26 (SM-S942Q), one slot of 3 min 31 s.

## Files left
- `graph/qwen3_main.cc`: the two flags. Keep.
- `results/`, kept:
  - `timing_s26_attached.json` and `speed_floor_s26_attached.txt`, overwritten with the new
    definition;
  - `phone_s26.md`, with a "Time" section;
  - `speed.md`, with a note.
- Not committed: the saved `.tflite` files (deleted at close), the dumps and the Mac timing JSON.

## Time
Start 02:13, end 02:25 (2026-10-17), JST.

| part | count | minutes |
|---|---|---|
| coding and reading | — | 7 (02:13–02:17 code and scripts; 02:22–02:25 closing) |
| builds | 1 host, 1 Android | 0.2 (4.2 s and 10 s) |
| waiting for the machine: quiet windows | 4 | 0 (0 s each) |
| waiting for the phone: the queue and cooling | 1 | 0 |
| runs and checks | Mac 1 check run and 6 timing runs, phone 4 runs, checks 1 | 4 (2 s; 52 s; 134 s; 2 s) |
| rework after a failure | 0 | 0 |
| phone in use | 1 | 4 (211 s) |
