# Phone run — Galaxy S26, CPU (XNNPACK) and OpenCL

Row 14, iteration 4. The phone ran the saved models built on the Mac (`--serialize_only`, prefill
length 32, prompts padded on the right with id 0) through `qwen3_main --load_tflite`; no graph was
built and no weights were read on the phone.

## Phone

- Galaxy S26 (SM-S942U1), a lab unit reached through Google's Developer Device Platform (no adb).
  SoC SM8850 (`ro.soc.model`), Adreno GPU (`ro.hardware.egl`).
- Android 16, build `BP4A.251205.006.S942U1UEU1AZAD`.
- MemTotal 11,389,764 kB. MemAvailable when the job started: 6,553,616 kB. Highest MemAvailable read
  during the job: 7,736,860 kB (after the CPU fp32 run).
- Thermal status 0 at every read, before and after each run. The highest temperature zone at the
  start was 41.7 °C; the zones were read only at the start.
- CPU frequency ceilings were below the hardware maximum for the whole job. At the start, policy0 was
  at 3,187,200 of 3,628,800 kHz and policy6 at 3,398,400 of 4,742,400 kHz. From the end of the CPU
  fp32 run to the end of the job, they were 2,899,200 and 2,880,000 kHz. The times below are taken
  under these ceilings.
- Uptime 2,184 s at the start and 2,207 s at the end: the phone did not restart.

## Route

- One lab session with one job and an execution timeout of 2,400 s.
- The job's binary was a shell script that ran the forms one after another and wrote every output
  to one directory. The session pushed the files flat into `/data/local/tmp/` before the job and
  pulled that output directory after it.
- Pushed files:
  - `qwen3_main`: 18,071,424 bytes
  - `libLiteRtClGlAccelerator.so`: 3,304,024 bytes
  - the prompt ids: 126 bytes
  - `r3_fp32.tflite`: 2,384,789,652 bytes
  - `r3_int8_lmh8.tflite`: 1,231,082,068 bytes
- Pulled: the generated ids, the timing JSON, the chain dumps of every run, each run's log and exit
  code, and the job's own log.
- The int4 + int8 lm_head file was not pushed: it fails the match on the Mac CPU (20 of 48), and
  Metal ends its first compile with a segmentation fault.
- The session took 284 s from creation to end. From the job's start to its execution's start was
  154.6 s; the report does not split this between allocation and the push. The execution took
  115.3 s, of which the script's own log covers 25 s (07:01:12 to 07:01:37 on the phone's clock).
- Android build: NDK 28.2.13676358, `bazelisk build -c opt --config=android_arm64`. Binary sha256
  `3bf835a252f06d7ab0417f0f0caa2b752e683f971d4e5674ccbd6aed1c68e7eb`. OpenCL/GL accelerator
  prebuilt sha256 `6ec1f5d515960dba1d022df8f966b34a1bfed44cc3981ca8aaa5de921f6d2d34`.

## Forms

| form | file bytes | delegation | ids equal /48 | KV chain (`manifest_chain`) | decode ms/token | prefill ms (len 32) p0 / p1 / p2 |
|---|---|---|---|---|---|---|
| cpu fp32 (the floor) | 2,384,789,652 | XNNPACK 3880 of 3994 nodes, 58 partitions | 48 (pass) | 16/16 for each prompt (pass) | 39.468 | 801.6 / 62.1 / 66.8 |
| cpu int8 + lm_head int8 | 1,231,082,068 | XNNPACK 3880 of 3994 nodes, 58 partitions | 48 (pass) | not checked against the fp32 bar | 11.665 | 42.2 / 37.3 / 36.9 |
| opencl int8 + lm_head int8 | 1,231,082,068 | not run | — | — | — | — |

- Decode ms/token is the median of the three per-prompt medians, as on the Mac. Each prompt has 15
  decode steps. The per-prompt medians were:
  - cpu fp32: 39.138, 39.468, 49.870
  - cpu int8 + lm_head int8: 11.665, 12.202, 11.651
- The p0 prefill of cpu fp32, 801.6 ms, and its first decode step, 813.3 ms, are the first calls of
  the process. The other prompts' prefills took 62 to 67 ms. The cause of the slow first calls was
  not measured.
- OpenCL int8 + lm_head int8 was not run because of the memory rule. The rule asks for at least six
  times the file size plus 1 GiB free:
  - The script took the file size as 1,203,416 kB (`du -k`), so it needed 8,269,072 kB. The exact
    bytes give 8,261,948 kB.
  - MemAvailable was 6,553,616 kB at the start of the job, and at most 7,736,860 kB at any read.
  - The fp32 file (15,021,953 kB needed) and the int4 file (7,246,153 kB needed) do not fit either.
  - So no form ran on this phone's GPU, and there is no OpenCL delegation report.
- `checks/speed_floor.py` exits 0: cpu int8 + lm_head int8, 11.665 ms/token, beats the floor, cpu
  fp32 at 39.468 ms/token (`speed_floor_s26.txt`).

## Phone CPU against the Mac CPU

- cpu fp32, same file on both machines:
  - The generated ids are identical.
  - 1 of the 48 chain logit files is byte-identical; the largest max abs difference is 9.20e-05
    (p2, step 6).
  - Against the reference, the largest chain max abs is 1.04e-04, within the bar of 0.05.
  - The Mac's padded and unpadded chains are byte-identical to each other, so the phone has the same
    distance to both.
- cpu int8 + lm_head int8, same file on both machines:
  - The generated ids are identical.
  - 25 of the 48 chain logit files are byte-identical.
  - The largest max abs difference is 1.47 (p0, step 14); p2 stays within 3.8e-06.

## Files

- `kv_chain_s26_cpu_fp32_p0.txt`, `kv_chain_s26_cpu_fp32_p1.txt`, `kv_chain_s26_cpu_fp32_p2.txt`
- `text_match_s26_cpu_fp32.txt`
- `text_match_s26_cpu_int8_lmh8.txt`
- `timing_s26.json`
- `speed_floor_s26.txt`

## Attached unit (SM-S942Q), OpenCL

Iteration 5. A second Galaxy S26 ran the OpenCL half of row 14. This unit had more free memory than
the lab unit, so both int8 forms passed the memory rule. The lab sections above are unchanged.

### Phone

- Galaxy S26 (SM-S942Q), attached over USB to the Mac and driven with adb.
  SoC SM8850 (`ro.soc.model`), platform `canoe`, Adreno GPU (`ro.hardware.egl`).
- Android 16 (SDK 36), build `BP4A.251205.006.S942QOPS1AZH9`.
- MemTotal 11,389,756 kB; swap 12,582,908 kB.
- MemAvailable:

  | read | MemAvailable (kB) |
  |---|---|
  | slot start | 7,334,792 |
  | before cpu fp32 | 7,400,744 |
  | before opencl int8 | 7,853,184 |
  | before opencl int8 + lm_head int8 | 7,734,692 |
  | before cpu int8 + lm_head int8 | 8,474,792 |
  | slot end | 8,455,168 |

- Uptime 97,951 s at the start and 98,316 s at the end. Every read after a run was higher than the
  read before it: the phone did not restart.
- The screen was on and not locked.
- No other process matching `litert`, `bench`, `qwen`, `_main` or `tflite` was running.
- CPU frequency ceilings were not read on this unit.

### Thermal

The phone had been in use by another job just before the slot. A first slot, about two minutes
earlier, was given back without running anything. Its readings were:

- Thermal Status 4;
- the AP sensor at 50.8 °C and the SKIN sensor at 45.6 °C (SKIN status 4);
- 54.4 °C on the hottest zone.

In this slot, nothing ran for the first 125 s. The thermal state was read every 30 s, until the AP
and SKIN sensors were at or below 45 °C and every sensor's status was at most 1.
Temperatures are the HAL values from `dumpsys thermalservice`. The zone maximum leaves out the
`trip` zones.

| read | Thermal Status | AP °C | SKIN °C | hottest zone °C |
|---|---|---|---|---|
| slot start | 2 | 43.8 | 41.8 | 48.2 |
| after 125 s (runs may start) | 1 | 41.3 | 39.8 | 45.5 |
| before cpu fp32 | 1 | 42.3 | 39.5 | 56.0 |
| before opencl int8 | 1 | 47.1 | 40.7 | 55.2 |
| before opencl int8 + lm_head int8 | 1 | 48.7 | 41.6 | 62.5 |
| before cpu int8 + lm_head int8 | 1 | 45.8 | 41.2 | 50.2 |
| slot end | 1 | 44.6 | 41.1 | 49.4 |

The AP sensor was above 45 °C at the start of the last three runs. The times below were taken
under these temperatures.

### Route

- One device slot: 02:00:14 to 02:06:23 (JST), 6 min 9 s. Of this:
  - 125 s of cooling;
  - 88 s of push;
  - 147 s of the four runs, from the first run's start to the last run's end;
  - 6 s of pull and delete.
- The files were pushed to `/data/local/tmp/tapi_qwen3`:
  - the same binary and OpenCL/GL accelerator as the lab run (same sha256);
  - the prompt ids;
  - three model files, saved again on the Mac with `--serialize_only --prefill_len=32`.
- The model files:

  | file | bytes | push |
  |---|---|---|
  | fp32 | 2,384,789,652 | 36.3 s |
  | int8 decoder weights, lm_head fp32 (`--weights_quant=int8`) | 1,071,852,948 | 22.2 s |
  | int8 + int8 lm_head copy | 1,231,082,068 | 26.6 s |

  The fp32 and int8 + int8 lm_head files have the same sizes as the lab run's.
- The directory was deleted at the end of the slot.
- Each run, from that directory:

```
LD_LIBRARY_PATH=. timeout -k 10 <s> ./qwen3_main --load_tflite=<file> --prefill_len=32 \
  --ids_file=ids_all.txt --greedy=16 --dump_chain --dump_dir=<dir> --out_ids=<dir>/ids.json \
  --time --timing_out=<json> --delegation_report=true <accelerator flags> [--weights_quant=int8 \
  [--weights_quant_lm_head]] --form="<form name>"
```

  - The OpenCL forms add `--accelerator=gpu --gpu_precision=fp32 --gpu_buffer_storage=buffer`.
  - The time limit was 480 s, and 300 s for cpu int8 + lm_head int8.
- Memory rule for the OpenCL forms on this unit: MemAvailable, read right before the run, at least
  five times the file size in kB plus 524,288 kB (512 MiB):

  | form | needs (kB) | MemAvailable before the run (kB) |
  |---|---|---|
  | int8 | 5,757,948 | 7,885,204 |
  | int8 + lm_head int8 | 6,535,433 | 7,856,952 |

  The fp32 file (needs 12,168,773 kB, more than MemTotal) was not run on the GPU.

### Forms

| form | file bytes | delegation | ids equal /48 | KV chain (`manifest_chain`) | decode ms/token | prefill ms (len 32) p0 / p1 / p2 | start AP °C |
|---|---|---|---|---|---|---|---|
| cpu fp32 (the floor) | 2,384,789,652 | XNNPACK 3880 of 3994 nodes, 58 partitions | 48 (pass) | 16/16 for each prompt (pass) | 37.335 | 63.4 / 53.7 / 55.3 | 42.3 |
| opencl int8 (lm_head fp32) | 1,071,852,948 | GPU 3994 of 3994 nodes, 2 partitions, fully accelerated | 48 (pass) | 0/16 for each prompt (fp32 bar; recorded) | 3.297 | 3.2 / 2.4 / 2.2 | 47.1 |
| opencl int8 + lm_head int8 | 1,231,082,068 | GPU 3994 of 3994 nodes, 2 partitions, fully accelerated | 48 (pass) | 0/16 for each prompt (fp32 bar; recorded) | 3.314 | 2.4 / 2.4 / 2.3 | 48.7 |
| cpu int8 + lm_head int8 | 1,231,082,068 | XNNPACK 3880 of 3994 nodes, 58 partitions | 48 (pass) | not checked against the fp32 bar | 12.729 | 60.8 / 48.4 / 52.7 | 45.8 |

- Decode ms/token is the median of the three per-prompt medians. The per-prompt medians:

  | form | p0 | p1 | p2 |
  |---|---|---|---|
  | cpu fp32 | 38.240 | 37.319 | 37.335 |
  | opencl int8 | 3.297 | 3.236 | 3.517 |
  | opencl int8 + lm_head int8 | 3.248 | 3.609 | 3.314 |
  | cpu int8 + lm_head int8 | 12.729 | 12.702 | 12.732 |

- What the time measures. Each number is the wall time of one `runner.Run` call; reading the logits
  is outside it. On OpenCL, the prefill of 32 positions took less time than one decode step.
  Whether `Run` on OpenCL returns only after the GPU has finished was not checked. So the OpenCL
  numbers may leave out GPU time that falls into the logits read: unverified.
- The KV chain of the OpenCL forms is checked against the fp32 bar (max abs 0.05, cosine 0.9999) as
  a record. Against the reference, the two forms reach:

  | form | largest max abs | smallest cosine |
  |---|---|---|
  | opencl int8 | 1.10 | 0.99885 |
  | opencl int8 + lm_head int8 | 1.08 | 0.99880 |

  The argmax agrees on all 48 rows for both forms. Their bar is the text match, 44 of 48 or more.
- OpenCL setup time:
  - The delegation report came 24 s (int8) and 29 s (int8 + lm_head int8) after the start.
  - The runner was ready 59 s and 55 s after the start.
  - The program compiles the model twice: once for the report, once for the runner.
- Memory during the OpenCL runs, read every 3 s:

  | form | first read (kB) | lowest read (kB) | drop | drop / file kB |
  |---|---|---|---|---|
  | int8 | 7,853,188 | 687,616 | 7,165,572 kB | 6.85 |
  | int8 + lm_head int8 | 7,730,216 | 1,841,156 | 5,889,060 kB | 4.90 |

  - The lowest reads came near the end of a compile:
    - int8: 1 s before the runner was ready;
    - int8 + lm_head int8: 2 s before the delegation report.
  - MemAvailable rose again after each compile.
  - A read every 3 s can miss a lower value in between.
  - The int8 file is the smaller of the two, but its run took more memory.
  - Five times the file plus 512 MiB was below the int8 run's drop.
- No operation fell back to the CPU in the OpenCL runs (all 3994 nodes delegated). No compile or
  runtime error was printed; the NPU warning of the lab run (`kLiteRtStatusErrorNotFound`) appears
  here too.
- `checks/speed_floor.py` exits 0: opencl int8, 3.297 ms/token, beats the floor, cpu fp32 at
  37.335 ms/token (`speed_floor_s26_attached.txt`).

### Against the other runs

- OpenCL against the Mac's Metal, iteration 3: the generated ids are identical, 48 of 48.
  - opencl int8 matches metal int8.
  - opencl int8 + lm_head int8 matches metal int8 + lm_head int8.
  - The Metal runs used the prompt-length prefill; their chain dumps were not kept, so only the ids
    are compared.
- OpenCL against this phone's CPU: the generated ids are identical in every pairing. No chain file
  is byte-identical. The largest max abs differences:
  - opencl int8 against cpu fp32: 1.10;
  - opencl int8 + lm_head int8 against cpu int8 + lm_head int8 (same file): 2.82;
  - opencl int8 + lm_head int8 against opencl int8: 0.195.
- On the Mac CPU, int8 with an fp32 lm_head gave 41 of 48 (iteration 3). On this phone's GPU it gives
  48 of 48, as on Metal.
- This unit's CPU against the lab unit's CPU, same files:
  - cpu fp32: ids identical, 48 of 48 chain files byte-identical;
  - cpu int8 + lm_head int8: ids identical, 48 of 48 chain files byte-identical.
  - The times differ: fp32 37.335 against 39.468 ms/token; int8 + lm_head int8 12.729 against
    11.665 ms/token.
  - The lab unit's first prefill (801.6 ms) did not repeat here (63.4 ms).

### Files

- `kv_chain_s26_attached_cpu_fp32_p0.txt`, `kv_chain_s26_attached_cpu_fp32_p1.txt`,
  `kv_chain_s26_attached_cpu_fp32_p2.txt`
- `kv_chain_s26_attached_opencl_int8_p0.txt`, `kv_chain_s26_attached_opencl_int8_p1.txt`,
  `kv_chain_s26_attached_opencl_int8_p2.txt`
- `kv_chain_s26_attached_opencl_int8_lm_head_int8_p0.txt`,
  `kv_chain_s26_attached_opencl_int8_lm_head_int8_p1.txt`,
  `kv_chain_s26_attached_opencl_int8_lm_head_int8_p2.txt`
- `text_match_s26_attached_cpu_fp32.txt`, `text_match_s26_attached_opencl_int8.txt`,
  `text_match_s26_attached_opencl_int8_lm_head_int8.txt`,
  `text_match_s26_attached_cpu_int8_lm_head_int8.txt`
- `timing_s26_attached.json`
- `speed_floor_s26_attached.txt`
