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
