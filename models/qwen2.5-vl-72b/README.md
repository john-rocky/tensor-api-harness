# Qwen2.5-VL-72B-Instruct on LiteRT-LM: the decoder written with the Tensor API

The text decoder of [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) (80 layers)
is written straight from the bf16 checkpoint with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor),
int4, with the same inputs and outputs that litert-torch's `export_hf` gives a decoder whose embedder is a separate
graph. The vision tower goes through litert-torch at one fixed picture size (448×448). The embedder, decoder, vision
encoder and vision adapter go into one `.litertlm`, and the stock LiteRT-LM 0.19.0 (PyPI wheel) runs it on the CPU and
on the GPU without any executor change.

Why the Tensor API here: the converter quantizes after it has written the whole fp32 graph, and for 72 B parameters
that graph alone is about 290 GB. The writer below never holds it: the weights are quantized one tensor at a time.

## Files

| file | what it does |
|---|---|
| `tapi_pack.py` | reads the checkpoint one tensor at a time and writes `weights.bin` + `index.tsv` + `config.tsv`: FC weights int4 blockwise-128 (symmetric min-max, scales floored at 2^-14), lm_head int8 per channel, norms and biases fp32. The quantization calls ai-edge-quantizer's own functions, so the bytes equal the converter path's. |
| `graph/qwen2_writer.cc`, `graph/BUILD` | the Tensor API graph: `prefill_<N>` and `decode` signatures, KV cache in and out per layer (V stored transposed), RoPE from `input_pos`, grouped-query attention over the whole cache with the runtime's mask, `ModelFactory::Save` with the weights mapped from `weights.bin`. |
| `embedder_export.py` | the token table as its own graph (`token_ids [1,1] → [1,1,8192]`). |
| `qwen25vl_vision.py` | the vision tower rewritten for one fixed size: windows batched by reshape / transpose (no gather), the patch Conv3d as a Conv2d, the 2×2 merge as four strided slices, RoPE tables from HF's own rotary module; `selftest` / `ref` / `export` / `check`. |
| `qwen72_bundle.py`, `qwen72_chat_template.jinja` | the `.litertlm`: sections, stop tokens, prompt templates, the jinja whose token ids equal the HF processor's on 8 conversations. |
| `lm_run.py` | one image + question through LiteRT-LM's Python API, with its BenchmarkInfo. |

## Steps

1. Pack the weights (147 GB of bf16 → 38.6 GB; about 1.6 h from a USB disk, peak memory about 11 GB):

   ```
   python tapi_pack.py SNAPSHOT PACK --fc int4 --block 128 --lm-head int8
   ```

2. Build the writer in a LiteRT checkout (tested at `dcb657e8`, LiteRT-LM 0.19.0's pin): link `graph/` into the
   checkout as `tensor/examples/qwen2_writer`, then

   ```
   bazel build -c opt //tensor/examples/qwen2_writer:qwen2_writer
   ```

3. Write the decoder (the prefill lengths and the KV cache length are flags):

   ```
   qwen2_writer --weights_dir=PACK --out=PACK/model.tflite --prefill=128,1024 --cache=4096
   ```

4. The embedder, then int8 per channel on its `EMBEDDING_LOOKUP` (ai-edge-quantizer):

   ```
   python embedder_export.py SNAPSHOT embedder_fp32.tflite --lightweight
   ```

5. The vision tower in fp32 (dynamic int8 moved the image embeddings by a relative L2 of about 0.2), then the rewrite
   that lets the GPU delegate run it ([`tfl_rank4.py`](https://github.com/john-rocky/hf-to-litertlm/tree/main/tools/gpu_graph)):

   ```
   python qwen25vl_vision.py export --src SNAPSHOT --g 448 --out-dir VISION
   python tfl_rank4.py rewrite VISION/vision_encoder_g448_fp32.tflite VISION/vision_encoder_g448_fp32_r4.tflite
   ```

6. The bundle:

   ```
   python qwen72_bundle.py --dec PACK --emb-path embedder_wi8.tflite --vis VISION --g 448 \
     --venc vision_encoder_g448_fp32_r4.tflite --tok SNAPSHOT/tokenizer.json --out-dir OUT \
     --name Qwen2.5-VL-72B-Instruct.litertlm --max-tokens 4096
   ```

7. Run:

   ```
   python lm_run.py OUT/Qwen2.5-VL-72B-Instruct.litertlm --backend gpu --image picture_448.png \
     --prompt "How many cats are in this picture?" --greedy
   ```

## How it was checked

- A tiny Qwen2 (2 layers, random weights): the writer's fp32 logits equal `export_hf`'s exactly, and its int4 logits
  equal `export_hf` + ai-edge-quantizer's with the same recipe exactly.
- The real weights cut to 2 layers, fp32: the same top-1 as HF at 160 of 160 positions, largest logit difference 4.2e-5.
- The full int4 model in LiteRT-LM on the Mac GPU: HF (the bf16 weights, fp32 arithmetic) picks the same token at
  98.8 % of the reply positions of four questions (three with a picture, one text only); the answers fit the
  pictures.
