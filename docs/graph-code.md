# The graph code

The skill table is [`skill_table.md`](../skill_table.md). An iteration is one run of the coding agent on one instruction file.

A model's graphs are authored in C++ as a LiteRT `tensor/examples/` directory.
`models/<name>/graph/` is such a directory; `tools/overlay.sh <LiteRT checkout> <name>` links it
into the checkout and prints the Bazel build line and the run line. Reference implementations in
the same form:

- LiteRT's own `tensor/examples/gemma3`: prefill and decode signatures, the KV cache as signature
  input and output, safetensors loading, the Metal accelerator prebuilt on macOS.
- Three merged litert-samples implementations under `models/<family>/<model>/tensor_api/`:
  Qwen3-TTS ([#241](https://github.com/google-ai-edge/litert-samples/pull/241)), LLaDA-8B
  ([#242](https://github.com/google-ai-edge/litert-samples/pull/242)) and SAM 2.1 video
  ([#301](https://github.com/google-ai-edge/litert-samples/pull/301)), each with its verification
  suite against an independent reference.

## License

Apache 2.0. The graph code follows the LiteRT examples' layout so it can be moved upstream.
