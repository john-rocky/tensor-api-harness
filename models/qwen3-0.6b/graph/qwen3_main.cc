// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B host loop: builds the prefill and decode signatures for each
// prompt (prefill length = the prompt length), runs prefill, then greedy
// decode with the KV caches rebound from the previous call's outputs by
// handle. Token ids come from a file; the tokenizer stays outside.
//
//   bazel build //tensor/examples/qwen3-0.6b:qwen3_main
//   qwen3_main --weights=<model.safetensors> --ids_file=<ids.txt> \
//     --accelerator=cpu [--dump_blocks --dump_dir=<dir>] \
//     [--greedy=16 --out_ids=<ids.json>]
//
// ids file: one prompt per line, "<id> <token id> <token id> ...".
// --dump_blocks adds emb, block_00..block_27 and final_norm to the prefill
// signature's outputs and writes them, with logits_last.f32 (the last
// position's logits), to <dump_dir>/<id>/ as raw fp32.
// --greedy=N writes <out_ids> as {"prompts": [{"id", "ids"}]}: the prompt
// ids followed by N generated ids (1 from prefill, N - 1 decode steps).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"
#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights, "",
          "model.safetensors path (empty = synthetic weights under the real "
          "tensor names)");
ABSL_FLAG(std::string, ids_file, "",
          "Prompt ids, one prompt per line: <id> <token id> ...");
ABSL_FLAG(int, max_seq, 64, "Fixed KV cache length");
ABSL_FLAG(int, greedy, 0,
          "Generated tokens per prompt (0 = prefill only)");
ABSL_FLAG(std::string, out_ids, "", "Where to write the ids JSON (--greedy)");
ABSL_FLAG(bool, dump_blocks, false,
          "Add the per-block outputs to the prefill signature and dump them");
ABSL_FLAG(std::string, dump_dir, "", "Dump directory (--dump_blocks)");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "fp32",
          "fp16|fp32 GPU calculation precision (the delegate default is fp16)");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d GPU tensor storage");
ABSL_FLAG(std::string, tflite_path, "/tmp/qwen3_0_6b.tflite",
          "Where to write the serialized model");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;
namespace qwen3 = ::litert::tensor::examples::qwen3;

constexpr float kMaskFloor = -30000.0f;  // fp16-safe

struct Prompt {
  std::string id;
  std::vector<int32_t> ids;
};

absl::StatusOr<std::vector<Prompt>> ReadIdsFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) return absl::NotFoundError("ids_file: " + path);
  std::vector<Prompt> prompts;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    Prompt p;
    if (!(ss >> p.id)) continue;
    int32_t v;
    while (ss >> v) p.ids.push_back(v);
    if (p.ids.empty()) {
      return absl::InvalidArgumentError("no ids for prompt " + p.id);
    }
    prompts.push_back(std::move(p));
  }
  if (prompts.empty()) return absl::InvalidArgumentError("empty ids_file");
  return prompts;
}

std::string ShapeStr(const std::vector<int>& s) {
  return absl::StrCat("[", absl::StrJoin(s, ", "), "]");
}

void BuildRopeCosSin(int pos, int seq, int head_dim, float base,
                     std::vector<float>& cos_out, std::vector<float>& sin_out) {
  cos_out.resize(static_cast<size_t>(seq) * head_dim);
  sin_out.resize(static_cast<size_t>(seq) * head_dim);
  int half = head_dim / 2;
  for (int s = 0; s < seq; ++s) {
    for (int i = 0; i < half; ++i) {
      float inv_freq =
          1.0f / std::pow(base, static_cast<float>(2 * i) / head_dim);
      float angle = static_cast<float>(pos + s) * inv_freq;
      cos_out[s * head_dim + i] = cos_out[s * head_dim + half + i] =
          std::cos(angle);
      sin_out[s * head_dim + i] = sin_out[s * head_dim + half + i] =
          std::sin(angle);
    }
  }
}

// Row s (absolute position valid_len + s) sees positions 0..valid_len + s.
std::vector<float> BuildMask(int seq_q, int valid_len, int max_seq) {
  std::vector<float> mask(static_cast<size_t>(seq_q) * max_seq, kMaskFloor);
  for (int s = 0; s < seq_q; ++s) {
    int limit = std::min(valid_len + s + 1, max_seq);
    for (int j = 0; j < limit; ++j) mask[s * max_seq + j] = 0.0f;
  }
  return mask;
}

absl::Status ReadFloats(LitertDynamicRunner& runner, const std::string& sig,
                        const std::string& name, std::vector<float>& out) {
  auto t = runner.GetOutput(sig, name);
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  out.assign(data, data + lock.size() / sizeof(float));
  return absl::OkStatus();
}

absl::Status WriteRaw(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return absl::InternalError("cannot write " + path);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
  return absl::OkStatus();
}

int32_t ArgMaxRow(const float* row, int n) {
  return static_cast<int32_t>(std::max_element(row, row + n) - row);
}

absl::StatusOr<::litert::Options> MakeOptions() {
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const std::string accel = absl::GetFlag(FLAGS_accelerator);
  if (accel != "cpu" && accel != "gpu") {
    return absl::InvalidArgumentError("--accelerator must be cpu|gpu");
  }
  const bool use_gpu = accel == "gpu";
  options->SetHardwareAccelerators(use_gpu ? ::litert::HwAccelerators::kGpu
                                           : ::litert::HwAccelerators::kCpu);
  if (use_gpu) {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      const std::string precision = absl::GetFlag(FLAGS_gpu_precision);
      if (precision == "fp32") {
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      } else if (precision != "fp16") {
        return absl::InvalidArgumentError("--gpu_precision must be fp16|fp32");
      }
      const std::string storage = absl::GetFlag(FLAGS_gpu_buffer_storage);
      if (storage == "buffer") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kBuffer);
      } else if (storage == "texture2d") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kTexture2D);
      } else if (storage != "default") {
        return absl::InvalidArgumentError(
            "--gpu_buffer_storage must be default|buffer|texture2d");
      }
      std::cout << "gpu precision: " << precision << ", storage: " << storage
                << std::endl;
    }
  }
  return std::move(*options);
}

// Builds both signatures for one prompt length, runs prefill (+ dumps) and
// the greedy loop. Appends the generated ids to `generated`.
absl::Status RunPrompt(::litert::Environment& env, qwen3::Qwen3Config config,
                       const qwen3::WeightMap& weights, const Prompt& prompt,
                       int greedy, std::vector<int32_t>& generated) {
  const int seq = static_cast<int>(prompt.ids.size());
  config.prefill_len = seq;
  if (seq + greedy > config.max_seq) {
    return absl::InvalidArgumentError(absl::StrCat(
        prompt.id, ": prompt ", seq, " + greedy ", greedy, " > max_seq ",
        config.max_seq));
  }
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);

  qwen3::Qwen3Inputs prefill_in = qwen3::MakeQwen3Inputs(config, false);
  qwen3::Qwen3Outputs prefill_out =
      qwen3::BuildQwen3(config, prefill_in, weights, false);
  qwen3::Qwen3Config decode_config = config;
  decode_config.dump_blocks = false;
  qwen3::Qwen3Inputs decode_in = qwen3::MakeQwen3Inputs(decode_config, true);
  qwen3::Qwen3Outputs decode_out =
      qwen3::BuildQwen3(decode_config, decode_in, weights, true);

  ModelFactory factory;
  {
    std::vector<TensorHandle> ins, outs;
    for (auto& t : prefill_in.AsList()) ins.push_back(t);
    for (auto& t : prefill_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "prefill");
    if (!status.ok()) return status;
  }
  {
    std::vector<TensorHandle> ins, outs;
    for (auto& t : decode_in.AsList()) ins.push_back(t);
    for (auto& t : decode_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "decode");
    if (!status.ok()) return status;
  }
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::error_code ec;
  auto tflite_bytes = std::filesystem::file_size(tflite_path, ec);
  std::cout << prompt.id << ": serialized " << tflite_path << " ("
            << (ec ? 0 : tflite_bytes) << " bytes), prefill_len " << seq
            << ", max_seq " << config.max_seq << std::endl;

  auto options_or = MakeOptions();
  if (!options_or.ok()) return options_or.status();
  auto runner_or = LitertDynamicRunner::Create(env, tflite_path, *options_or);
  if (!runner_or.ok()) return runner_or.status();
  auto runner = std::move(*runner_or);

  const int hd = config.head_dim;
  const int vocab = config.vocab_size;
  std::vector<float> cos_v, sin_v;
  auto set_step = [&](const std::string& sig, const std::vector<int32_t>& ids,
                      int pos) -> absl::Status {
    const int n = static_cast<int>(ids.size());
    auto st = runner.SetInput(
        sig, "input_ids",
        Create("input_ids", Type::kI32, {n}, std::vector<int32_t>(ids)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "cache_position",
                         Create("cache_position", Type::kI32, {4},
                                std::vector<int32_t>{0, 0, pos, 0}));
    if (!st.ok()) return st;
    BuildRopeCosSin(pos, n, hd, config.rope_base, cos_v, sin_v);
    st = runner.SetInput(sig, "rope_cos",
                         Create("rope_cos", Type::kFP32, {1, 1, n, hd},
                                std::vector<float>(cos_v)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "rope_sin",
                         Create("rope_sin", Type::kFP32, {1, 1, n, hd},
                                std::vector<float>(sin_v)));
    if (!st.ok()) return st;
    return runner.SetInput(sig, "attention_mask",
                           Create("attention_mask", Type::kFP32,
                                  {1, 1, n, config.max_seq},
                                  BuildMask(n, pos, config.max_seq)));
  };
  auto print_outputs = [&](const std::string& sig,
                           const std::vector<qwen3::TfTensor>& outs)
      -> absl::Status {
    // The graph shape next to the element count the runtime returned.
    for (const auto& t : outs) {
      std::string name(t.GetName());
      if (name.rfind("key_cache_", 0) == 0 && name != "key_cache_0_new") {
        continue;
      }
      if (name.rfind("value_cache_", 0) == 0 && name != "value_cache_0_new") {
        continue;
      }
      std::vector<float> v;
      auto st = ReadFloats(runner, sig, name, v);
      if (!st.ok()) return st;
      std::vector<int> shape(t.GetShape().begin(), t.GetShape().end());
      size_t expect = 1;
      for (int d : shape) expect *= static_cast<size_t>(d);
      std::cout << prompt.id << " " << sig << " output " << name << " "
                << ShapeStr(shape) << " elements " << v.size()
                << (v.size() == expect ? "" : " SHAPE MISMATCH") << std::endl;
      if (v.size() != expect) {
        return absl::InternalError("output size mismatch: " + name);
      }
    }
    return absl::OkStatus();
  };

  // --- Prefill ---
  auto st = set_step("prefill", prompt.ids, 0);
  if (!st.ok()) return st;
  const size_t cache_n =
      static_cast<size_t>(config.n_kv_groups) * config.max_seq * hd;
  for (int i = 0; i < config.n_layers; ++i) {
    for (const char* kind : {"key_cache_", "value_cache_"}) {
      std::string name = absl::StrCat(kind, i);
      st = runner.SetInput(
          "prefill", name,
          Create(name.c_str(), Type::kFP32,
                 {1, config.n_kv_groups, config.max_seq, hd},
                 std::vector<float>(cache_n, 0.0f)));
      if (!st.ok()) return st;
    }
  }
  st = runner.Run("prefill");
  if (!st.ok()) return st;
  st = print_outputs("prefill", prefill_out.AsList());
  if (!st.ok()) return st;

  std::vector<float> logits;
  st = ReadFloats(runner, "prefill", "logits", logits);
  if (!st.ok()) return st;
  const float* last = logits.data() + static_cast<size_t>(seq - 1) * vocab;

  if (config.dump_blocks) {
    const std::string dir =
        absl::StrCat(absl::GetFlag(FLAGS_dump_dir), "/", prompt.id);
    std::filesystem::create_directories(dir, ec);
    for (const auto& t : prefill_out.dumps) {
      std::vector<float> v;
      st = ReadFloats(runner, "prefill", std::string(t.GetName()), v);
      if (!st.ok()) return st;
      st = WriteRaw(absl::StrCat(dir, "/", t.GetName(), ".f32"), v);
      if (!st.ok()) return st;
    }
    st = WriteRaw(dir + "/logits_last.f32", std::vector<float>(last, last + vocab));
    if (!st.ok()) return st;
    std::ofstream am(dir + "/argmax.json");
    am << "{\"argmax\": [";
    for (int s = 0; s < seq; ++s) {
      am << (s ? ", " : "")
         << ArgMaxRow(logits.data() + static_cast<size_t>(s) * vocab, vocab);
    }
    am << "]}\n";
    std::cout << prompt.id << ": dumped " << prefill_out.dumps.size() + 1
              << " tensors to " << dir << std::endl;
  }

  if (greedy <= 0) return absl::OkStatus();

  // --- Greedy decode: token 1 from prefill, then greedy - 1 decode steps ---
  auto rebind = [&](const std::string& from_sig) -> absl::Status {
    for (int i = 0; i < config.n_layers; ++i) {
      for (const char* kind : {"key_cache_", "value_cache_"}) {
        auto t = runner.GetOutput(from_sig, absl::StrCat(kind, i, "_new"));
        if (!t.ok()) return t.status();
        auto s = runner.SetInput("decode", absl::StrCat(kind, i), *t);
        if (!s.ok()) return s;
      }
    }
    return absl::OkStatus();
  };
  int32_t tok = ArgMaxRow(last, vocab);
  generated.push_back(tok);
  st = rebind("prefill");
  if (!st.ok()) return st;
  for (int step = 1; step < greedy; ++step) {
    const int pos = seq + step - 1;
    st = set_step("decode", {tok}, pos);
    if (!st.ok()) return st;
    st = runner.Run("decode");
    if (!st.ok()) return st;
    if (step == 1) {
      st = print_outputs("decode", decode_out.AsList());
      if (!st.ok()) return st;
    }
    st = ReadFloats(runner, "decode", "logits", logits);
    if (!st.ok()) return st;
    tok = ArgMaxRow(logits.data(), vocab);
    generated.push_back(tok);
    st = rebind("decode");
    if (!st.ok()) return st;
  }
  std::cout << prompt.id << ": generated " << absl::StrJoin(generated, " ")
            << std::endl;
  return absl::OkStatus();
}

absl::Status Run() {
  qwen3::Qwen3Config config;
  config.max_seq = absl::GetFlag(FLAGS_max_seq);
  config.dump_blocks = absl::GetFlag(FLAGS_dump_blocks);
  const int greedy = absl::GetFlag(FLAGS_greedy);
  if (config.dump_blocks && absl::GetFlag(FLAGS_dump_dir).empty()) {
    return absl::InvalidArgumentError("--dump_blocks needs --dump_dir");
  }
  if (greedy > 0 && absl::GetFlag(FLAGS_out_ids).empty()) {
    return absl::InvalidArgumentError("--greedy needs --out_ids");
  }
  auto prompts_or = ReadIdsFile(absl::GetFlag(FLAGS_ids_file));
  if (!prompts_or.ok()) return prompts_or.status();

  qwen3::WeightMap weights;
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    weights = qwen3::MakeSyntheticWeights(config, /*seed=*/42);
    std::cout << "weights: synthetic (seed 42), " << weights.size()
              << " tensors" << std::endl;
  } else {
    auto weights_or = qwen3::LoadCheckpointWeights(config, weights_path);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    std::cout << "weights: " << weights_path << " (" << weights.size()
              << " tensors, bf16->fp32, lm_head tied to "
                 "model.embed_tokens.weight)"
              << std::endl;
  }

  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");

  std::ostringstream json;
  json << "{\"prompts\": [";
  bool first = true;
  for (const Prompt& prompt : *prompts_or) {
    std::vector<int32_t> generated;
    auto st = RunPrompt(*env, config, weights, prompt, greedy, generated);
    if (!st.ok()) return st;
    std::vector<int32_t> all = prompt.ids;
    all.insert(all.end(), generated.begin(), generated.end());
    json << (first ? "" : ", ") << "{\"id\": \"" << prompt.id
         << "\", \"ids\": [" << absl::StrJoin(all, ", ") << "]}";
    first = false;
  }
  json << "]}\n";
  if (greedy > 0) {
    std::ofstream out(absl::GetFlag(FLAGS_out_ids));
    if (!out) return absl::InternalError("cannot write --out_ids");
    out << json.str();
    std::cout << "wrote " << absl::GetFlag(FLAGS_out_ids) << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = Run();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  std::cout << "DONE qwen3_main" << std::endl;
  return 0;
}
