// SPDX-License-Identifier: Apache-2.0
// Minimal reproduction: one odml.sdpa_transposed composite, run on the CPU and
// on the GPU accelerator from the same .tflite file.
//
// The model's only op is the odml.sdpa_transposed composite with the operand
// layout LiteRT-Torch exports for it (export_hf/experimental/composites/
// sdpa.py, scaled_dot_product_attention_transposed): the query already scaled
// by 1/sqrt(head_dim) [1, q_heads, seq, head_dim], the key cache
// [1, kv_heads, kv_len, head_dim], the value cache stored transposed
// [1, kv_heads, head_dim, kv_len], the mask [1, 1, seq, kv_len] and, with
// --param_tensor, the [1, 1, 1, 7] int32 param tensor; attributes k_ts_idx 2
// and v_ts_idx 3, no scale attribute. The decomposition packs the query heads
// of each kv head along the sequence axis (query head h reads kv head
// h / (q_heads / kv_heads)); it is what the CPU (XNNPACK) computes.
//
// The inputs come from a fixed seed (uniform: keys and values in [-1.5, 1.5]
// at every cache position, the query in [-1.5, 1.5] / sqrt(head_dim)). The
// tool compiles the file once for the CPU and once for the GPU, runs the CPU
// once and the GPU --gpu_runs times on the same inputs, and compares the
// outputs with each other and with a float64 reference computed on the host
// from the same inputs. It prints the largest CPU-vs-GPU difference, the
// first element over --threshold, the largest difference per query head and
// per query token, and the delegation metrics, and exits 0 (PASS: every GPU
// run within --threshold of the CPU), 2 (FAIL) or 1 (error).
//
//   sdpa_repro_main --seq=5 --tflite_path=/tmp/sdpa.tflite
//
// On macOS the GPU accelerator library is found through DYLD_LIBRARY_PATH
// (<LiteRT checkout>/litert/prebuilt/macos_arm64); see run.sh.
//
// Mask: query token s sees keys 0..position + s (--mask_pattern=causal) or
// every key (all). float = additive, 0 where visible and --mask_floor
// elsewhere; bool = true where visible (the decomposition then uses
// SelectV2(mask, scores, -10000)); none = three operands.
// --context=harness wraps the composite as the Qwen3-0.6B harness graph does:
// the query is scaled in the graph, the new key and value rows (values in the
// transposed layout) are written into the caches with DYNAMIC_UPDATE_SLICE at
// --position, the updated caches are also signature outputs, a bool mask is
// made in the graph from the float input (Equal(mask, 0)), and the output is
// transposed to [1, seq, q_heads, head_dim] and reshaped to
// [1, seq, q_heads * head_dim].

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

ABSL_FLAG(int, q_heads, 16, "Query heads");
ABSL_FLAG(int, kv_heads, 8, "Key/value heads (q_heads must be a multiple)");
ABSL_FLAG(int, head_dim, 128, "Head dimension");
ABSL_FLAG(int, kv_len, 64, "Cache length (keys per kv head)");
ABSL_FLAG(int, seq, 5, "Query tokens");
ABSL_FLAG(int, position, -1,
          "Cache position of the first query token: with the causal mask, "
          "query token s sees keys 0..position + s (-1 = 0 for seq > 1, 5 for "
          "seq 1)");
ABSL_FLAG(std::string, mask, "float", "float|bool|none: the mask operand");
ABSL_FLAG(std::string, mask_pattern, "causal",
          "causal|all: which keys each query token sees");
ABSL_FLAG(float, mask_floor, -30000.0f,
          "float mask: the additive value of a hidden key");
ABSL_FLAG(std::string, is_causal, "false",
          "false|true|absent: the is_causal attribute");
ABSL_FLAG(bool, param_tensor, false,
          "Add the [1, 1, 1, 7] int32 param tensor as the last operand");
ABSL_FLAG(std::string, param_values, "",
          "Seven comma-separated param tensor values (empty = {position, "
          "position + seq, position + seq, seq, 0, 0, 0})");
ABSL_FLAG(std::string, from_cache_update, "absent",
          "absent|true: the from_cache_update attribute");
ABSL_FLAG(std::string, context, "none", "none|harness (see the file comment)");
ABSL_FLAG(bool, composite, true,
          "false: emit the decomposition's ops directly, without the "
          "odml.sdpa_transposed composite around them (the control)");
ABSL_FLAG(std::string, gpu_precision, "fp32", "fp16|fp32");
ABSL_FLAG(std::string, gpu_buffer_storage, "buffer",
          "default|buffer|texture2d GPU tensor storage");
ABSL_FLAG(int, gpu_runs, 3, "GPU runs on the same inputs");
ABSL_FLAG(bool, run_cpu, true, "Run the CPU (XNNPACK) model");
ABSL_FLAG(bool, run_gpu, true, "Run the GPU model");
ABSL_FLAG(int, seed, 1, "Input seed");
ABSL_FLAG(double, threshold, 1e-3,
          "PASS line: the largest |CPU - GPU| over the output");
ABSL_FLAG(std::string, tflite_path, "/tmp/sdpa_transposed_repro.tflite",
          "Where to write the model");
ABSL_FLAG(std::string, dump_dir, "",
          "Write the inputs, the outputs and summary.json here (empty = no "
          "dump)");
ABSL_FLAG(std::string, log_level, "",
          "info|debug|verbose: the LiteRT logger's minimum severity (empty = "
          "the runtime default)");
ABSL_FLAG(std::string, label, "", "Case label for the output and the summary");

namespace litert::tensor::examples::sdpa_repro {
namespace {

using TfTensor = Tensor<TfLiteMixinTag>;

constexpr char kSignature[] = "sdpa";
constexpr size_t kSignatureIndex = 0;

struct Case {
  int q_heads, kv_heads, head_dim, kv_len, seq, position;
  std::string mask;          // float|bool|none
  std::string mask_pattern;  // causal|all
  float mask_floor;
  std::string is_causal;     // false|true|absent
  bool param_tensor;
  std::vector<int32_t> param_values;
  std::string from_cache_update;  // absent|true
  bool harness;                   // --context=harness
  bool composite;                 // false: the bare decomposition ops
  int group() const { return q_heads / kv_heads; }
  bool visible(int s, int j) const {
    return mask_pattern == "all" || j <= position + s;
  }
};

// Host inputs. q is what the composite reads (already scaled); with
// --context=harness the graph input is q_raw and the graph scales it.
struct Inputs {
  std::vector<float> q, q_raw, k_cache, v_cache, k_new, v_new;
  std::vector<float> mask_f;    // [seq, kv_len]
  std::vector<uint8_t> mask_b;  // [seq, kv_len]
};

// Deterministic LCG so the inputs are the same on every machine.
std::vector<float> Uniform(size_t count, uint32_t& state, float scale) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = scale * (static_cast<float>(state >> 8) /
                           static_cast<float>(1u << 24) * 2.0f -
                       1.0f);
  }
  return data;
}

TfTensor Const1(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

std::vector<uint8_t> CompositeAttributes(const Case& c) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("k_ts_idx", 2);
    fbb.Int("v_ts_idx", 3);
    if (c.is_causal != "absent") fbb.Bool("is_causal", c.is_causal == "true");
    if (c.from_cache_update == "true") fbb.Bool("from_cache_update", true);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// The decomposition (the same ops as the harness's
// SdpaTransposedGqaAttention): the g query heads of each kv head packed along
// the sequence axis (row gi * seq + s reads mask row s), scores = q k^T, the
// mask added (float) or selected (bool, hidden = -10000), softmax over the
// keys, then the transposed value cache.
TfTensor Decomposition(const TfTensor& dq, const TfTensor& dk, const TfTensor& dv,
              const TfTensor* dm, int G, int g, int seq, int D) {
  TfTensor qp = Reshape(dq, {1, G, g * seq, D});
  TfTensor scores = BatchMatMul(qp, dk, /*adj_x=*/false, /*adj_y=*/true);
  if (dm != nullptr) {
    TfTensor mp = g == 1 ? *dm : Tile(*dm, {1, 1, g, 1});  // [1,1,g*seq,kv]
    if (dm->GetType() == Type::kBOOL) {
      scores = SelectV2(mp, scores, Const1(-10000.0f));
    } else {
      scores = Add(scores, mp);
    }
  }
  TfTensor attn = Softmax(scores);
  TfTensor o = BatchMatMul(attn, dv, /*adj_x=*/false, /*adj_y=*/true);
  return Reshape(o, {1, G * g, seq, D});  // [1, q_heads, seq, head_dim]
}

TfTensor SdpaComposite(const Case& c, const TfTensor& q, const TfTensor& k,
                   const TfTensor& v, const TfTensor* mask,
                   const TfTensor* param) {
  const int G = c.kv_heads, g = c.group(), seq = c.seq, D = c.head_dim;
  if (!c.composite) return Decomposition(q, k, v, mask, G, g, seq, D);
  StableHLOCompositeOptions opts{.name = "odml.sdpa_transposed",
                                 .composite_attributes =
                                     CompositeAttributes(c)};
  if (mask != nullptr && param != nullptr) {
    return StableHLOComposite(
        opts,
        [=](TfTensor dq, TfTensor dk, TfTensor dv, TfTensor dm,
            TfTensor /*dp*/) { return Decomposition(dq, dk, dv, &dm, G, g, seq, D); },
        q, k, v, *mask, *param);
  }
  if (mask != nullptr) {
    return StableHLOComposite(
        opts,
        [=](TfTensor dq, TfTensor dk, TfTensor dv, TfTensor dm) {
          return Decomposition(dq, dk, dv, &dm, G, g, seq, D);
        },
        q, k, v, *mask);
  }
  if (param != nullptr) {
    return StableHLOComposite(
        opts,
        [=](TfTensor dq, TfTensor dk, TfTensor dv, TfTensor /*dp*/) {
          return Decomposition(dq, dk, dv, nullptr, G, g, seq, D);
        },
        q, k, v, *param);
  }
  return StableHLOComposite(
      opts,
      [=](TfTensor dq, TfTensor dk, TfTensor dv) {
        return Decomposition(dq, dk, dv, nullptr, G, g, seq, D);
      },
      q, k, v);
}

TfTensor GraphInput(const char* name, Type type, std::vector<int> shape) {
  return TfTensor({.name = name, .type = type, .shape = std::move(shape)});
}

// Builds the one-signature model and saves it. The signature's inputs and
// outputs are named after the tensors.
absl::Status BuildAndSave(const Case& c, const std::string& path) {
  const int Hq = c.q_heads, Hkv = c.kv_heads, D = c.head_dim, KV = c.kv_len,
            S = c.seq;
  std::vector<TensorHandle> ins, outs;
  TfTensor mask_in;
  TfTensor param_in;
  if (c.mask != "none") {
    const bool bool_input = c.mask == "bool" && !c.harness;
    mask_in = GraphInput("attention_mask", bool_input ? Type::kBOOL : Type::kFP32,
                    {1, 1, S, KV});
  }
  if (c.param_tensor) param_in = GraphInput("param_tensor", Type::kI32, {1, 1, 1, 7});

  TfTensor out;
  if (!c.harness) {
    TfTensor q = GraphInput("query", Type::kFP32, {1, Hq, S, D});
    TfTensor k = GraphInput("key_cache", Type::kFP32, {1, Hkv, KV, D});
    TfTensor v = GraphInput("value_cache", Type::kFP32, {1, Hkv, D, KV});
    ins = {q, k, v};
    if (c.mask != "none") ins.push_back(mask_in);
    if (c.param_tensor) ins.push_back(param_in);
    out = SdpaComposite(c,q, k, v, c.mask != "none" ? &mask_in : nullptr,
                    c.param_tensor ? &param_in : nullptr);
    out.SetName("output");
    outs = {out};
  } else {
    TfTensor q_raw = GraphInput("query", Type::kFP32, {1, Hq, S, D});
    TfTensor k_new = GraphInput("key_new", Type::kFP32, {1, Hkv, S, D});
    TfTensor v_new = GraphInput("value_new", Type::kFP32, {1, Hkv, D, S});
    TfTensor k_cache = GraphInput("key_cache", Type::kFP32, {1, Hkv, KV, D});
    TfTensor v_cache = GraphInput("value_cache", Type::kFP32, {1, Hkv, D, KV});
    TfTensor cache_pos = GraphInput("cache_position", Type::kI32, {4});
    TfTensor v_cache_pos = GraphInput("value_cache_position", Type::kI32, {4});
    ins = {q_raw, k_new, v_new, k_cache, v_cache, cache_pos, v_cache_pos};
    if (c.mask != "none") ins.push_back(mask_in);
    if (c.param_tensor) ins.push_back(param_in);
    TfTensor qs = Mul(q_raw, Const1(1.0f / std::sqrt(static_cast<float>(D))));
    TfTensor k_upd = DynamicUpdateSlice(k_cache, k_new, cache_pos);
    k_upd.SetName("key_cache_new");
    TfTensor v_upd = DynamicUpdateSlice(v_cache, v_new, v_cache_pos);
    v_upd.SetName("value_cache_new");
    TfTensor mask_op;
    if (c.mask == "bool") mask_op = Equal(mask_in, Const1(0.0f));
    if (c.mask == "float") mask_op = mask_in;
    out = SdpaComposite(c,qs, k_upd, v_upd, c.mask != "none" ? &mask_op : nullptr,
                    c.param_tensor ? &param_in : nullptr);
    out = Reshape(Transpose(out, {0, 2, 1, 3}), {1, S, Hq * D});
    out.SetName("output");
    outs = {out, k_upd, v_upd};
  }
  ModelFactory factory;
  auto status = factory.AddSignature(ins, outs, kSignature);
  if (!status.ok()) return status;
  return factory.Save(path);
}

Inputs MakeInputs(const Case& c, uint32_t seed) {
  const int Hq = c.q_heads, Hkv = c.kv_heads, D = c.head_dim, KV = c.kv_len,
            S = c.seq;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
  uint32_t state = seed;
  Inputs in;
  in.q_raw = Uniform(static_cast<size_t>(Hq) * S * D, state, 1.5f);
  in.q.resize(in.q_raw.size());
  for (size_t i = 0; i < in.q.size(); ++i) in.q[i] = in.q_raw[i] * inv_sqrt_d;
  in.k_cache = Uniform(static_cast<size_t>(Hkv) * KV * D, state, 1.5f);
  in.v_cache = Uniform(static_cast<size_t>(Hkv) * D * KV, state, 1.5f);
  in.k_new = Uniform(static_cast<size_t>(Hkv) * S * D, state, 1.5f);
  in.v_new = Uniform(static_cast<size_t>(Hkv) * D * S, state, 1.5f);
  in.mask_f.assign(static_cast<size_t>(S) * KV, c.mask_floor);
  in.mask_b.assign(static_cast<size_t>(S) * KV, 0);
  for (int s = 0; s < S; ++s) {
    for (int j = 0; j < KV; ++j) {
      if (c.visible(s, j)) {
        in.mask_f[s * KV + j] = 0.0f;
        in.mask_b[s * KV + j] = 1;
      }
    }
  }
  return in;
}

// The caches the composite reads: with --context=harness the new rows are
// written at the position first (key rows, value columns).
void EffectiveCaches(const Case& c, const Inputs& in, std::vector<float>& k,
                     std::vector<float>& v) {
  k = in.k_cache;
  v = in.v_cache;
  if (!c.harness) return;
  const int Hkv = c.kv_heads, D = c.head_dim, KV = c.kv_len, S = c.seq;
  for (int h = 0; h < Hkv; ++h) {
    for (int s = 0; s < S; ++s) {
      for (int d = 0; d < D; ++d) {
        k[(static_cast<size_t>(h) * KV + c.position + s) * D + d] =
            in.k_new[(static_cast<size_t>(h) * S + s) * D + d];
        v[(static_cast<size_t>(h) * D + d) * KV + c.position + s] =
            in.v_new[(static_cast<size_t>(h) * D + d) * S + s];
      }
    }
  }
}

// float64 reference of the decomposition, [q_heads, seq, head_dim].
std::vector<double> Reference(const Case& c, const Inputs& in) {
  const int Hq = c.q_heads, D = c.head_dim, KV = c.kv_len, S = c.seq,
            g = c.group();
  std::vector<float> k, v;
  EffectiveCaches(c, in, k, v);
  std::vector<double> out(static_cast<size_t>(Hq) * S * D, 0.0);
  std::vector<double> p(KV);
  for (int h = 0; h < Hq; ++h) {
    const int kh = h / g;
    for (int s = 0; s < S; ++s) {
      const float* qrow = &in.q[(static_cast<size_t>(h) * S + s) * D];
      double max_score = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < KV; ++j) {
        const float* krow = &k[(static_cast<size_t>(kh) * KV + j) * D];
        double score = 0.0;
        for (int d = 0; d < D; ++d) score += double(qrow[d]) * krow[d];
        if (c.mask == "float") score += in.mask_f[s * KV + j];
        if (c.mask == "bool" && !in.mask_b[s * KV + j]) score = -10000.0;
        p[j] = score;
        max_score = std::max(max_score, score);
      }
      double sum = 0.0;
      for (int j = 0; j < KV; ++j) {
        p[j] = std::exp(p[j] - max_score);
        sum += p[j];
      }
      for (int d = 0; d < D; ++d) {
        const float* vrow = &v[(static_cast<size_t>(kh) * D + d) * KV];
        double acc = 0.0;
        for (int j = 0; j < KV; ++j) acc += p[j] * vrow[j];
        out[(static_cast<size_t>(h) * S + s) * D + d] = acc / sum;
      }
    }
  }
  return out;
}

// Index of output element (h, s, d) in the output buffer.
size_t OutIndex(const Case& c, int h, int s, int d) {
  if (c.harness) {  // [1, seq, q_heads * head_dim]
    return (static_cast<size_t>(s) * c.q_heads + h) * c.head_dim + d;
  }
  return (static_cast<size_t>(h) * c.seq + s) * c.head_dim + d;
}

struct Diff {
  double max_abs = 0.0;
  size_t nan = 0;
  int bad_h = -1, bad_s = -1, bad_d = -1;  // first element over the line
  size_t over = 0;                          // elements over the line
  std::vector<double> per_head, per_token;
};

// |a - b| over (h, s, d); NaN anywhere counts as an infinite difference.
template <typename A, typename B>
Diff Compare(const Case& c, const A& a, const B& b, double line) {
  Diff r;
  r.per_head.assign(c.q_heads, 0.0);
  r.per_token.assign(c.seq, 0.0);
  for (int h = 0; h < c.q_heads; ++h) {
    for (int s = 0; s < c.seq; ++s) {
      for (int d = 0; d < c.head_dim; ++d) {
        const size_t i = OutIndex(c, h, s, d);
        double diff = std::abs(double(a[i]) - double(b[i]));
        if (std::isnan(diff)) {
          ++r.nan;
          diff = std::numeric_limits<double>::infinity();
        }
        r.max_abs = std::max(r.max_abs, diff);
        r.per_head[h] = std::max(r.per_head[h], diff);
        r.per_token[s] = std::max(r.per_token[s], diff);
        if (diff > line) {
          if (r.over == 0) {
            r.bad_h = h;
            r.bad_s = s;
            r.bad_d = d;
          }
          ++r.over;
        }
      }
    }
  }
  return r;
}

// A number for the output and the JSON summary (inf and nan quoted).
std::string Num(double v) {
  if (std::isnan(v)) return "\"nan\"";
  if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
  std::ostringstream ss;
  ss.precision(4);
  ss << v;
  return ss.str();
}

std::string NumList(const std::vector<double>& v) {
  std::vector<std::string> parts;
  for (double x : v) parts.push_back(Num(x));
  return absl::StrCat("[", absl::StrJoin(parts, ", "), "]");
}

template <typename T>
absl::Status WriteRaw(const std::string& path, const std::vector<T>& v) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return absl::InternalError("cannot write " + path);
  out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  return absl::OkStatus();
}

absl::StatusOr<::litert::Options> MakeOptions(bool gpu) {
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  options->SetHardwareAccelerators(gpu ? ::litert::HwAccelerators::kGpu
                                       : ::litert::HwAccelerators::kCpu);
  if (!gpu) return std::move(*options);
  auto gpu_options = options->GetGpuOptions();
  if (!gpu_options.HasValue()) return absl::InternalError("no GPU options");
  const std::string precision = absl::GetFlag(FLAGS_gpu_precision);
  if (precision == "fp32") {
    gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
  } else if (precision == "fp16") {
    gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp16);
  } else {
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
  return std::move(*options);
}

std::string HwName(LiteRtHwAcceleratorSet hw) {
  std::vector<std::string> names;
  if (hw & kLiteRtHwAcceleratorCpu) names.push_back("cpu");
  if (hw & kLiteRtHwAcceleratorGpu) names.push_back("gpu");
  if (hw & kLiteRtHwAcceleratorNpu) names.push_back("npu");
  if (names.empty()) names.push_back("none");
  return absl::StrJoin(names, "+");
}

// Compiles the model, prints the delegation metrics and returns them as JSON.
absl::StatusOr<::litert::CompiledModel> Compile(::litert::Environment& env,
                                                const std::string& path,
                                                bool gpu, std::string& json) {
  const char* tag = gpu ? "gpu" : "cpu";
  auto options = MakeOptions(gpu);
  if (!options.ok()) return options.status();
  auto compiled = ::litert::CompiledModel::Create(env, path, *options);
  if (!compiled) {
    return absl::InternalError(absl::StrCat(
        tag, ": CompiledModel::Create failed: ", compiled.Error().Message()));
  }
  auto metrics = compiled->GetDelegationMetrics();
  if (!metrics) {
    std::cout << tag << " delegation: unavailable ("
              << metrics.Error().Message() << ")" << std::endl;
    json = "null";
    return std::move(*compiled);
  }
  std::vector<std::string> accs;
  std::cout << tag << " delegation: total_nodes " << metrics->total_node_count;
  for (const auto& acc : metrics->accelerators) {
    std::cout << " | " << acc.accelerator_name << " (" << HwName(acc.hardware_type)
              << ") delegated " << acc.delegated_node_count << " partitions "
              << acc.partition_count;
    accs.push_back(absl::StrCat(
        "{\"name\": \"", acc.accelerator_name, "\", \"hw\": \"",
        HwName(acc.hardware_type), "\", \"delegated_nodes\": ",
        acc.delegated_node_count, ", \"partitions\": ", acc.partition_count,
        "}"));
  }
  std::cout << std::endl;
  json = absl::StrCat("{\"total_nodes\": ", metrics->total_node_count,
                      ", \"accelerators\": [", absl::StrJoin(accs, ", "),
                      "]}");
  return std::move(*compiled);
}

// Writes the host inputs into the compiled model's input buffers (by name)
// and runs it `runs` times, returning each run's output.
absl::StatusOr<std::vector<std::vector<float>>> RunModel(
    ::litert::CompiledModel& model, const Case& c, const Inputs& in, int runs) {
  auto names = model.GetSignatureInputNames(kSignatureIndex);
  if (!names) return absl::InternalError("GetSignatureInputNames failed");
  std::vector<::litert::TensorBuffer> inputs;
  const int P = c.position;
  for (const auto& name_view : *names) {
    const std::string name(name_view);
    auto buffer = model.CreateInputBuffer(kSignature, name);
    if (!buffer) {
      return absl::InternalError("CreateInputBuffer failed: " + name);
    }
    ::litert::Expected<void> written;
    if (name == "query") {
      written = buffer->Write<float>(
          absl::MakeConstSpan(c.harness ? in.q_raw : in.q));
    } else if (name == "key_cache") {
      written = buffer->Write<float>(absl::MakeConstSpan(in.k_cache));
    } else if (name == "value_cache") {
      written = buffer->Write<float>(absl::MakeConstSpan(in.v_cache));
    } else if (name == "key_new") {
      written = buffer->Write<float>(absl::MakeConstSpan(in.k_new));
    } else if (name == "value_new") {
      written = buffer->Write<float>(absl::MakeConstSpan(in.v_new));
    } else if (name == "cache_position") {
      const std::vector<int32_t> pos = {0, 0, P, 0};
      written = buffer->Write<int32_t>(absl::MakeConstSpan(pos));
    } else if (name == "value_cache_position") {
      const std::vector<int32_t> pos = {0, 0, 0, P};
      written = buffer->Write<int32_t>(absl::MakeConstSpan(pos));
    } else if (name == "attention_mask") {
      if (c.mask == "bool" && !c.harness) {
        written = buffer->Write<uint8_t>(absl::MakeConstSpan(in.mask_b));
      } else {
        written = buffer->Write<float>(absl::MakeConstSpan(in.mask_f));
      }
    } else if (name == "param_tensor") {
      written = buffer->Write<int32_t>(absl::MakeConstSpan(c.param_values));
    } else {
      return absl::InternalError("unexpected input: " + name);
    }
    if (!written) {
      return absl::InternalError(absl::StrCat(
          "write ", name, ": ", written.Error().Message()));
    }
    inputs.push_back(std::move(*buffer));
  }
  auto outputs = model.CreateOutputBuffers(kSignatureIndex);
  if (!outputs) return absl::InternalError("CreateOutputBuffers failed");
  auto out_names = model.GetSignatureOutputNames(kSignatureIndex);
  if (!out_names) return absl::InternalError("GetSignatureOutputNames failed");
  size_t out_index = 0;
  while (out_index < out_names->size() &&
         std::string((*out_names)[out_index]) != "output") {
    ++out_index;
  }
  if (out_index == out_names->size()) {
    return absl::InternalError("no output named \"output\"");
  }
  const size_t n = static_cast<size_t>(c.q_heads) * c.seq * c.head_dim;
  std::vector<std::vector<float>> results;
  for (int r = 0; r < runs; ++r) {
    auto ran = model.Run(kSignatureIndex, inputs, *outputs);
    if (!ran) {
      return absl::InternalError(
          absl::StrCat("Run failed: ", ran.Error().Message()));
    }
    std::vector<float> out(n);
    auto read = (*outputs)[out_index].Read<float>(absl::MakeSpan(out));
    if (!read) {
      return absl::InternalError(
          absl::StrCat("read output: ", read.Error().Message()));
    }
    results.push_back(std::move(out));
  }
  return results;
}

absl::StatusOr<Case> ParseCase() {
  Case c;
  c.q_heads = absl::GetFlag(FLAGS_q_heads);
  c.kv_heads = absl::GetFlag(FLAGS_kv_heads);
  c.head_dim = absl::GetFlag(FLAGS_head_dim);
  c.kv_len = absl::GetFlag(FLAGS_kv_len);
  c.seq = absl::GetFlag(FLAGS_seq);
  c.position = absl::GetFlag(FLAGS_position);
  if (c.position < 0) c.position = c.seq == 1 ? 5 : 0;
  c.mask = absl::GetFlag(FLAGS_mask);
  c.mask_pattern = absl::GetFlag(FLAGS_mask_pattern);
  c.mask_floor = absl::GetFlag(FLAGS_mask_floor);
  c.is_causal = absl::GetFlag(FLAGS_is_causal);
  c.param_tensor = absl::GetFlag(FLAGS_param_tensor);
  c.from_cache_update = absl::GetFlag(FLAGS_from_cache_update);
  const std::string context = absl::GetFlag(FLAGS_context);
  c.harness = context == "harness";
  c.composite = absl::GetFlag(FLAGS_composite);
  if (!c.composite && c.param_tensor) {
    return absl::InvalidArgumentError(
        "--composite=false takes no param tensor (no op would read it)");
  }
  if (c.q_heads <= 0 || c.kv_heads <= 0 || c.q_heads % c.kv_heads != 0) {
    return absl::InvalidArgumentError("q_heads must be a multiple of kv_heads");
  }
  if (c.seq <= 0 || c.head_dim <= 0 || c.position + c.seq > c.kv_len) {
    return absl::InvalidArgumentError("need position + seq <= kv_len");
  }
  if (c.mask != "float" && c.mask != "bool" && c.mask != "none") {
    return absl::InvalidArgumentError("--mask must be float|bool|none");
  }
  if (c.mask_pattern != "causal" && c.mask_pattern != "all") {
    return absl::InvalidArgumentError("--mask_pattern must be causal|all");
  }
  if (c.is_causal != "false" && c.is_causal != "true" &&
      c.is_causal != "absent") {
    return absl::InvalidArgumentError("--is_causal must be false|true|absent");
  }
  if (c.from_cache_update != "absent" && c.from_cache_update != "true") {
    return absl::InvalidArgumentError("--from_cache_update must be absent|true");
  }
  if (context != "none" && context != "harness") {
    return absl::InvalidArgumentError("--context must be none|harness");
  }
  const std::string values = absl::GetFlag(FLAGS_param_values);
  if (values.empty()) {
    const int end = c.position + c.seq;
    c.param_values = {c.position, end, end, c.seq, 0, 0, 0};
  } else {
    for (absl::string_view part : absl::StrSplit(values, ',')) {
      int32_t v;
      if (!absl::SimpleAtoi(part, &v)) {
        return absl::InvalidArgumentError("--param_values: not an integer");
      }
      c.param_values.push_back(v);
    }
    if (c.param_values.size() != 7) {
      return absl::InvalidArgumentError("--param_values needs 7 values");
    }
  }
  return c;
}

}  // namespace

absl::Status Run(int& exit_code) {
  auto case_or = ParseCase();
  if (!case_or.ok()) return case_or.status();
  const Case c = *case_or;
  const std::string label = absl::GetFlag(FLAGS_label);
  const double line = absl::GetFlag(FLAGS_threshold);
  const int gpu_runs = absl::GetFlag(FLAGS_gpu_runs);
  const bool run_cpu = absl::GetFlag(FLAGS_run_cpu);
  const bool run_gpu = absl::GetFlag(FLAGS_run_gpu) && gpu_runs > 0;

  const std::string attrs = absl::StrCat(
      "k_ts_idx=2 v_ts_idx=3",
      c.is_causal == "absent" ? "" : absl::StrCat(" is_causal=", c.is_causal),
      c.from_cache_update == "true" ? " from_cache_update=true" : "");
  std::cout << "case " << label << ": query [1, " << c.q_heads << ", " << c.seq
            << ", " << c.head_dim << "], key_cache [1, " << c.kv_heads << ", "
            << c.kv_len << ", " << c.head_dim << "], value_cache [1, "
            << c.kv_heads << ", " << c.head_dim << ", " << c.kv_len
            << "], mask " << c.mask << " " << c.mask_pattern << " (position "
            << c.position << "), param_tensor "
            << (c.param_tensor
                    ? absl::StrCat("{", absl::StrJoin(c.param_values, ", "), "}")
                    : std::string("none"))
            << ", attributes " << attrs << ", context "
            << (c.harness ? "harness" : "none")
            << (c.composite ? "" : ", no composite (bare ops)") << ", gpu "
            << absl::GetFlag(FLAGS_gpu_precision) << " storage "
            << absl::GetFlag(FLAGS_gpu_buffer_storage) << ", seed "
            << absl::GetFlag(FLAGS_seed) << std::endl;

  const std::string path = absl::GetFlag(FLAGS_tflite_path);
  auto saved = BuildAndSave(c, path);
  if (!saved.ok()) return saved;
  std::error_code ec;
  std::cout << "model: " << path << " ("
            << std::filesystem::file_size(path, ec) << " bytes)" << std::endl;

  const Inputs in =
      MakeInputs(c, static_cast<uint32_t>(absl::GetFlag(FLAGS_seed)));
  const std::vector<double> ref_hsd = Reference(c, in);
  // The reference in the output buffer's layout.
  std::vector<double> ref(ref_hsd.size());
  for (int h = 0; h < c.q_heads; ++h) {
    for (int s = 0; s < c.seq; ++s) {
      for (int d = 0; d < c.head_dim; ++d) {
        ref[OutIndex(c, h, s, d)] =
            ref_hsd[(static_cast<size_t>(h) * c.seq + s) * c.head_dim + d];
      }
    }
  }

  std::vector<::litert::EnvironmentOptions::Option> env_options;
  const std::string log_level = absl::GetFlag(FLAGS_log_level);
  if (!log_level.empty()) {
    int64_t severity;
    if (log_level == "info") {
      severity = kLiteRtLogSeverityInfo;
    } else if (log_level == "debug") {
      severity = kLiteRtLogSeverityDebug;
    } else if (log_level == "verbose") {
      severity = kLiteRtLogSeverityVerbose;
    } else {
      return absl::InvalidArgumentError("--log_level must be info|debug|verbose");
    }
    env_options.emplace_back(
        ::litert::EnvironmentOptions::Tag::kMinLoggerSeverity,
        ::litert::LiteRtVariant(severity));
  }
  auto env = ::litert::Environment::Create(
      ::litert::EnvironmentOptions(env_options));
  if (!env) return absl::InternalError("Environment::Create failed");

  std::vector<float> cpu_out;
  std::string cpu_json = "null", gpu_json = "null";
  if (run_cpu) {
    auto model = Compile(*env, path, /*gpu=*/false, cpu_json);
    if (!model.ok()) return model.status();
    auto outs = RunModel(*model, c, in, 1);
    if (!outs.ok()) return outs.status();
    cpu_out = std::move((*outs)[0]);
  }
  std::vector<std::vector<float>> gpu_outs;
  if (run_gpu) {
    auto model = Compile(*env, path, /*gpu=*/true, gpu_json);
    if (!model.ok()) return model.status();
    auto outs = RunModel(*model, c, in, gpu_runs);
    if (!outs.ok()) return outs.status();
    gpu_outs = std::move(*outs);
  }

  std::ostringstream json;
  json << "{\"label\": \"" << label << "\", \"q_heads\": " << c.q_heads
       << ", \"kv_heads\": " << c.kv_heads << ", \"head_dim\": " << c.head_dim
       << ", \"kv_len\": " << c.kv_len << ", \"seq\": " << c.seq
       << ", \"position\": " << c.position << ", \"mask\": \"" << c.mask
       << "\", \"mask_pattern\": \"" << c.mask_pattern
       << "\", \"mask_floor\": " << Num(c.mask_floor)
       << ", \"is_causal\": \"" << c.is_causal << "\", \"param_tensor\": "
       << (c.param_tensor
               ? absl::StrCat("[", absl::StrJoin(c.param_values, ", "), "]")
               : std::string("null"))
       << ", \"from_cache_update\": \"" << c.from_cache_update
       << "\", \"context\": \"" << (c.harness ? "harness" : "none")
       << "\", \"composite\": " << (c.composite ? "true" : "false")
       << ", \"gpu_precision\": \"" << absl::GetFlag(FLAGS_gpu_precision)
       << "\", \"gpu_buffer_storage\": \""
       << absl::GetFlag(FLAGS_gpu_buffer_storage)
       << "\", \"seed\": " << absl::GetFlag(FLAGS_seed)
       << ", \"threshold\": " << Num(line) << ", \"delegation\": {\"cpu\": "
       << cpu_json << ", \"gpu\": " << gpu_json << "}";

  bool pass = true;
  if (run_cpu) {
    const Diff d = Compare(c, cpu_out, ref, line);
    std::cout << "cpu vs float64 reference: max_abs_diff " << Num(d.max_abs)
              << (d.nan ? absl::StrCat(" (", d.nan, " NaN)") : "")
              << std::endl;
    json << ", \"cpu_vs_ref\": " << Num(d.max_abs);
  }
  std::vector<std::string> gpu_ref_json, gpu_cpu_json, gpu_run1_json;
  for (int r = 0; r < static_cast<int>(gpu_outs.size()); ++r) {
    const Diff dr = Compare(c, gpu_outs[r], ref, line);
    gpu_ref_json.push_back(Num(dr.max_abs));
    gpu_run1_json.push_back(
        Num(Compare(c, gpu_outs[r], gpu_outs[0], line).max_abs));
    std::cout << "gpu run " << r + 1 << " vs float64 reference: max_abs_diff "
              << Num(dr.max_abs)
              << (dr.nan ? absl::StrCat(" (", dr.nan, " NaN)") : "")
              << ", vs gpu run 1: " << gpu_run1_json.back() << std::endl;
    if (!run_cpu) continue;
    const Diff d = Compare(c, gpu_outs[r], cpu_out, line);
    gpu_cpu_json.push_back(Num(d.max_abs));
    std::cout << "cpu vs gpu run " << r + 1 << ": max_abs_diff "
              << Num(d.max_abs) << ", " << d.over << " of "
              << gpu_outs[r].size() << " elements over " << line << std::endl;
    if (d.over > 0 || d.nan > 0) pass = false;
    if (r == 0) {
      if (d.over > 0) {
        const size_t i = OutIndex(c, d.bad_h, d.bad_s, d.bad_d);
        std::cout << "first element over the line (output order): head "
                  << d.bad_h << ", token " << d.bad_s << ", dim " << d.bad_d
                  << ": cpu " << cpu_out[i] << ", gpu " << gpu_outs[r][i]
                  << ", reference " << ref[i] << std::endl;
        json << ", \"first_over\": {\"head\": " << d.bad_h
             << ", \"token\": " << d.bad_s << ", \"dim\": " << d.bad_d
             << ", \"cpu\": " << Num(cpu_out[i])
             << ", \"gpu\": " << Num(gpu_outs[r][i])
             << ", \"reference\": " << Num(ref[i]) << "}";
      }
      std::cout << "cpu vs gpu run 1, max per query head: "
                << NumList(d.per_head) << std::endl;
      std::cout << "cpu vs gpu run 1, max per query token: "
                << NumList(d.per_token) << std::endl;
      json << ", \"over_line\": " << d.over
           << ", \"per_head_max\": " << NumList(d.per_head)
           << ", \"per_token_max\": " << NumList(d.per_token);
    }
  }
  json << ", \"gpu_vs_ref\": [" << absl::StrJoin(gpu_ref_json, ", ")
       << "], \"cpu_vs_gpu\": [" << absl::StrJoin(gpu_cpu_json, ", ")
       << "], \"gpu_vs_gpu_run1\": [" << absl::StrJoin(gpu_run1_json, ", ")
       << "]";
  const bool compared = run_cpu && run_gpu;
  const std::string result = compared ? (pass ? "PASS" : "FAIL") : "NOT_COMPARED";
  json << ", \"result\": \"" << result << "\"}\n";

  const std::string dump_dir = absl::GetFlag(FLAGS_dump_dir);
  if (!dump_dir.empty()) {
    std::filesystem::create_directories(dump_dir, ec);
    auto st = WriteRaw(dump_dir + "/query.f32", c.harness ? in.q_raw : in.q);
    if (st.ok()) st = WriteRaw(dump_dir + "/key_cache.f32", in.k_cache);
    if (st.ok()) st = WriteRaw(dump_dir + "/value_cache.f32", in.v_cache);
    if (st.ok() && c.harness) st = WriteRaw(dump_dir + "/key_new.f32", in.k_new);
    if (st.ok() && c.harness) {
      st = WriteRaw(dump_dir + "/value_new.f32", in.v_new);
    }
    if (st.ok() && c.mask == "bool" && !c.harness) {
      st = WriteRaw(dump_dir + "/attention_mask.u8", in.mask_b);
    } else if (st.ok() && c.mask != "none") {
      st = WriteRaw(dump_dir + "/attention_mask.f32", in.mask_f);
    }
    std::vector<float> ref_f(ref.begin(), ref.end());
    if (st.ok()) st = WriteRaw(dump_dir + "/reference.f32", ref_f);
    if (st.ok() && run_cpu) st = WriteRaw(dump_dir + "/cpu.f32", cpu_out);
    for (int r = 0; st.ok() && r < static_cast<int>(gpu_outs.size()); ++r) {
      st = WriteRaw(absl::StrCat(dump_dir, "/gpu_run", r + 1, ".f32"),
                    gpu_outs[r]);
    }
    if (!st.ok()) return st;
    std::ofstream summary(dump_dir + "/summary.json");
    summary << json.str();
    std::cout << "dumped to " << dump_dir << std::endl;
  }
  std::cout << "RESULT " << result << " " << label << std::endl;
  exit_code = compared ? (pass ? 0 : 2) : 0;
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::sdpa_repro

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  int exit_code = 1;
  absl::Status status =
      litert::tensor::examples::sdpa_repro::Run(exit_code);
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  return exit_code;
}
