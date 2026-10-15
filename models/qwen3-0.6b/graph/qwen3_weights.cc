// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B checkpoint loading. See qwen3_weights.h.

#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::qwen3 {

namespace {

constexpr int kInt4BlockSize = 32;

// Mirrors the loader's is_layernorm predicate (safetensor_loader.cc).
bool LoaderAddsGemmaNormOffset(const std::string& name) {
  return absl::StrContains(name, "layernorm") ||
         absl::EndsWith(name, "norm.weight");
}

absl::Status SubtractOne(TfTensor& tensor) {
  auto buffer = tensor.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->LockMutable();
  float* data = reinterpret_cast<float*>(lock.data());
  size_t count = lock.size() / sizeof(float);
  for (size_t i = 0; i < count; ++i) data[i] -= 1.0f;
  return absl::OkStatus();
}

// The decoder's 2-D [out, in] matmul weights: q/k/v/o and gate/up/down.
bool IsQuantizableFcWeight(const std::string& name) {
  return absl::StrContains(name, "_proj.weight");
}

// Bit-packs int4 codes (values in [-7, 7], one int8 each) into the kI4 layout
// (even index in the low nibble) and attaches blockwise-32 quantization.
TfTensor BuildInt4Tensor(const int8_t* q, std::vector<float> scales,
                         int out_channels, int in_dim,
                         const std::string& name) {
  const size_t count =
      static_cast<size_t>(out_channels) * static_cast<size_t>(in_dim);
  std::vector<char> packed((count + 1) / 2, 0);
  for (size_t i = 0; i < count; ++i) {
    char nib = static_cast<char>(q[i] & 0xF);
    packed[i / 2] |= (i % 2 == 0) ? nib : static_cast<char>(nib << 4);
  }
  auto q_buffer = OwningCpuBuffer::Copy(packed.data(), packed.size());
  TfTensor quantized({.name = name,
                      .type = Type::kI4,
                      .shape = {out_channels, in_dim},
                      .buffer = std::move(q_buffer)});
  quantized.SetQuantization(std::make_shared<BlockwiseQuantization>(
      std::move(scales), std::vector<int64_t>(), kInt4BlockSize,
      /*quantized_dimension=*/0));
  return quantized;
}

// Symmetric nearest-integer quantization of a [out, in] fp32 weight. int8:
// one scale per output channel. int4: one fp16-rounded scale per block of 32
// input elements. Adds the stored bytes (codes plus scales) to *bytes.
absl::StatusOr<TfTensor> QuantizeFcWeight(const TfTensor& tensor,
                                          const std::string& name,
                                          WeightQuantMode mode,
                                          int64_t* bytes) {
  const auto& shape = tensor.GetShape();
  if (shape.size() != 2) {
    return absl::FailedPreconditionError(
        absl::StrCat(name, ": expected 2-D weight for quantization"));
  }
  const int out_channels = shape[0];
  const int in_dim = shape[1];
  auto buffer = tensor.GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* w = reinterpret_cast<const float*>(lock.data());

  const bool int4 = mode == WeightQuantMode::kInt4;
  const float qmax = int4 ? 7.0f : 127.0f;
  const int block = int4 ? kInt4BlockSize : in_dim;
  if (in_dim % block != 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        name, ": in_dim ", in_dim, " not divisible by block ", block));
  }
  const int blocks_per_row = in_dim / block;

  std::vector<float> scales(static_cast<size_t>(out_channels) *
                            blocks_per_row);
  std::vector<int8_t> q(static_cast<size_t>(out_channels) * in_dim);
  for (int c = 0; c < out_channels; ++c) {
    const float* row = w + static_cast<size_t>(c) * in_dim;
    int8_t* q_row = q.data() + static_cast<size_t>(c) * in_dim;
    for (int b = 0; b < blocks_per_row; ++b) {
      float amax = 0.0f;
      for (int i = b * block; i < (b + 1) * block; ++i) {
        amax = std::max(amax, std::fabs(row[i]));
      }
      float scale = amax > 0.0f ? amax / qmax : 1.0f;
      if (int4) {
        // Blockwise scales are serialized as fp16; quantize against the
        // fp16-rounded value the runtime will use.
        scale = static_cast<float>(fp16_t(scale));
      }
      scales[static_cast<size_t>(c) * blocks_per_row + b] = scale;
      for (int i = b * block; i < (b + 1) * block; ++i) {
        float v = static_cast<float>(std::lround(row[i] / scale));
        q_row[i] = static_cast<int8_t>(std::min(qmax, std::max(-qmax, v)));
      }
    }
  }

  const int64_t codes = static_cast<int64_t>(q.size());
  if (int4) {
    *bytes += (codes + 1) / 2 + static_cast<int64_t>(scales.size()) * 2;
    return BuildInt4Tensor(q.data(), std::move(scales), out_channels, in_dim,
                           name);
  }
  *bytes += codes + static_cast<int64_t>(scales.size()) * 4;
  auto q_buffer = OwningCpuBuffer::Copy(
      reinterpret_cast<const char*>(q.data()), q.size());
  TfTensor quantized({.name = name,
                      .type = Type::kI8,
                      .shape = {out_channels, in_dim},
                      .buffer = std::move(q_buffer)});
  quantized.SetQuantization(std::make_shared<PerChannelAffineQuantization>(
      std::move(scales), std::vector<int64_t>(out_channels, 0),
      /*quantized_dimension=*/0));
  return quantized;
}

}  // namespace

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Qwen3Config& config,
                                                const std::string& path,
                                                WeightQuantMode quant_mode,
                                                bool quantized_lm_head,
                                                QuantStats* stats) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  QuantStats local_stats;
  QuantStats& st = stats ? *stats : local_stats;
  WeightMap weights;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    auto handle_or = loader.LoadTensor(
        spec.name, SafetensorLoader::QuantizedLoadMode::kDequantizeToFp32);
    if (!handle_or.ok()) {
      return absl::NotFoundError(absl::StrCat(
          "checkpoint missing ", spec.name, ": ", handle_or.status().message()));
    }
    TfTensor tensor(*handle_or);

    const auto& loaded_shape = tensor.GetShape();
    std::vector<int> loaded(loaded_shape.begin(), loaded_shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name, ": checkpoint shape [", absl::StrJoin(loaded, ","),
          "] != expected [", absl::StrJoin(spec.shape, ","), "]"));
    }
    if (tensor.GetType() != Type::kFP32) {
      return absl::FailedPreconditionError(
          absl::StrCat(spec.name, ": loader did not return fp32"));
    }
    if (LoaderAddsGemmaNormOffset(spec.name)) {
      auto status = SubtractOne(tensor);
      if (!status.ok()) return status;
    }
    if (quant_mode != WeightQuantMode::kNone &&
        IsQuantizableFcWeight(spec.name)) {
      int64_t count = 1;
      for (int d : spec.shape) count *= d;
      st.fp32_bytes += count * 4;
      auto quantized_or =
          QuantizeFcWeight(tensor, spec.name, quant_mode, &st.quantized_bytes);
      if (!quantized_or.ok()) return quantized_or.status();
      ++st.quantized_tensors;
      weights[spec.name] = std::move(*quantized_or);
      continue;
    }
    if (quantized_lm_head && spec.name == "model.embed_tokens.weight") {
      auto lm_head_or = QuantizeFcWeight(tensor, "lm_head.weight",
                                         WeightQuantMode::kInt8,
                                         &st.lm_head_bytes);
      if (!lm_head_or.ok()) return lm_head_or.status();
      weights["lm_head.weight"] = std::move(*lm_head_or);
    }
    tensor.SetName(spec.name);
    weights[spec.name] = std::move(tensor);
  }
  return weights;
}

}  // namespace litert::tensor::examples::qwen3
