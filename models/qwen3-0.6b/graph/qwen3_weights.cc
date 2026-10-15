// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B checkpoint loading. See qwen3_weights.h.

#include "tensor/examples/qwen3-0.6b/qwen3_weights.h"

#include <cstddef>
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

}  // namespace

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Qwen3Config& config,
                                                const std::string& path) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

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
    tensor.SetName(spec.name);
    weights[spec.name] = std::move(tensor);
  }
  return weights;
}

}  // namespace litert::tensor::examples::qwen3
