// SPDX-License-Identifier: Apache-2.0
// Qwen3-0.6B prefill and decode graphs. See qwen3_graph.h.

#include "tensor/examples/qwen3-0.6b/qwen3_graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/qwen3-0.6b/qwen3_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::qwen3 {

namespace {

TfTensor Const1(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

// Deterministic LCG so synthetic runs are reproducible without an RNG.
std::vector<float> SyntheticData(size_t count, unsigned& state, float scale) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = scale * (static_cast<float>(state >> 8) /
                           static_cast<float>(1u << 24) * 2.0f -
                       1.0f);
  }
  return data;
}

const TfTensor& GetWeight(const WeightMap& weights, const std::string& name) {
  auto it = weights.find(name);
  ABSL_CHECK(it != weights.end()) << "missing weight: " << name;
  return it->second;
}

// RMS normalization over the last axis, raw weight (no +1 offset).
TfTensor RmsNormRaw(const TfTensor& input, const TfTensor& scale, float eps) {
  TfTensor x_squared = Mul(input, input);
  int last_axis = static_cast<int>(input.GetShape().size()) - 1;
  TfTensor mean_squared = Mean(x_squared, {last_axis}, /*keep_dims=*/true);
  TfTensor inv_rms = Rsqrt(Add(mean_squared, Const1(eps)));
  return Mul(Mul(input, inv_rms), scale);
}

std::vector<uint8_t> RmsNormAttributes(float eps) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", eps); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// RMS norm, raw or as the odml.rms_norm composite whose decomposition is the
// raw math.
TfTensor RmsNorm(const TfTensor& input, const TfTensor& scale, float eps,
                 bool composite) {
  if (!composite) return RmsNormRaw(input, scale, eps);
  StableHLOCompositeOptions opts{.name = "odml.rms_norm",
                                 .composite_attributes =
                                     RmsNormAttributes(eps)};
  return StableHLOComposite(
      opts,
      [eps](TfTensor x, TfTensor s) { return RmsNormRaw(x, s, eps); }, input,
      scale);
}

TfTensor Silu(const TfTensor& x) { return Mul(x, Logistic(x)); }

// Rotate-half RoPE with host-provided cos/sin rows.
TfTensor ApplyRope(const TfTensor& x, const TfTensor& cos,
                   const TfTensor& sin) {
  const auto& s = x.GetShape();  // [1, heads, seq, head_dim]
  int half = s[3] / 2;
  TfTensor x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  TfTensor x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  TfTensor rotated = Concatenation({Mul(x2, Const1(-1.0f)), x1}, /*axis=*/3);
  return Add(Mul(x, cos), Mul(rotated, sin));
}

// GQA without materializing repeated K/V: kv groups fold into the batch
// axis and each group's q heads into the sequence axis, so q head h reads
// kv group h / (heads / kv) (repeat_kv semantics).
// q [1, heads, seq, hd], k/v [1, kv, kv_seq, hd], mask [1, 1, seq, kv_seq]
// -> [1, heads, seq, hd].
TfTensor GqaAttention(const TfTensor& q, const TfTensor& k, const TfTensor& v,
                      const TfTensor& mask, int n_heads, int n_kv_groups,
                      int head_dim) {
  int g = n_heads / n_kv_groups;
  int seq = q.GetShape()[2];
  int kv_seq = k.GetShape()[2];
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  TfTensor qg = Reshape(q, {n_kv_groups, 1, g * seq, head_dim});
  TfTensor kg = Reshape(k, {n_kv_groups, 1, kv_seq, head_dim});
  TfTensor vg = Reshape(v, {n_kv_groups, 1, kv_seq, head_dim});

  TfTensor scores = BatchMatMul(qg, kg, /*adj_x=*/false, /*adj_y=*/true);
  scores = Reshape(scores, {1, n_heads, seq, kv_seq});
  scores = Mul(scores, Const1(scale));
  scores = Add(scores, mask);
  TfTensor attn = Softmax(scores);
  attn = Reshape(attn, {n_kv_groups, 1, g * seq, kv_seq});
  TfTensor out = BatchMatMul(attn, vg);
  return Reshape(out, {1, n_heads, seq, head_dim});
}

std::vector<uint8_t> SdpaAttributes(float scale) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("scale", scale); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// The same GQA fold as GqaAttention, presented to the
// odml.scaled_dot_product_attention composite as batched multi-head shapes:
// each kv group is one batch entry holding one head over g * seq query rows,
// q [G, g*seq, 1, hd] against k/v [G, kv_seq, 1, hd] (BSND). The un-folded
// grouped form is claimed by the Metal delegate and computed wrong when the
// query length is above 1. Folded query row j * seq + s reads mask row s, so
// the mask is tiled to [1, 1, g*seq, kv_seq] before it enters the composite.
// The decomposition runs the raw fold in the folded shapes; every element
// gets the same scale, mask term and softmax row as in GqaAttention.
TfTensor SdpaGqaAttention(const TfTensor& q, const TfTensor& k,
                          const TfTensor& v, const TfTensor& mask, int n_heads,
                          int n_kv_groups, int head_dim) {
  const int G = n_kv_groups;
  const int g = n_heads / n_kv_groups;
  const int seq = q.GetShape()[2];
  const int kv_seq = k.GetShape()[2];
  const int D = head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  TfTensor qf = Reshape(q, {G, g * seq, 1, D});
  TfTensor kf = Reshape(k, {G, kv_seq, 1, D});
  TfTensor vf = Reshape(v, {G, kv_seq, 1, D});
  TfTensor mf = g == 1 ? mask
                       : Reshape(Tile(mask, {1, g, 1, 1}),
                                 {1, 1, g * seq, kv_seq});
  StableHLOCompositeOptions opts{
      .name = "odml.scaled_dot_product_attention",
      .composite_attributes = SdpaAttributes(scale)};
  TfTensor out = StableHLOComposite(
      opts,
      [G, g, seq, kv_seq, D, scale](TfTensor dq, TfTensor dk, TfTensor dv,
                                    TfTensor dm) {
        // Axis 1 <-> 2 swaps with a size-1 axis keep the linear order, so
        // Reshape stands in for the BSND -> BNSD Transpose.
        TfTensor qt = Reshape(dq, {G, 1, g * seq, D});
        TfTensor kt = Reshape(dk, {G, 1, kv_seq, D});
        TfTensor vt = Reshape(dv, {G, 1, kv_seq, D});
        TfTensor scores =
            BatchMatMul(qt, kt, /*adj_x=*/false, /*adj_y=*/true);
        scores = Mul(scores, Const1(scale));
        scores = Add(scores, dm);
        TfTensor attn = Softmax(scores);
        TfTensor o = BatchMatMul(attn, vt);  // [G, 1, g*seq, D]
        return Reshape(o, {G, g * seq, 1, D});
      },
      qf, kf, vf, mf);
  return Reshape(out, {1, n_heads, seq, head_dim});
}

struct AttentionResult {
  TfTensor output;
  TfTensor key_cache;
  TfTensor value_cache;
};

// q/k/v projections -> per-head q_norm / k_norm -> RoPE -> K/V written into
// the fixed-size caches at the runtime cache_position -> GQA -> o_proj.
AttentionResult Attention(const Qwen3Config& config, const TfTensor& input,
                          const std::string& prefix, const Qwen3Inputs& inputs,
                          int layer_idx, const WeightMap& weights) {
  int seq = input.GetShape()[1];

  TfTensor q =
      FullyConnected(input, GetWeight(weights, prefix + ".q_proj.weight"));
  TfTensor k =
      FullyConnected(input, GetWeight(weights, prefix + ".k_proj.weight"));
  TfTensor v =
      FullyConnected(input, GetWeight(weights, prefix + ".v_proj.weight"));

  q = Transpose(Reshape(q, {1, seq, config.n_heads, config.head_dim}),
                {0, 2, 1, 3});
  k = Transpose(Reshape(k, {1, seq, config.n_kv_groups, config.head_dim}),
                {0, 2, 1, 3});
  v = Transpose(Reshape(v, {1, seq, config.n_kv_groups, config.head_dim}),
                {0, 2, 1, 3});

  q = RmsNorm(q, GetWeight(weights, prefix + ".q_norm.weight"),
              config.rms_norm_eps, config.rms_norm_composite);
  k = RmsNorm(k, GetWeight(weights, prefix + ".k_norm.weight"),
              config.rms_norm_eps, config.rms_norm_composite);

  q = ApplyRope(q, inputs.rope_cos, inputs.rope_sin);
  k = ApplyRope(k, inputs.rope_cos, inputs.rope_sin);

  TfTensor key_cache_new = DynamicUpdateSlice(inputs.key_caches[layer_idx], k,
                                              inputs.cache_position);
  key_cache_new.SetName(absl::StrCat("key_cache_", layer_idx, "_new"));
  TfTensor value_cache_new = DynamicUpdateSlice(
      inputs.value_caches[layer_idx], v, inputs.cache_position);
  value_cache_new.SetName(absl::StrCat("value_cache_", layer_idx, "_new"));

  TfTensor out =
      config.sdpa_attention
          ? SdpaGqaAttention(q, key_cache_new, value_cache_new,
                             inputs.attention_mask, config.n_heads,
                             config.n_kv_groups, config.head_dim)
          : GqaAttention(q, key_cache_new, value_cache_new,
                         inputs.attention_mask, config.n_heads,
                         config.n_kv_groups, config.head_dim);
  out = Reshape(Transpose(out, {0, 2, 1, 3}), {1, seq, config.qkv_out_dim()});
  out = FullyConnected(out, GetWeight(weights, prefix + ".o_proj.weight"));
  return {out, key_cache_new, value_cache_new};
}

TfTensor FeedForward(const TfTensor& input, const std::string& prefix,
                     const WeightMap& weights) {
  TfTensor gate =
      FullyConnected(input, GetWeight(weights, prefix + ".gate_proj.weight"));
  TfTensor up =
      FullyConnected(input, GetWeight(weights, prefix + ".up_proj.weight"));
  return FullyConnected(Mul(Silu(gate), up),
                        GetWeight(weights, prefix + ".down_proj.weight"));
}

}  // namespace

std::vector<TfTensor> Qwen3Inputs::AsList() const {
  std::vector<TfTensor> list = {input_ids, cache_position, rope_cos, rope_sin,
                                attention_mask};
  for (size_t i = 0; i < key_caches.size(); ++i) {
    list.push_back(key_caches[i]);
    list.push_back(value_caches[i]);
  }
  return list;
}

std::vector<TfTensor> Qwen3Outputs::AsList() const {
  std::vector<TfTensor> list;
  if (!omit_logits) list.push_back(logits);
  for (size_t i = 0; i < key_caches.size(); ++i) {
    list.push_back(key_caches[i]);
    list.push_back(value_caches[i]);
  }
  for (const TfTensor& t : dumps) list.push_back(t);
  return list;
}

Qwen3Inputs MakeQwen3Inputs(const Qwen3Config& config, bool is_decode) {
  Qwen3Inputs inputs;
  int seq = is_decode ? 1 : config.prefill_len;
  inputs.input_ids =
      TfTensor({.name = "input_ids", .type = Type::kI32, .shape = {seq}});
  inputs.cache_position =
      TfTensor({.name = "cache_position", .type = Type::kI32, .shape = {4}});
  inputs.rope_cos = TfTensor({.name = "rope_cos",
                              .type = Type::kFP32,
                              .shape = {1, 1, seq, config.head_dim}});
  inputs.rope_sin = TfTensor({.name = "rope_sin",
                              .type = Type::kFP32,
                              .shape = {1, 1, seq, config.head_dim}});
  inputs.attention_mask = TfTensor({.name = "attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, seq, config.max_seq}});
  for (int i = 0; i < config.n_layers; ++i) {
    std::vector<int> shape = {1, config.n_kv_groups, config.max_seq,
                              config.head_dim};
    inputs.key_caches.push_back(TfTensor({.name = absl::StrCat("key_cache_", i),
                                          .type = Type::kFP32,
                                          .shape = shape}));
    inputs.value_caches.push_back(
        TfTensor({.name = absl::StrCat("value_cache_", i),
                  .type = Type::kFP32,
                  .shape = shape}));
  }
  return inputs;
}

std::vector<WeightSpec> GetWeightSpecs(const Qwen3Config& config) {
  std::vector<WeightSpec> specs;
  auto add = [&](const std::string& name, const std::vector<int>& shape,
                 float scale = 0.02f) { specs.push_back({name, shape, scale}); };
  add("model.embed_tokens.weight", {config.vocab_size, config.emb_dim});
  for (int i = 0; i < config.n_layers; ++i) {
    std::string p = absl::StrCat("model.layers.", i);
    add(p + ".input_layernorm.weight", {config.emb_dim}, 1.0f);
    add(p + ".self_attn.q_proj.weight", {config.qkv_out_dim(), config.emb_dim});
    add(p + ".self_attn.k_proj.weight", {config.kv_out_dim(), config.emb_dim});
    add(p + ".self_attn.v_proj.weight", {config.kv_out_dim(), config.emb_dim});
    add(p + ".self_attn.o_proj.weight", {config.emb_dim, config.qkv_out_dim()});
    add(p + ".self_attn.q_norm.weight", {config.head_dim}, 1.0f);
    add(p + ".self_attn.k_norm.weight", {config.head_dim}, 1.0f);
    add(p + ".post_attention_layernorm.weight", {config.emb_dim}, 1.0f);
    add(p + ".mlp.gate_proj.weight", {config.hidden_dim, config.emb_dim});
    add(p + ".mlp.up_proj.weight", {config.hidden_dim, config.emb_dim});
    add(p + ".mlp.down_proj.weight", {config.emb_dim, config.hidden_dim});
  }
  add("model.norm.weight", {config.emb_dim}, 1.0f);
  return specs;
}

WeightMap MakeSyntheticWeights(const Qwen3Config& config, unsigned seed) {
  WeightMap weights;
  unsigned state = seed;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    size_t count = 1;
    for (int d : spec.shape) count *= static_cast<size_t>(d);
    std::vector<float> data;
    if (spec.init_scale == 1.0f) {
      data.assign(count, 1.0f);  // norm scales
    } else {
      data = SyntheticData(count, state, spec.init_scale);
    }
    weights[spec.name] =
        TfTensor({.name = spec.name,
                  .type = Type::kFP32,
                  .shape = spec.shape,
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(data)});
  }
  return weights;
}

Qwen3Outputs BuildQwen3(const Qwen3Config& config, const Qwen3Inputs& inputs,
                        const WeightMap& weights, bool is_decode) {
  const bool dump = config.dump_blocks && !is_decode;
  int seq = inputs.input_ids.GetShape()[0];
  // One table for the lookup and the tied lm_head.
  const TfTensor& table = GetWeight(weights, "model.embed_tokens.weight");

  Qwen3Outputs outputs;
  TfTensor hidden = Reshape(EmbeddingLookup(inputs.input_ids, table),
                            {1, seq, config.emb_dim});
  if (dump) {
    hidden.SetName("emb");
    outputs.dumps.push_back(hidden);
  }

  for (int i = 0; i < config.n_layers; ++i) {
    std::string p = absl::StrCat("model.layers.", i);
    TfTensor attn_in =
        RmsNorm(hidden, GetWeight(weights, p + ".input_layernorm.weight"),
                config.rms_norm_eps, config.rms_norm_composite);
    AttentionResult attn =
        Attention(config, attn_in, p + ".self_attn", inputs, i, weights);
    hidden = Add(hidden, attn.output);
    outputs.key_caches.push_back(attn.key_cache);
    outputs.value_caches.push_back(attn.value_cache);

    TfTensor ffn_in = RmsNorm(
        hidden, GetWeight(weights, p + ".post_attention_layernorm.weight"),
        config.rms_norm_eps, config.rms_norm_composite);
    hidden = Add(hidden, FeedForward(ffn_in, p + ".mlp", weights));
    if (dump) {
      hidden.SetName(absl::StrCat("block_", i < 10 ? "0" : "", i));
      outputs.dumps.push_back(hidden);
    }
  }

  if (config.omit_logits && !dump) {
    outputs.omit_logits = true;
    return outputs;
  }
  hidden = RmsNorm(hidden, GetWeight(weights, "model.norm.weight"),
                   config.rms_norm_eps, config.rms_norm_composite);
  if (dump) {
    hidden.SetName("final_norm");
    outputs.dumps.push_back(hidden);
    if (config.dump_final_norm_identity) hidden = Mul(hidden, Const1(1.0f));
  }
  const TfTensor& lm_head = config.separate_lm_head
                                ? GetWeight(weights, "lm_head.weight")
                                : table;
  outputs.logits = FullyConnected(hidden, lm_head);  // [1, seq, vocab]
  outputs.logits.SetName("logits");
  return outputs;
}

}  // namespace litert::tensor::examples::qwen3
