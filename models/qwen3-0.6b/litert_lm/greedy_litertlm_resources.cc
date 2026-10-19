// SPDX-License-Identifier: Apache-2.0
// For the output match only: a ModelResourcesRegistry entry for model paths
// that end in ".greedy.litertlm" (a link to a .litertlm file). The file is
// loaded with LiteRT-LM's own .litertlm resources class, with two changes:
// - the sampler parameters of the LLM metadata become greedy (TOP_P with
//   k = 1, the parameters LiteRT-LM sets for CPU and GPU when a file has
//   none), so every backend picks the most likely token;
// - the tokenizer logs the ids of every text it encodes and every id
//   sequence it decodes, so the ids that enter prefill and the generated ids
//   can be read from the log whatever the backend.
// The executor, the TFLite model and the tokenizer itself are unchanged, and a
// path without the suffix is loaded by LiteRT-LM as before. Link this file
// with alwayslink = 1.

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_backend_registry.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace {

namespace lm = ::litert::lm;
using ::litert::support::TokenIds;
using ::litert::support::Tokenizer;
using ::litert::support::TokenizerType;

// Passes every call to the file's tokenizer and logs the ids.
class IdLoggingTokenizer : public Tokenizer {
 public:
  explicit IdLoggingTokenizer(std::unique_ptr<Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {}

  using Tokenizer::TokenIdsToText;

  TokenizerType GetTokenizerType() const override {
    return tokenizer_->GetTokenizerType();
  }
  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    absl::StatusOr<TokenIds> ids = tokenizer_->TextToTokenIds(text);
    if (ids.ok()) {
      ABSL_LOG(INFO) << "greedy.litertlm tokenizer encode: " << ids->size()
                     << " ids [" << absl::StrJoin(*ids, " ") << "]";
    }
    return ids;
  }
  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return tokenizer_->TokenToId(token);
  }
  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    ABSL_LOG(INFO) << "greedy.litertlm tokenizer decode: ["
                   << absl::StrJoin(token_ids, " ") << "]";
    return tokenizer_->TokenIdsToText(token_ids, skip_special_tokens);
  }
  std::vector<std::string> GetTokens() const override {
    return tokenizer_->GetTokens();
  }
  int GetVocabSize() const override { return tokenizer_->GetVocabSize(); }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
};

class GreedyLitertLmResources : public lm::ModelResourcesLitertLm {
 public:
  explicit GreedyLitertLmResources(std::unique_ptr<lm::LitertLmLoader> loader)
      : ModelResourcesLitertLm(std::move(loader),
                               /*enable_file_backed_model_loading=*/false) {}

  using ModelResourcesLitertLm::GetTokenizer;

  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    if (!metadata_.has_value()) {
      ABSL_ASSIGN_OR_RETURN(const lm::proto::LlmMetadata* file_metadata,
                            ModelResourcesLitertLm::GetLlmMetadata());
      metadata_ = *file_metadata;
      lm::proto::SamplerParameters& sampler =
          *metadata_->mutable_sampler_params();
      ABSL_LOG(INFO) << "greedy.litertlm: the file's sampler: type "
                     << lm::proto::SamplerParameters::Type_Name(sampler.type())
                     << " k " << sampler.k() << " p " << sampler.p()
                     << " temperature " << sampler.temperature()
                     << "; replaced by TOP_P k 1 (greedy)";
      sampler.set_type(lm::proto::SamplerParameters::TOP_P);
      sampler.set_k(1);
      sampler.set_p(0.95f);
      sampler.set_temperature(1.0f);
      sampler.set_seed(0);
    }
    return &*metadata_;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                          ModelResourcesLitertLm::GetTokenizer());
    return std::make_unique<IdLoggingTokenizer>(std::move(tokenizer));
  }

 private:
  std::optional<lm::proto::LlmMetadata> metadata_;
};

absl::StatusOr<std::unique_ptr<lm::ModelResources>> CreateGreedyResources(
    const lm::ModelAssets& model_assets) {
  ABSL_LOG(INFO) << "greedy.litertlm: ModelResourcesRegistry loader called for "
                 << model_assets.GetPath().value_or("(no path)");
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<lm::ScopedFile> file,
                        model_assets.GetOrCreateScopedFile());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::LitertLmLoader> loader,
                        lm::LitertLmLoader::Create(std::move(file)));
  return std::make_unique<GreedyLitertLmResources>(std::move(loader));
}

}  // namespace

LITERT_LM_REGISTER_MODEL_RESOURCES(".greedy.litertlm", CreateGreedyResources);
