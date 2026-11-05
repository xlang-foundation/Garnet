// SPDX-License-Identifier: Apache-2.0
#include "tokenizer_assets.h"
#include "qwen_tokenizer.h"
#include "json_bpe_tokenizer.h"
#include "tokenizer_utf8.h"

namespace Garnet::Tokenization {
namespace {
// Keep the legacy Qwen C ABI byte-oriented while the common text API returns
// valid UTF-8, including replacement characters for incomplete token bytes.
class QwenTextTokenizer final : public Tokenizer {
    QwenTokenizer tokenizer;
public:
    bool LoadFromFolder(const std::string& folder, std::string* error) override {
        return tokenizer.LoadFromFolder(folder, error);
    }
    std::vector<int64_t> Encode(const std::string& text, bool special) const override {
        if (!Utf8::Valid(text)) throw std::runtime_error("tokenizer input is not valid UTF-8");
        return tokenizer.Encode(text, special);
    }
    std::string Decode(const std::vector<int64_t>& ids, bool skip) const override {
        return Utf8::ReplaceInvalid(tokenizer.Decode(ids, skip));
    }
    int64_t TokenId(const std::string& text) const override { return tokenizer.TokenId(text); }
    TokenizerInfo Info() const override { return tokenizer.Info(); }
};
}
std::unique_ptr<Tokenizer> OpenTokenizer(const std::string& folder, std::string* error) {
    if (error) error->clear();
    try {
        auto config = Assets::Read(folder, "config.json");
        auto tokenizerConfig = Assets::Read(folder, "tokenizer_config.json");
        auto family = Assets::Family(config, tokenizerConfig);
        if (family.empty()) {
            // Preserve standalone Qwen tokenizer folders without model config.
            auto root = Assets::Read(folder, "tokenizer.json", true);
            const auto& vocab = root.at("model").at("vocab");
            if (vocab.is_object() && vocab.contains("<|im_start|>") && vocab.contains("<|im_end|>"))
                family = "qwen";
        }
        std::unique_ptr<Tokenizer> tokenizer;
        if (family == "qwen") tokenizer = std::make_unique<QwenTextTokenizer>();
        else if (family == "mistral" || family == "gemma" || family == "llama" || family == "deepseek")
            tokenizer = std::make_unique<JsonBpeTokenizer>();
        else throw std::runtime_error("unsupported tokenizer family; expected Qwen, Mistral, Gemma, Llama or DeepSeek artifacts");
        if (!tokenizer->LoadFromFolder(folder, error)) return nullptr;
        return tokenizer;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }
}
}
