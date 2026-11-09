// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tokenizer.h"
#include <memory>

namespace Garnet::Tokenization {
// Deterministic tokenizer.json Unigram/Metaspace path, including the checkpoint
// vocabulary scores, declared normalizer, added tokens and EOS template.
class UnigramTokenizer final : public Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> m_impl;
public:
    UnigramTokenizer();
    ~UnigramTokenizer() override;
    bool LoadFromFolder(const std::string& folder,std::string* error=nullptr) override;
    std::vector<int64_t> Encode(const std::string& text,bool addSpecialTokens=false) const override;
    std::string Decode(const std::vector<int64_t>& ids,bool skipSpecialTokens=true) const override;
    int64_t TokenId(const std::string& token) const override;
    TokenizerInfo Info() const override;
};
}
