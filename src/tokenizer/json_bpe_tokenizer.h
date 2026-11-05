// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tokenizer.h"

namespace Garnet::Tokenization {
// Native tokenizer.json BPE pipelines. Unsupported stages fail at load time.
// Instances are immutable after loading and may be shared by requests.
class JsonBpeTokenizer final : public Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> m_impl;
public:
    JsonBpeTokenizer();
    ~JsonBpeTokenizer() override;
    bool LoadFromFolder(const std::string&, std::string* error = nullptr) override;
    std::vector<int64_t> Encode(const std::string&, bool addSpecialTokens = false) const override;
    std::string Decode(const std::vector<int64_t>&, bool skipSpecialTokens = true) const override;
    int64_t TokenId(const std::string&) const override;
    TokenizerInfo Info() const override;
};
}
