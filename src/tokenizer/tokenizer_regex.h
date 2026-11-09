// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace Garnet::Tokenization {
// Shared native implementation of tokenizer.json's regex Split/Isolated stage.
// The regex is checkpoint-owned. No model-family regex is substituted for it.
class TokenizerRegex {
    struct Impl;
    std::shared_ptr<const Impl> m_impl;
public:
    explicit TokenizerRegex(const std::string& pattern);
    std::vector<std::string> SplitIsolated(const std::string& text) const;
    std::string ReplaceAll(const std::string& text,const std::string& replacement) const;
};
}
