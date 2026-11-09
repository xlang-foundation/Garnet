// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tokenizer.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace Garnet::Tokenization::Assets {
inline nlohmann::json Read(const std::string& folder, const char* name, bool required = false) {
    const auto path = std::filesystem::path(folder) / name;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (required || std::filesystem::exists(path))
            throw std::runtime_error("cannot read tokenizer asset: " + path.string());
        return nlohmann::json::object();
    }
    auto result = nlohmann::json::parse(file);
    if (!result.is_object()) throw std::runtime_error("tokenizer asset must be an object: " + path.string());
    return result;
}
inline std::string Family(const nlohmann::json& config, const nlohmann::json& tokenizerConfig) {
    const auto type = config.value("model_type", std::string());
    if (type.rfind("qwen", 0) == 0) return "qwen";
    if (type.rfind("gemma", 0) == 0) return "gemma";
    if (type.rfind("deepseek", 0) == 0) return "deepseek";
    if (type == "mistral" || type == "mixtral") return "mistral";
    if (type == "llama") return "llama";
    if (type == "t5" || type == "mt5" || type == "umt5") return "t5";
    const auto cls = tokenizerConfig.value("tokenizer_class", std::string());
    if (cls.rfind("Qwen", 0) == 0) return "qwen";
    if (cls.rfind("Gemma", 0) == 0) return "gemma";
    if (cls.rfind("T5", 0) == 0 || cls.rfind("MT5", 0) == 0 || cls.rfind("UMT5", 0) == 0) return "t5";
    return {};
}
inline int64_t SpecialId(const Tokenizer& tokenizer, const nlohmann::json& config,
                         const nlohmann::json& tokenizerConfig, const char* key) {
    auto it = tokenizerConfig.find(key);
    if (it != tokenizerConfig.end()) {
        if (it->is_null()) return -1;
        if (it->is_string()) return tokenizer.TokenId(it->get<std::string>());
        if (it->is_object() && it->contains("content"))
            return tokenizer.TokenId((*it)["content"].get<std::string>());
        throw std::runtime_error(std::string("unsupported special token configuration: ") + key);
    }
    it = config.find(std::string(key) + "_id");
    return it != config.end() && it->is_number_integer() ? it->get<int64_t>() : -1;
}
}
