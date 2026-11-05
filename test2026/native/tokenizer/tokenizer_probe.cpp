// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.h"
#include "qwen_tokenizer.h"
#include "nlohmann/json.hpp"
#include <iostream>

// CPU-only JSON-lines driver: tests use Python references, never inference.
int main(int argc, char** argv) {
    using namespace Garnet::Tokenization;
    using Json = nlohmann::json;
    if (argc != 2) { std::cerr << "usage: tokenizer_probe MODEL_FOLDER\n"; return 2; }
    std::string error;
    auto tokenizer = OpenTokenizer(argv[1], &error);
    if (!tokenizer) { std::cerr << error << '\n'; return 1; }
    for (std::string line; std::getline(std::cin, line);) {
        Json result;
        try {
            auto request = Json::parse(line);
            if (request.value("op", std::string()) == "decode")
                result["text"] = tokenizer->Decode(request.at("ids").get<std::vector<int64_t>>(), request.value("skip_special_tokens", false));
            else if (request.value("op", std::string()) == "legacy_qwen_load") {
                QwenTokenizer legacy; result["loaded"] = legacy.LoadFromFolder(argv[1], &error); result["error"] = error;
            } else {
                result["ids"] = tokenizer->Encode(request.at("text").get<std::string>(), request.value("add_special_tokens", false));
                result["text"] = tokenizer->Decode(result["ids"].get<std::vector<int64_t>>(), request.value("skip_special_tokens", false));
            }
            auto info = tokenizer->Info();
            result["family"] = info.family; result["implementation"] = info.implementation;
            result["vocab_size"] = info.vocabularySize; result["bos_token_id"] = info.bosTokenId;
            result["eos_token_id"] = info.eosTokenId;
        } catch (const std::exception& e) { result = Json{{"error", e.what()}}; }
        std::cout << result.dump() << std::endl;
    }
}
