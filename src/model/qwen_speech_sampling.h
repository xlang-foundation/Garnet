// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

namespace Garnet
{
    constexpr int QwenTTSPredictorGroups = 15;
    constexpr int QwenTTSPredictorTopK = 50;
    constexpr int QwenTTSPredictorVocab = 2048;
    constexpr float QwenTTSPredictorTemperature = 0.9F;
    using QwenTTSPredictorNoise = std::vector<float>;
    inline float NextQwenTTSGumbel(std::mt19937_64& random) {
        const double uniform=(std::max)((std::numeric_limits<double>::min)(),
            (std::min)(std::nextafter(1.0,0.0),std::generate_canonical<double,53>(random)));
        return static_cast<float>(-std::log(-std::log(uniform)));
    }
    inline QwenTTSPredictorNoise MakeQwenTTSPredictorNoise(std::mt19937_64& random) {
        QwenTTSPredictorNoise result(QwenTTSPredictorGroups * QwenTTSPredictorVocab);
        for (auto& value : result) {
            value=NextQwenTTSGumbel(random);
        }
        return result;
    }
    // Match Qwen3-TTS generate(): the final 1024 codec IDs are reserved,
    // except EOS, and EOS is suppressed for the first two generated frames.
    // Repetition penalty follows Transformers' once-per-token semantics.
    inline bool PrepareQwenSpeechLogits(std::vector<float>& logits,
        int eosToken, const std::vector<long long>& history, float penalty)
    {
        constexpr size_t reservedTokens = 1024;
        if (logits.size() <= reservedTokens || eosToken < 0 ||
            static_cast<size_t>(eosToken) >= logits.size() ||
            !std::isfinite(penalty) || penalty < 1.0F) return false;

        std::vector<bool> seen(logits.size(), false);
        for (const auto token : history) {
            if (token >= 0 && static_cast<size_t>(token) < seen.size())
                seen[static_cast<size_t>(token)] = true;
        }
        const auto speechTokens = logits.size() - reservedTokens;
        for (size_t token = 0; token < logits.size(); ++token) {
            auto& value = logits[token];
            if ((token >= speechTokens && static_cast<int>(token) != eosToken) ||
                (static_cast<int>(token) == eosToken && history.size() < 2)) {
                value = -std::numeric_limits<float>::infinity();
                continue;
            }
            if (!std::isfinite(value)) return false;
            if (seen[token]) value = value < 0.0F ? value * penalty : value / penalty;
        }
        return true;
    }
}
