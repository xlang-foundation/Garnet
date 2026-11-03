// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../src/model/qwen_speech_sampling.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace
{
    void Require(bool condition, const char* reason)
    {
        if (!condition) throw std::runtime_error(reason);
    }
}

int main()
{
    constexpr int eos = 2150;
    std::vector<float> logits(3072, -10.0F);
    logits[1] = 8.0F;
    logits[2] = -8.0F;
    logits[2148] = 1000.0F; // Codec padding must never win sampling.
    logits[2149] = 999.0F;  // Neither may codec BOS.
    logits[eos] = 900.0F;
    Require(Garnet::PrepareQwenSpeechLogits(logits, eos, {}, 2.0F), "valid logits rejected");
    Require(std::max_element(logits.begin(), logits.end()) - logits.begin() == 1,
        "reserved token or premature EOS can be selected");
    for (int token = 2048; token < 3072; ++token)
        Require(logits[token] == -std::numeric_limits<float>::infinity(), "reserved token not suppressed");

    logits.assign(3072, 0.0F);
    logits[1] = 8.0F; logits[2] = -8.0F; logits[3] = 7.0F; logits[eos] = 9.0F;
    Require(Garnet::PrepareQwenSpeechLogits(logits, eos, {1, 1, 1, 2, 2, -1, 9999}, 2.0F),
        "history rejected");
    Require(logits[1] == 4.0F && logits[2] == -16.0F,
        "duplicate history compounded repetition penalty");
    Require(logits[3] == 7.0F && logits[eos] == 9.0F, "unseen token or permitted EOS changed");
    Require(std::max_element(logits.begin(), logits.end()) - logits.begin() == eos,
        "EOS cannot terminate a completed utterance");

    logits.assign(3072, 0.0F); logits[eos] = 10.0F;
    Require(Garnet::PrepareQwenSpeechLogits(logits, eos, {1}, 1.0F), "one frame rejected");
    Require(!std::isfinite(logits[eos]), "EOS admitted before two frames");
    logits.assign(3072, 0.0F); logits[0] = std::numeric_limits<float>::quiet_NaN();
    Require(!Garnet::PrepareQwenSpeechLogits(logits, eos, {}, 1.05F), "NaN logit admitted");
    logits.assign(3072, 0.0F);
    Require(!Garnet::PrepareQwenSpeechLogits(logits, 3072, {}, 1.05F), "invalid EOS admitted");
    Require(!Garnet::PrepareQwenSpeechLogits(logits, eos, {}, 0.0F), "invalid penalty admitted");
    std::vector<float> empty;
    Require(!Garnet::PrepareQwenSpeechLogits(empty, eos, {}, 1.05F), "empty vocabulary admitted");
    std::mt19937_64 first(0),same(0),different(1);
    const auto noise=Garnet::MakeQwenTTSPredictorNoise(first);
    Require(noise==Garnet::MakeQwenTTSPredictorNoise(same),"same seed does not reproduce predictor entropy");
    Require(noise!=Garnet::MakeQwenTTSPredictorNoise(different),"different requests shared predictor entropy");
    for(const auto value:noise)Require(std::isfinite(value),"predictor entropy contains a nonfinite value");
    Require(noise!=Garnet::MakeQwenTTSPredictorNoise(first),"entropy repeated on the next audio frame");
    // Gumbel-max must sample categorical probabilities, not always argmax.
    std::mt19937_64 distribution(23);
    int counts[3]{};const double masses[3]={0.6,0.3,0.1};
    for(int iteration=0;iteration<12000;++iteration){
        const float values[3]={Garnet::NextQwenTTSGumbel(distribution),Garnet::NextQwenTTSGumbel(distribution),Garnet::NextQwenTTSGumbel(distribution)};
        int selected=0;double best=std::log(masses[0])+values[0];
        for(int index=1;index<3;++index){const double score=std::log(masses[index])+values[index];if(score>best){best=score;selected=index;}}
        ++counts[selected];
    }
    for(int index=0;index<3;++index)Require(std::abs(counts[index]/12000.0-masses[index])<0.025,"native predictor RNG is not categorical");
    std::cout << "Qwen speech reserved tokens, EOS and unique repetition contracts passed\n";
}
