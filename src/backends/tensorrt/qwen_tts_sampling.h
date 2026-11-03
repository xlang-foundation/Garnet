// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Garnet {
// Gumbel-max samples categorical softmax(logits / temperature) within top-k.
// Entropy is an explicit input from the native request/frame RNG.
inline nvinfer1::ITensor* LowerQwenTTSSample(nvinfer1::INetworkDefinition& network,
    std::deque<std::vector<float>>& constants, nvinfer1::ITensor* logits,
    nvinfer1::ITensor* gumbels, int group, int topK, float temperature) {
    using namespace nvinfer1;
    auto checked=[](ILayer* layer) -> ITensor* {
        if (!layer || !layer->getOutput(0)) throw std::runtime_error("TTS sampling layer construction failed");
        return layer->getOutput(0);
    };
    if (!logits || !gumbels) throw std::invalid_argument("TTS sampling inputs missing");
    const auto shape=logits->getDimensions(),noise=gumbels->getDimensions();
    if (shape.nbDims!=3 || shape.d[0]<=0 || shape.d[1]!=1 || shape.d[2]<=0 ||
        noise.nbDims!=3 || noise.d[0]!=shape.d[0] || noise.d[1]<=0 || noise.d[1]>16 ||
        group<0 || group>=noise.d[1] || topK<=0 || topK>256 || topK>shape.d[2] || noise.d[2]!=shape.d[2] ||
        !std::isfinite(temperature) || temperature<=0 || gumbels->getType()!=DataType::kFLOAT)
        throw std::invalid_argument("TTS sampling requires [B,1,V] logits and [B,G,V] FP32 entropy");
    logits=checked(network.addCast(*logits,DataType::kFLOAT));
    auto* top=network.addTopK(*logits,TopKOperation::kMAX,topK,1U<<2);
    auto* values=checked(top);
    auto* threshold=checked(network.addSlice(*values,Dims{3,{0,0,topK-1}},
        Dims{3,{shape.d[0],1,1}},Dims{3,{1,1,1}}));
    // Match reference top-k threshold semantics, including all boundary ties.
    auto* excluded=checked(network.addElementWise(*logits,*threshold,ElementWiseOperation::kLESS));
    constants.push_back({temperature});
    auto* scale=checked(network.addConstant(Dims{3,{1,1,1}},
        Weights{DataType::kFLOAT,constants.back().data(),1}));
    values=checked(network.addElementWise(*logits,*scale,ElementWiseOperation::kDIV));
    auto* entropy=checked(network.addSlice(*gumbels,Dims{3,{0,group,0}},
        Dims{3,{shape.d[0],1,shape.d[2]}},Dims{3,{1,1,1}}));
    auto* scores=checked(network.addElementWise(*values,*entropy,ElementWiseOperation::kSUM));
    constants.push_back({-std::numeric_limits<float>::infinity()});
    auto* forbidden=checked(network.addConstant(Dims{3,{1,1,1}},
        Weights{DataType::kFLOAT,constants.back().data(),1}));
    scores=checked(network.addSelect(*excluded,*forbidden,*scores));
    auto* selected=network.addTopK(*scores,TopKOperation::kMAX,1,1U<<2);
    checked(selected);
    auto* token=checked(network.addCast(*selected->getOutput(1),DataType::kINT64));
    auto* squeeze=network.addShuffle(*token);checked(squeeze);
    squeeze->setReshapeDimensions(Dims{2,{shape.d[0],1}});
    return squeeze->getOutput(0);
}
}
