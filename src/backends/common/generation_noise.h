// SPDX-FileCopyrightText: 2026 CantorAI Inc. and The XLang Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <vector>
namespace Garnet {
// MT19937 state initialization follows XLang3's random module. Drawing a pair
// of 53-bit uniforms then cached Box-Muller matches random.Random(seed).gauss,
// including the order of the sine/cosine values. No global RNG state is used.
class GenerationRandom {
    std::array<uint32_t,624> words_{};
    size_t index_=624;
    uint32_t Next(){
        if(index_==624){
            for(size_t i=0;i<624;++i){
                const uint32_t y=(words_[i]&0x80000000u)|(words_[(i+1)%624]&0x7fffffffu);
                words_[i]=words_[(i+397)%624]^(y>>1)^((y&1u)?0x9908b0dfu:0u);
            }
            index_=0;
        }
        auto y=words_[index_++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;return y^(y>>18);
    }
public:
    explicit GenerationRandom(uint64_t seed){
        const uint32_t keys[]={static_cast<uint32_t>(seed),static_cast<uint32_t>(seed>>32)};
        const size_t count=keys[1]?2:1;
        words_[0]=19650218u;
        for(size_t i=1;i<624;++i)words_[i]=1812433253u*(words_[i-1]^(words_[i-1]>>30))+static_cast<uint32_t>(i);
        size_t i=1,j=0;
        for(size_t k=624;k;--k){
            words_[i]=(words_[i]^((words_[i-1]^(words_[i-1]>>30))*1664525u))+keys[j]+static_cast<uint32_t>(j);
            if(++i==624){words_[0]=words_[623];i=1;}if(++j==count)j=0;
        }
        for(size_t k=623;k;--k){
            words_[i]=(words_[i]^((words_[i-1]^(words_[i-1]>>30))*1566083941u))-static_cast<uint32_t>(i);
            if(++i==624){words_[0]=words_[623];i=1;}
        }
        words_[0]=0x80000000u;
    }
    double Uniform(){const uint32_t a=Next()>>5,b=Next()>>6;return (a*67108864.0+b)*(1.0/9007199254740992.0);}
};

inline X::Value GenerationGaussianNoise(X3PackageHost* host,const X::ARGS& args,const X::KWARGS& kwargs){
    if(args.size()!=2||!kwargs.empty()||!args[0].IsList()||!args[0].Size()||args[0].Size()>32)
        throw X::Error("gaussian_noise requires a positive shape and uint64 seed");
    std::vector<int64_t> shape;uint64_t count=1;
    for(uint64_t i=0;i<args[0].Size();++i){
        const auto value=args[0].Get(i);
        if((!value.IsInt64()&&!value.IsUInt64())||(value.IsUInt64()&&value.ToUInt64()>INT64_MAX))throw X::Error("invalid noise dimension");
        const auto dimension=value.ToLongLong();
        if(dimension<=0||count>static_cast<uint64_t>(INT64_MAX)/sizeof(float)/dimension)throw X::Error("invalid or overflowing noise shape");
        count*=dimension;shape.push_back(dimension);
    }
    const auto text=args[1].ToString();uint64_t seed=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),seed);
    if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size())throw X::Error("noise seed must be an unsigned 64-bit integer");
    auto result=X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape);
    auto use=result.Acquire(X3_TENSOR_WRITE);auto* output=static_cast<float*>(result.Info().data);
    GenerationRandom random(seed);
    constexpr double tau=6.283185307179586476925286766559;
    for(uint64_t i=0;i<count;i+=2){
        const double angle=random.Uniform()*tau;
        const double radius=std::sqrt(-2.0*std::log(1.0-random.Uniform()));
        output[i]=static_cast<float>(std::cos(angle)*radius);
        if(i+1<count)output[i+1]=static_cast<float>(std::sin(angle)*radius);
    }
    return result;
}
}
