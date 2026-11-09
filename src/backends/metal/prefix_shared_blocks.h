// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
namespace Garnet {
inline constexpr size_t kPrefixBlockTokens=64;
inline size_t PrefixWordCount(size_t groups,size_t tokens,size_t width) {
    if(!groups||!tokens||!width||width>SIZE_MAX/tokens||groups>SIZE_MAX/(tokens*width))
        throw std::invalid_argument("Invalid dense prefix block geometry");
    return groups*tokens*width;
}
struct DensePrefixBlock {
    const size_t tokens,groups,width;
    const std::vector<float> words;
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
    const bool allFinite;
#endif
    DensePrefixBlock(size_t t,size_t g,size_t w,std::vector<float>&& data)
        :tokens(t),groups(g),width(w),words(std::move(data))
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
        ,allFinite(std::all_of(words.begin(),words.end(),[](float value){return std::isfinite(value);}))
#endif
    {
        if(t>kPrefixBlockTokens||words.size()!=PrefixWordCount(g,t,w))
            throw std::invalid_argument("Dense prefix block shape mismatch");
    }
    size_t Charge() const {
        const size_t overhead=sizeof(DensePrefixBlock)+2*sizeof(void*);
        if(words.capacity()>(SIZE_MAX-overhead)/sizeof(float))throw std::overflow_error("Dense prefix block charge overflow");
        return overhead+words.capacity()*sizeof(float);
    }
};
inline void PrefixRange(const std::vector<float>& data,size_t groups,size_t total,size_t width,size_t first,size_t count) {
    if(!count||count>kPrefixBlockTokens||first>total||count>total-first||data.size()!=PrefixWordCount(groups,total,width))
        throw std::invalid_argument("Dense prefix block range mismatch");
}
inline std::shared_ptr<const DensePrefixBlock> MakePrefixBlock(const std::vector<float>& data,size_t groups,size_t total,size_t width,size_t first,size_t count) {
    PrefixRange(data,groups,total,width,first,count);std::vector<float> result(PrefixWordCount(groups,count,width));
    for(size_t group=0;group<groups;++group)
        std::memcpy(result.data()+group*count*width,data.data()+(group*total+first)*width,count*width*sizeof(float));
    return std::make_shared<const DensePrefixBlock>(count,groups,width,std::move(result));
}
inline bool PrefixBlockMatches(const DensePrefixBlock& block,const std::vector<float>& data,size_t groups,size_t total,size_t width,size_t first,size_t count) {
    PrefixRange(data,groups,total,width,first,count);
    if(block.tokens!=count||block.groups!=groups||block.width!=width)return false;
    for(size_t group=0;group<groups;++group)
        if(std::memcmp(block.words.data()+group*count*width,data.data()+(group*total+first)*width,count*width*sizeof(float)))return false;
    return true;
}
inline void WritePrefixBlock(const DensePrefixBlock& block,std::vector<float>& data,size_t groups,size_t total,size_t width,size_t first,size_t count) {
    PrefixRange(data,groups,total,width,first,count);
    if(block.tokens!=count||block.groups!=groups||block.width!=width)throw std::invalid_argument("Restored dense prefix block shape mismatch");
    for(size_t group=0;group<groups;++group)
        std::memcpy(data.data()+(group*total+first)*width,block.words.data()+group*count*width,count*width*sizeof(float));
}
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
struct DensePrefixCapture {
    std::vector<std::shared_ptr<const DensePrefixBlock>> blocks;
    size_t comparedBytes=0,copiedBytes=0,reusedBlocks=0,newBlocks=0;
};
// The context validates ownership, completion, row span and device before
// lending this pointer for the duration of this call. Nothing borrows it after
// return. Group stride uses capacity, while immutable blocks are compact.
inline DensePrefixCapture CapturePrefixBlocks(const float* source,size_t columns,
    size_t groups,size_t capacity,size_t width,size_t tokens,size_t commonTokens,
    const std::vector<std::shared_ptr<const DensePrefixBlock>>& candidates) {
    if(!source||!tokens||tokens>capacity||commonTokens>tokens||
       columns!=PrefixWordCount(groups,capacity,width)||
       PrefixWordCount(groups,tokens,width)>128*1024*1024/sizeof(float))
        throw std::invalid_argument("Direct dense prefix capture layout/bounds mismatch");
    DensePrefixCapture result;result.blocks.reserve((tokens+kPrefixBlockTokens-1)/kPrefixBlockTokens);
    for(size_t first=0,index=0;first<tokens;first+=kPrefixBlockTokens,++index) {
        const size_t count=std::min(kPrefixBlockTokens,tokens-first);
        std::shared_ptr<const DensePrefixBlock> block;
        if(first+count<=commonTokens&&index<candidates.size()) {
            const auto& old=candidates[index];
            if(old&&old->allFinite&&old->tokens==count&&old->groups==groups&&old->width==width) {
                bool equal=true;
                for(size_t group=0;group<groups;++group) {
                    result.comparedBytes+=count*width*sizeof(float);
                    if(std::memcmp(old->words.data()+group*count*width,
                        source+(group*capacity+first)*width,count*width*sizeof(float))){equal=false;break;}
                }
                if(equal){block=old;++result.reusedBlocks;}
            }
        }
        if(!block) {
            std::vector<float> words(PrefixWordCount(groups,count,width));
            for(size_t group=0;group<groups;++group)
                std::memcpy(words.data()+group*count*width,source+(group*capacity+first)*width,count*width*sizeof(float));
            auto fresh=std::make_shared<const DensePrefixBlock>(count,groups,width,std::move(words));
            if(!fresh->allFinite)throw std::runtime_error("Cannot capture nonfinite direct request-row prefix");
            result.copiedBytes+=fresh->words.size()*sizeof(float);++result.newBlocks;block=std::move(fresh);
        }
        result.blocks.push_back(std::move(block));
    }
    return result;
}
#endif

}
