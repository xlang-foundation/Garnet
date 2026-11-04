// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/chunk_attention_index_bounds.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
std::array<uint64_t,16> Profile(uint64_t b,uint64_t t) {
    return {b,t,24,4,256,4096,6144,1024,1024,4194304,4194304,t,t,6144,6144,1};
}
int main(){try{
    for(auto b:{1,2,4,8,64})for(auto t:{1,2,8,16,64,128,256})Need(ChunkAttentionIndex32Fits(Profile(b,t)),"Valid bounded Qwen attention rejected");
    const uint64_t maximum=std::numeric_limits<uint32_t>::max();
    auto p=Profile(2,128);
    for(auto field:{6,7,8,13,14}) {
        auto good=p;good[field]=maximum/256;Need(ChunkAttentionIndex32Fits(good),"Valid token stride boundary rejected");
        ++good[field];Need(!ChunkAttentionIndex32Fits(good),"Token index span overflow accepted");
    }
    for(auto field:{9,10,11,12}) {
        auto good=p;good[field]=maximum/2;Need(ChunkAttentionIndex32Fits(good),"Valid batch stride boundary rejected");
        ++good[field];Need(!ChunkAttentionIndex32Fits(good),"Batch index span overflow accepted");
    }
    for(size_t field=0;field<16;++field){auto bad=p;bad[field]=UINT64_MAX;Need(!ChunkAttentionIndex32Fits(bad),"Wide field accepted");}
    for(auto field:{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}){auto bad=p;bad[field]=0;Need(!ChunkAttentionIndex32Fits(bad),"Zero dimension/stride accepted");}
    auto bad=p;bad[0]=65;Need(!ChunkAttentionIndex32Fits(bad),"Batch beyond the native row bound accepted");
    bad=p;bad[1]=257;Need(!ChunkAttentionIndex32Fits(bad),"Chunk beyond the native bound accepted");
    bad=p;bad[5]=4097;Need(!ChunkAttentionIndex32Fits(bad),"Context beyond4K accepted");
    bad=p;bad[2]=23;Need(!ChunkAttentionIndex32Fits(bad),"Nonintegral grouped query heads accepted");
    bad=p;bad[15]=2;Need(!ChunkAttentionIndex32Fits(bad),"Invalid gate contract accepted");
    std::cout<<"Chunk attention uint32 bounds:35 Qwen profiles, nine stride boundaries, all16 wide fields and invalid geometry PASS\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
