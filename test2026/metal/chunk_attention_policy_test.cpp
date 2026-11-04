// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/chunk_attention_policy.h"
#include <iostream>
#include <limits>
using namespace Garnet;
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(){try{
    Need(ChunkAttentionContextBuckets(4096)==std::vector<size_t>({128,256,512,1024,2048,4096}),"4K bucket envelope differs");
    Need(ChunkAttentionContextBuckets(768)==std::vector<size_t>({128,256,512,768}),"Non-power-of-two terminal bucket differs");
    Need(ChunkAttentionContextBuckets(64)==std::vector<size_t>({64}),"Small context envelope differs");
    std::vector<float> slots(2*136),active(2*140);
    for(size_t visible:{1,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096}) {
        std::fill(active.begin(),active.end(),0);slots[0]=float(visible-1);active[0]=1;
        const auto p=SelectChunkAttentionPolicy(4096,2,128,slots.data(),136,active.data(),140);
        const auto buckets=ChunkAttentionContextBuckets(4096);
        Need(p.known&&p.scalar&&p.activeTokens==1&&p.capacity==*std::lower_bound(buckets.begin(),buckets.end(),visible),"Actual slot crossed a compiled bucket incorrectly");
    }
    for(size_t count:{0,1,4,6,8,9,63,64,127,128,129,255,256}) {
        std::fill(active.begin(),active.end(),0);
        for(size_t b=0;b<2;++b)for(size_t t=0;t<128;++t){slots[b*136+t]=float(1920+t);active[b*140+t]=float(b*128+t<count);}
        const auto p=SelectChunkAttentionPolicy(4096,2,128,slots.data(),136,active.data(),140);
        Need(p.known&&p.activeTokens==count&&p.scalar==(count<=8),"Sparse-tail routing differs from actual token count");
    }
    // Inactive future slots may exceed the context, as in native validation.
    active.assign(active.size(),0);slots[0]=8192;
    Need(SelectChunkAttentionPolicy(4096,2,128,slots.data(),136,active.data(),140).scalar,"Inactive future slot rejected");
    const auto unknown=SelectChunkAttentionPolicy(4096,2,128,nullptr,0,nullptr,0);
    Need(!unknown.known&&!unknown.scalar&&unknown.capacity==4096,"Unknown GPU controls narrowed the profile");
    for(float bad:{-1.f,.5f,4096.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        slots[0]=bad;active[0]=1;bool rejected=false;
        try{SelectChunkAttentionPolicy(4096,2,128,slots.data(),136,active.data(),140);}catch(const std::invalid_argument&){rejected=true;}
        Need(rejected,"Invalid active slot was accepted");
    }
    for(float bad:{-1.f,.5f,2.f,std::numeric_limits<float>::quiet_NaN()}) {
        slots[0]=0;active[0]=bad;bool rejected=false;
        try{SelectChunkAttentionPolicy(4096,2,128,slots.data(),136,active.data(),140);}catch(const std::invalid_argument&){rejected=true;}
        Need(rejected,"Invalid active flag was accepted");
    }
    std::cout<<"Compiled context buckets:18 boundaries,13 actual token counts, padded strides, inactive future slots, unknown controls and invalid inputs PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
