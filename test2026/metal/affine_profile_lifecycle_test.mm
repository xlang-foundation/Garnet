// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include <iostream>
#include <stdexcept>
using namespace Garnet;
int main() {
    try {
        const auto before=MetalContext::MemoryStats().allocated;
        uint64_t warmedClosed=0;
        for(size_t cycle=0;cycle<3;++cycle) {
            uint64_t afterScope=0;
            @autoreleasepool {
                {
                    MetalContext gpu;
                    const auto layout=AffineInt4Layout::Create(5127,64);
                    std::vector<uint32_t> packed(layout.weightBytes/4,0x76543210u);
                    std::vector<uint16_t> scales(layout.coefficientBytes/2,0x3d00),biases(scales.size(),0xbe00);
                    auto weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,5127,64);
                    auto a=gpu.Upload(std::vector<float>(64,.0625f),1,64),out=gpu.Allocate(1,5127);
                    if(!gpu.PrepareAffineInt4Profile(a,weight,out,128)||gpu.AffineProfilePipelineCount()!=11)
                        throw std::runtime_error("Lifecycle fixture did not prepare every bounded projection route");
                    gpu.Begin();gpu.EncodeAffineInt4MatMul(a,weight,out);gpu.SubmitAndWait();
                    if(gpu.LastAffineProfileDispatches()!=1||gpu.LastAffineProfileMisses()!=0)
                        throw std::runtime_error("Lifecycle fixture missed its prepared pipeline");
                }
                afterScope=MetalContext::MemoryStats().allocated;
            }
            const auto afterPool=MetalContext::MemoryStats().allocated;
            const auto owners=MetalContext::ProfileOwnerStats();
            if(owners.matrices||owners.affineWeights||owners.bf16Weights||owners.pipelineSets)
                throw std::runtime_error("Garnet retained native resource owners after profile context teardown");
            if(cycle==0)warmedClosed=afterPool;
            std::cout<<"affine_profile_lifecycle cycle="<<cycle<<" before="<<before
                <<" after_cpp_scope="<<afterScope<<" after_pool="<<afterPool<<" live_garnet_owners=0\n"<<std::flush;
            if(afterPool!=warmedClosed)throw std::runtime_error("Profile creation/destruction grew the warmed driver allocation baseline");
        }
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
