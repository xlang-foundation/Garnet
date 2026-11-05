// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/affine_profile_index_bounds.h"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
void Need(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
int main() {
    try {
        for(auto shape:std::array<std::array<size_t,2>,9>{{{48,5120},{1024,5120},{5120,6144},
            {5120,17408},{6144,5120},{10240,5120},{12288,5120},{17408,5120},{248320,5120}}})
            for(size_t rows:{1,2,4,17,64,128,256,512})
                Need(AffineProfileIndex32Fits(rows,shape[1],shape[0],shape[1],shape[0]),"Qwen profile unexpectedly exceeds uint32 indexes");
        constexpr size_t maximum=std::numeric_limits<uint32_t>::max();
        Need(AffineProfileIndex32Fits(512,64,48,maximum/512,48),"Valid padded input stride boundary rejected");
        Need(!AffineProfileIndex32Fits(512,64,48,maximum/512+1,48),"Input row-stride product overflows");
        Need(AffineProfileIndex32Fits(512,64,48,64,maximum/512),"Valid padded output stride boundary rejected");
        Need(!AffineProfileIndex32Fits(512,64,48,64,maximum/512+1),"Output row-stride product overflows");
        Need(AffineProfileIndex32Fits(1,64,maximum/8,64,maximum/8),"Valid packed index boundary rejected");
        Need(!AffineProfileIndex32Fits(1,64,maximum/8+1,64,maximum/8+1),"Packed weight indexes overflow");
        for(auto bad:std::array<std::array<size_t,5>,10>{{{0,64,48,64,48},{513,64,48,64,48},
            {1,0,48,64,48},{1,63,48,64,48},{1,65,48,65,48},{1,64,0,64,48},
            {1,64,48,63,48},{1,64,48,64,47},{1,SIZE_MAX,48,SIZE_MAX,48},
            {1,64,SIZE_MAX,64,SIZE_MAX}}})
            Need(!AffineProfileIndex32Fits(bad[0],bad[1],bad[2],bad[3],bad[4]),"Invalid/overflowing profile accepted");
        std::cout<<"Affine uint32 index proof:72 Qwen row/shape profiles, packed/stride boundaries and10 invalid ranges PASS\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
