// SPDX-License-Identifier: Apache-2.0
// CPU-only configuration policy; no CUDA API is called or linked.
#include "bf16_kv_low_batch_candidate.h"
#include <iostream>
#include <stdexcept>
int main() {
    using namespace Garnet;
    int checked=0;
    for (int batch : {0,1,2,4,8}) for (int tokens : {0,4,8,32,128}) {
        Qwen35AttentionCoreConfig c{batch,tokens,24,4,4096,256};
        const bool expected=batch==1 && (tokens==4 || tokens==32);
        if (BF16KVLowBatchCandidateProfile(c)!=expected) throw std::runtime_error("Low batch admission differs");
        ++checked;
    }
    for (int tokens : {4,32}) for (int mode=0;mode<4;++mode) {
        Qwen35AttentionCoreConfig c{1,tokens,24,4,4096,256};
        if(mode==0)c.heads=23;if(mode==1)c.kv=3;if(mode==2)c.capacity=4095;if(mode==3)c.dim=128;
        if(BF16KVLowBatchCandidateProfile(c))throw std::runtime_error("Unqualified geometry admitted");++checked;
    }
    std::cout << "LOW_BATCH_CPU_PROFILE_PASS cases=" << checked << "\n";
}
