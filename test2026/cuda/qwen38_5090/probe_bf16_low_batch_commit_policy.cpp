// SPDX-License-Identifier: Apache-2.0
// CPU-only policy controls, no CUDA API use or runtime imports.
#include "bf16_low_batch_commit_candidate.h"
#include <iostream>
#include <stdexcept>
int main(){
    int checked=0;
    for(int batch:{-1,0,1,2,3,4,8})for(int chunk:{-1,0,1,2,4,8,16,32,64}){
        const bool expected=batch==1&&(chunk==4||chunk==32);
        if(Garnet::BF16LowBatchCommitCandidateProfile(batch,chunk)!=expected)
            throw std::runtime_error("B1-only commit shape policy differs");
        ++checked;
    }
    std::cout<<"LOW_BATCH_COMMIT_CPU_POLICY_PASS cases="<<checked<<"\n";
}
