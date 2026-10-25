// SPDX-License-Identifier: Apache-2.0
// CPU executable. Link and inspect imports before running; never loads CUDA.
#include "qwen38_bf16_prefill128_layout_contract.h"
#include "qwen38_bf16_prefill128_commit.cuh"
#include "qwen38_bf16_low_batch_kv_commit.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
using nlohmann::json;
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
template<class F>void reject(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"Unqualified metadata accepted");}
int main(int argc,char** argv){
    try{
        check(argc==2,"Pass exact CPU fixture JSON");std::ifstream input(argv[1]);check(bool(input),"Missing CPU fixture");json fixture;input>>fixture;
        const auto& candidate=fixture.at("candidate");auto parsed=Garnet::BF16Prefill128KVContract::Parse(candidate);
        check(parsed.batch==1&&parsed.chunk==128&&parsed.target==44564480&&parsed.canonical==106823680,"T128 parsed widths differ");
        reject([&](){Garnet::BF16LowBatchKVContract::Parse(candidate);});
        auto retagged=candidate;retagged["contract"]="garnet_qwen38_bf16_low_batch_delta_study_v1";
        reject([&](){Garnet::BF16LowBatchKVContract::Parse(retagged);});
        int old=0,invalid=0,policies=0;
        for(const auto& layout:fixture.at("qualified")){
            Garnet::BF16LowBatchKVContract::Parse(layout);++old;
            reject([&](){Garnet::BF16Prefill128KVContract::Parse(layout);});
        }
        check(old==2,"Both original T4/T32 contracts required");
        for(const auto& patch:fixture.at("invalid")){
            auto changed=candidate.patch(json::array({patch}));
            reject([&](){Garnet::BF16Prefill128KVContract::Parse(changed);});++invalid;
        }
        for(int batch=-1;batch<=9;++batch)for(int chunk=0;chunk<=256;++chunk){
            check(Garnet::BF16Prefill128CommitProfile(batch,chunk)==(batch==1&&chunk==128),"T128 native profile differs");
            check(Garnet::BF16LowBatchCommitCandidateProfile(batch,chunk)==(batch==1&&(chunk==4||chunk==32)),"Old native profile changed");++policies;
        }
        check(invalid>1000,"Complete metadata mutation inventory missing");
        std::cout<<"{\"status\":\"prefill128_cpu_metadata_passed_gpu_unrun\",\"qualified_old_profiles\":"<<old<<",\"invalid_metadata\":"<<invalid<<",\"native_profile_tuples\":"<<policies<<",\"gpu_executed\":false,\"runtime_qualified\":false}\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
