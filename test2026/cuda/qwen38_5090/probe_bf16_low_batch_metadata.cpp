// SPDX-License-Identifier: Apache-2.0
// Pure CPU metadata parsing; no CUDA/Garnet/TensorRT runtime use.
#include "bf16_low_batch_layout_contract.h"
#include "qwen38_bf16_kv_layout_contract.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
using nlohmann::json;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class Function>void reject(Function function){bool failed=false;try{function();}catch(const std::exception&){failed=true;}require(failed,"Cross-dialect or malformed layout accepted");}
int main(int argc,char** argv){
    try{
        require(argc==2,"Pass a B1 metadata fixture");std::ifstream file(argv[1]);require(bool(file),"Fixture missing");json corpus;file>>corpus;
        const auto& bases=corpus.at("bases");const auto& old=corpus.at("legacy_bases");require(bases.size()==2&&old.size()==4,"Metadata base inventory differs");int valid=0,rejected=0,cross=0;
        for(const auto& base:bases){
            auto p=Garnet::BF16LowBatchKVContract::Parse(base);
            require(p.batch==1&&p.chunk==(valid?32:4)&&p.canonical==106823680&&p.replay==39714816&&p.target==(valid?40927232:43030528),"B1 metadata geometry differs");++valid;
            reject([&](){Garnet::BF16KVContract::Parse(base);});++cross;
            auto tagged=base;tagged["contract"]="garnet_qwen38_bf16_kv_delta_study_v1";
            reject([&](){Garnet::BF16KVContract::Parse(tagged);});++cross;
        }
        int legacyValid=0;
        for(const auto& base:old){
            auto p=Garnet::BF16KVContract::Parse(base);require(p.batch==(legacyValid<2?4:8)&&p.chunk==(legacyValid%2?32:4),"Legacy geometry differs");++legacyValid;
            reject([&](){Garnet::BF16LowBatchKVContract::Parse(base);});++cross;
            auto tagged=base;tagged["contract"]="garnet_qwen38_bf16_low_batch_delta_study_v1";
            reject([&](){Garnet::BF16LowBatchKVContract::Parse(tagged);});++cross;
        }
        for(const auto& test:corpus.at("mutations")){
            // Invalid JSON patches fail here instead of masquerading as parser rejection.
            auto candidate=bases.at(test.at("base").get<size_t>()).patch(json::array({test.at("patch")}));
            reject([&](){Garnet::BF16LowBatchKVContract::Parse(candidate);});++rejected;
        }
        require(valid==2&&legacyValid==4&&cross==12&&rejected==int(corpus.at("mutations").size())&&rejected>4000,"CPU metadata coverage missing");
        std::cout<<"{\"status\":\"b1_metadata_cpu_contract_passed\",\"b1_profiles\":"<<valid<<",\"legacy_profiles\":"<<legacyValid<<",\"cross_dialect_rejections\":"<<cross<<",\"invalid_layouts\":"<<rejected<<",\"gpu_used\":false,\"model_qualified\":false}\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
