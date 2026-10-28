// SPDX-License-Identifier: Apache-2.0
// CPU only; no Garnet runtime, CUDA API, TensorRT DLL or model loading.
#include "qwen38_bf16_kv_layout_contract.h"
#include <fstream>
#include <cstdio>
#include <stdexcept>
using nlohmann::json;
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv){
    try{
        require(argc==2,"Pass metadata fixture path");std::ifstream file(argv[1]);require(bool(file),"Cannot open fixture");json corpus;file>>corpus;
        const auto& bases=corpus.at("bases");require(bases.size()==4,"Exactly four metadata profiles required");int valid=0,rejected=0;
        for(const auto& base:bases){
            const auto p=Garnet::BF16KVContract::Parse(base);
            require(p.batch==(valid<2?4:8)&&p.chunk==(valid%2==0?4:32)&&p.canonical==106823680&&p.replay==39714816&&
                    p.target==(p.chunk==4?43030528:40927232),"Independent valid profile geometry differs");++valid;
        }
        for(const auto& test:corpus.at("mutations")){
            // Patch application is outside the rejection try/catch: an invalid
            // test fixture must never masquerade as a metadata rejection.
            const auto candidate=bases.at(test.at("base").get<size_t>()).patch(json::array({test.at("patch")}));
            bool failed=false;try{(void)Garnet::BF16KVContract::Parse(candidate);}catch(const std::exception&){failed=true;}
            require(failed,"Malformed metadata accepted");++rejected;
        }
        require(rejected==int(corpus.at("mutations").size())&&rejected>4000,"Rejection coverage unexpectedly small");
        std::printf("{\"status\":\"bf16_kv_metadata_cpu_contract_passed\",\"profiles\":%d,\"invalid_layouts\":%d,\"gpu_used\":false,\"model_qualified\":false}\n",valid,rejected);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
