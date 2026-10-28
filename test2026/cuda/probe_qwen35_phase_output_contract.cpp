// SPDX-License-Identifier: Apache-2.0
// CPU-only contract test. No CUDA API calls or device allocation.
#include "../../src/backends/tensorrt/qwen35_phase_output_contract.h"
#include <filesystem>
#include <fstream>
#include <iostream>
using nlohmann::json;
json normalized(const Garnet::Qwen35PhaseOutputContract& p){
    const auto& c=p.config;json linear=json::array();
    for(const auto& x:p.linear)linear.push_back({x.state_offset,x.payload_offset});
    return {{"batch",c.batch},{"tokens",c.tokens},{"capacity",c.capacity},{"state_width",c.state_width},
        {"packed_width",c.packed_width},{"hidden_width",c.hidden_width},{"output_state_width",c.output_state_width},
        {"status_offset",c.status_offset},{"linear_layers",c.linear_layers},{"full_layers",c.full_layers},
        {"copy_only",bool(c.copy_only)},{"replay_only",p.replay_only},{"first_layer",p.first_layer},{"layer_count",p.layer_count},
        {"linear",linear},{"full",p.full},{"parameter_layers",p.parameter_layers}};
}
int main(int argc,char** argv){
    try{
        if(argc!=3||std::filesystem::exists(argv[2]))throw std::runtime_error("Preserve original CPU probe result; expected fixture/output");
        std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Fixture unavailable");
        json fixture;input>>fixture;json rows=json::array();size_t failed=0,acceptedCount=0,rejectedCount=0;
        for(const auto& test:fixture.at("cases")){
            bool accepted=false;json actual;std::string error;
            try{actual=normalized(Garnet::ParseQwen35PhaseOutputContract(test.at("spec")));accepted=true;}
            catch(const std::exception& ex){error=ex.what();}
            if(accepted)++acceptedCount;else ++rejectedCount;
            bool passed=accepted==test.at("accept").get<bool>()&&(!accepted||actual==test.at("expected"));
            failed+=!passed;rows.push_back({{"name",test.at("name")},{"passed",passed},{"accepted",accepted},{"error",error},{"actual",actual}});
        }
        json result={{"all_passed",failed==0},{"cases",rows},{"accepted",acceptedCount},{"rejected",rejectedCount},
            {"GPU_work_or_model_loaded",false},{"native_GPU_backend_integration_tested",false},{"performance_measured",false}};
        std::ofstream output(argv[2]);output<<result.dump(2)<<'\n';output.close();
        std::cout<<"PHASE_CONTRACT_CPU "<<rows.size()<<" cases accepted="<<acceptedCount<<" rejected="<<rejectedCount<<" failed="<<failed<<'\n';
        return failed?1:0;
    }catch(const std::exception& ex){std::cerr<<"PHASE_CONTRACT_CPU_FATAL "<<ex.what()<<'\n';return 2;}
}
