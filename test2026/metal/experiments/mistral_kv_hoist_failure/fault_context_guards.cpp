// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
int main(){try{{MetalContext gpu;auto need=[](bool b){if(!b)throw std::runtime_error("Fault arming guard failure");};
 auto witness=[&](){return nlohmann::json::parse(gpu.KVHoistTestFaultWitnessJson());};
 gpu.ArmKVHoistTestFault(1);need(witness()["armed"]==1);
 bool invalid=false;try{gpu.ArmKVHoistTestFault(3);}catch(const std::invalid_argument&){invalid=true;}need(invalid&&witness()["armed"]==1);
 auto a=gpu.Upload({1,2,3,4},1,4),zero=gpu.Upload({0,0,0,0},1,4),out=gpu.Allocate(1,4);
 gpu.Begin();bool open=false;try{gpu.ArmKVHoistTestFault(2);}catch(const std::logic_error&){open=true;}need(open&&witness()["armed"]==1);
 gpu.EncodeAdd(a,zero,out);gpu.Flush();bool partial=false;try{gpu.ArmKVHoistTestFault(2);}catch(const std::logic_error&){partial=true;}need(partial&&witness()["armed"]==1);
 gpu.Abort();gpu.DisarmKVHoistTestFault();need(witness()["armed"]==0);
 gpu.Begin();gpu.EncodeAdd(a,zero,out);gpu.SubmitAndWait();need(gpu.Download(out)==std::vector<float>({1,2,3,4}));
 std::cout<<"Fault arming invalid/open/partly-submitted preservation, disarm and context abort/reuse PASS\n";
 }auto owners=MetalContext::ProfileOwnerStats();if(owners.matrices||owners.affineWeights||owners.bf16Weights||owners.pipelineSets)throw std::runtime_error("Fault guard native owners retained");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
