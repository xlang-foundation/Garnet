// SPDX-License-Identifier: Apache-2.0
// Actual native Core ML calls, independent exact FP16 fixture oracle.
#include "../../src/backends/coreml/coreml_session.h"
#include "../../src/model/float16_codec.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace {
size_t checks=0;
void Require(bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);}
template<class F> void Reject(F call){bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}Require(rejected,"Missing native Core ML validation rejection");}
CoreMLTensor Input(float value){CoreMLTensor t{"x","float16",{1,32,1,16},std::vector<uint8_t>(1024)};uint16_t half=Float32ToFloat16RNE(value);for(size_t i=0;i<512;++i)std::memcpy(t.bytes.data()+2*i,&half,2);return t;}
void Output(const std::vector<CoreMLTensor>& tensors,float expected){Require(tensors.size()==1&&tensors[0].name=="y"&&tensors[0].bytes.size()==1024,"Wrong Core ML output");for(size_t i=0;i<512;++i){uint16_t h;std::memcpy(&h,tensors[0].bytes.data()+2*i,2);Require(Float16ToFloat32(h)==expected,"Native Core ML output differs from independent exact oracle");}}
std::string Profile(const std::string& policy){
    nlohmann::json tensor{{"dtype","float16"},{"maximum_shape",{1,32,1,16}}};
    nlohmann::json profile{{"compute_units",policy},{"maximum_io_bytes",size_t(2048)},
        {"maximum_state_bytes",size_t(1024)},{"require_ane_plan",false}};
    profile["inputs"]["x"]=tensor;profile["outputs"]["y"]=tensor;return profile.dump();
}
}
int main(int argc,char** argv){try{
    Require(argc==2,"Compiled fixture path required");CoreMLSession closed;Reject([&]{closed.Predict({Input(1)});});Reject([&]{closed.Reset();});
    for(const std::string policy:{"cpu","cpu_ane"}){
        CoreMLSession session;auto profile=Profile(policy);session.Open(argv[1],profile);Reject([&]{session.Open(argv[1],profile);});
        Output(session.Predict({Input(2)}),1.f);auto two=session.SnapshotState();Require(two.size()==1&&two[0].name=="accumulator","Missing persistent Core ML state");
        Output(session.Predict({Input(3)}),2.5f);auto five=session.SnapshotState();Require(two[0].bytes!=five[0].bytes,"Core ML snapshot aliases mutable state");
        session.Reset();Output(session.Predict({Input(1)}),.5f);session.RestoreState(five);Require(session.SnapshotState()[0].bytes==five[0].bytes,"State restore changed bits");
        auto invalid=Input(1);invalid.shape[0]=2;Reject([&]{session.Predict({invalid});});invalid=Input(1);invalid.bytes.pop_back();Reject([&]{session.Predict({invalid});});
        Reject([&]{session.Predict({Input(1),Input(1)});});Reject([&]{session.RestoreState({});});auto bad=five;bad[0].dtype="float32";Reject([&]{session.RestoreState(bad);});bad=five;bad[0].bytes.pop_back();Reject([&]{session.RestoreState(bad);});
        Require(session.SnapshotState()[0].bytes==five[0].bytes,"Rejected input/restore changed state");Output(session.Predict({Input(2)}),3.5f);
        {CoreMLSession independent;independent.Open(argv[1],profile);Output(independent.Predict({Input(2)}),1.f);}
        auto report=nlohmann::json::parse(session.ReportJSON());Require(report["compute_units"]==policy&&report["declared_state_bytes"]==1024&&report["predictions"]==4,"Incorrect Core ML report");
        if(report["anticipated_operation_devices"].value("neural_engine",size_t(0))==0){
            CoreMLSession strict;auto requireANE=nlohmann::json::parse(profile);requireANE["require_ane_plan"]=true;
            Reject([&]{strict.Open(argv[1],requireANE.dump());});
        }
        std::cout<<session.ReportJSON()<<'\n';session.Close();session.Close();Reject([&]{session.SnapshotState();});
        auto limits=nlohmann::json::parse(profile);limits["maximum_io_bytes"]=1;Reject([&]{session.Open(argv[1],limits.dump());});limits=nlohmann::json::parse(profile);limits["maximum_state_bytes"]=0;Reject([&]{session.Open(argv[1],limits.dump());});
        limits=nlohmann::json::parse(profile);limits["compute_units"]="ane_only";Reject([&]{session.Open(argv[1],limits.dump());});session.Open(argv[1],profile);Output(session.Predict({Input(2)}),1.f);
    }
    std::cout<<"coreml_state_runtime checks="<<checks<<" passed; anticipated placement only, no ANE execution/performance claim\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
