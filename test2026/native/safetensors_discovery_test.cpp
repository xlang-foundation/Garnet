// SPDX-License-Identifier: Apache-2.0
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs=std::filesystem;
using Json=nlohmann::json;
void Require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
void TensorFile(const fs::path& path,const std::string& name){
    Json metadata={{name,{{"dtype","F32"},{"shape",{2,2}},{"data_offsets",{0,16}}}}};
    auto header=metadata.dump();header.append((8-header.size()%8)%8,' ');
    const uint64_t bytes=header.size();const float values[4]={1,2,3,4};
    std::ofstream output(path,std::ios::binary);
    output.write(reinterpret_cast<const char*>(&bytes),8);output.write(header.data(),header.size());
    output.write(reinterpret_cast<const char*>(values),sizeof(values));Require(bool(output),"fixture write failed");
}
int main(int argc,char** argv){
    try{
        Require(argc==2,"usage: safetensors_discovery_test ARTIFACT_DIRECTORY");
        const fs::path root=fs::absolute(argv[1]);
        for(const std::string prefix:{"model","diffusion_pytorch_model"}){
            const auto single=root/(prefix+"-single");fs::create_directories(single);
            TensorFile(single/(prefix+".safetensors"),"weight");
            Garnet::SafeTensorsIndex index;std::string error;
            Require(index.Open(single,error),error);
            Require(index.TensorCount()==1 && index.TensorBytes()==16 && index.Find("weight"),"single-file discovery failed");
            const auto sharded=root/(prefix+"-sharded");fs::create_directories(sharded);
            TensorFile(sharded/"first.safetensors","first");TensorFile(sharded/"second.safetensors","second");
            const Json manifest={{"weight_map",{{"first","first.safetensors"},{"second","second.safetensors"}}}};
            {std::ofstream output(sharded/(prefix+".safetensors.index.json"));output<<manifest.dump();}
            Require(index.Open(sharded,error),error);
            Require(index.TensorCount()==2 && index.TensorBytes()==32 && index.Find("first") && index.Find("second"),"shard discovery failed");
        }
        Garnet::SafeTensorsIndex index;std::string error;
        const auto empty=root/"empty";fs::create_directories(empty);
        Require(!index.Open(empty,error) && error.find("no supported")!=std::string::npos,"empty-folder diagnosis failed");
        Require(!index.Open(empty/"missing.index.json",error) && error.find("missing or unreadable")!=std::string::npos,"missing-index diagnosis failed");
        {std::ofstream output(empty/"invalid.index.json");output<<"{";}
        Require(!index.Open(empty/"invalid.index.json",error) && error.find("invalid safetensors index")!=std::string::npos,"invalid-index diagnosis failed");
        std::cout<<"Transformers/Diffusers single/sharded safetensors discovery passed\n";
        return 0;
    }catch(const std::exception& failure){std::cerr<<failure.what()<<'\n';return 1;}
}
