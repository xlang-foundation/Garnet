// SPDX-License-Identifier: Apache-2.0
// Native, policy-driven FP32-to-BF16 checkpoint preparation.
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
namespace fs=std::filesystem;
using Json=nlohmann::json;
bool Keep(const std::string& name,const std::vector<std::string>& modules){
    for(const auto& segment:modules){
        for(size_t position=name.find(segment);position!=std::string::npos;position=name.find(segment,position+1)){
            if((position==0 || name[position-1]=='.') &&
                (position+segment.size()==name.size() || name[position+segment.size()]=='.'))return true;
        }
    }
    return false;
}
int main(int argc,char** argv){
    try{
        if(argc!=4 && argc!=5)throw std::runtime_error("garnet_prepare_precision SOURCE_COMPONENT OUTPUT_COMPONENT POLICY_JSON [--plan]");
        const bool plan=argc==5;
        if(plan && std::string(argv[4])!="--plan")throw std::runtime_error("unknown preparation option");
        Json policy;{std::ifstream input(argv[3]);if(!input)throw std::runtime_error("precision policy missing");input>>policy;}
        if(policy.at("target_dtype")!="BF16")throw std::runtime_error("native preparer currently supports BF16 output");
        auto modules=policy.at("keep_fp32_modules").get<std::vector<std::string>>();
        for(const auto& module:modules)if(module.empty())throw std::runtime_error("precision policy module names cannot be empty");
        fs::path source=fs::absolute(argv[1]),destination=fs::absolute(argv[2]);
        if(source==destination)throw std::runtime_error("source and destination must differ");
        Garnet::SafeTensorsIndex index;std::string error;if(!index.Open(source,error))throw std::runtime_error(error);
        // Explicit mappings also permit splitting a monolithic checkpoint into
        // named model components without loading its complete payload into RAM.
        std::map<std::string,std::string> names;const auto selected=policy.value("tensor_map",Json());
        if(!selected.is_null()&&!selected.is_object())throw std::runtime_error("tensor_map must be an object");
        std::map<std::string,bool> outputs;
        for(const auto& item:index.Entries()){
            if(!selected.is_null()&&!selected.contains(item.first))continue;
            const auto target=selected.is_null()?item.first:selected.at(item.first).get<std::string>();
            if(target.empty()||target=="__metadata__"||!outputs.emplace(target,true).second)throw std::runtime_error("invalid or duplicate output tensor name");
            names.emplace(item.first,target);
        }
        if(names.empty()||(!selected.is_null()&&names.size()!=selected.size()))throw std::runtime_error("tensor_map is empty or references a missing source tensor");
        const auto configSource=policy.contains("config_file")?fs::path(policy.at("config_file").get<std::string>()):source/"config.json";
        Json config;{std::ifstream input(configSource);if(!input)throw std::runtime_error("source config missing");input>>config;}
        uint64_t plannedBytes=0;size_t plannedConvert=0,plannedKeep=0;
        for(const auto& item:index.Entries()){
            if(!names.count(item.first))continue;
            bool convert=item.second.dataType=="F32" && !Keep(item.first,modules);
            plannedBytes+=convert?item.second.dataSize/2:item.second.dataSize;
            if(convert)++plannedConvert;else ++plannedKeep;
        }
        if(plan){std::cout<<Json{{"converted_tensors",plannedConvert},{"retained_tensors",plannedKeep},{"tensor_bytes",plannedBytes},{"policy",policy}}.dump(2)<<std::endl;return 0;}
        if(fs::exists(destination) && (!fs::is_directory(destination) || !fs::is_empty(destination)))throw std::runtime_error("output component directory must be empty");
        fs::create_directories(destination);
        std::map<fs::path,std::vector<std::pair<std::string,Garnet::SafeTensorMetadata>>> files;
        for(const auto& item:index.Entries())if(names.count(item.first))files[item.second.filePath].push_back(item);
        Json weightMap=Json::object();uint64_t total=0;size_t convertedCount=0,keptCount=0;
        for(auto& group:files){
            auto& entries=group.second;
            std::sort(entries.begin(),entries.end(),[](auto& a,auto& b){return a.second.dataOffset<b.second.dataOffset;});
            Json header={{"__metadata__",{{"format","pt"},{"garnet_precision","mixed-bf16"}}}};
            uint64_t offset=0;
            for(const auto& [name,meta]:entries){
                const bool convert=meta.dataType=="F32" && !Keep(name,modules);
                const auto size=convert?meta.dataSize/2:meta.dataSize;
                header[names.at(name)]={{"dtype",convert?"BF16":meta.dataType},{"shape",meta.shape},{"data_offsets",{offset,offset+size}}};
                weightMap[names.at(name)]=group.first.filename().string();offset+=size;
                if(convert)++convertedCount;else ++keptCount;
            }
            auto text=header.dump();text.append((8-text.size()%8)%8,' ');const uint64_t headerBytes=text.size();
            Garnet::SafeTensorsMappedFile mapping;if(!mapping.Open(group.first,error))throw std::runtime_error(error);
            std::ofstream output(destination/group.first.filename(),std::ios::binary);
            output.write(reinterpret_cast<const char*>(&headerBytes),8);output.write(text.data(),text.size());
            for(const auto& [name,meta]:entries){
                auto* data=static_cast<const unsigned char*>(mapping.DataAt(meta.dataOffset,meta.dataSize));
                if(!data)throw std::runtime_error("source tensor out of range");
                if(meta.dataType=="F32" && !Keep(name,modules)){
                    constexpr size_t chunkWords=1<<20;std::vector<uint16_t> converted(chunkWords);
                    for(uint64_t start=0;start<meta.dataSize/4;start+=chunkWords){
                        const size_t count=std::min<uint64_t>(chunkWords,meta.dataSize/4-start);
                        for(size_t i=0;i<count;++i){
                            uint32_t bits;std::memcpy(&bits,data+(start+i)*4,4);
                            float value;std::memcpy(&value,&bits,4);if(!std::isfinite(value))throw std::runtime_error("nonfinite checkpoint weight: "+name);
                            bits+=0x7fff+((bits>>16)&1);converted[i]=uint16_t(bits>>16);
                            if((converted[i]&0x7fff)==0x7f80)throw std::runtime_error("BF16 conversion overflow: "+name);
                        }
                        output.write(reinterpret_cast<const char*>(converted.data()),count*2);
                    }
                }else output.write(reinterpret_cast<const char*>(data),meta.dataSize);
                if(!output)throw std::runtime_error("checkpoint output failed");
            }
            total+=offset;std::cout<<"Prepared "<<group.first.filename().string()<<" tensor_bytes "<<offset<<std::endl;
        }
        {std::ofstream output(destination/"diffusion_pytorch_model.safetensors.index.json");output<<Json{{"metadata",{{"total_size",total}}},{"weight_map",weightMap}}.dump(2);}
        config["torch_dtype"]="bfloat16";
        if(config.contains("dtype"))config["dtype"]="bfloat16";
        config["garnet_precision_policy"]={{"weight_dtype","bfloat16"},{"keep_fp32_modules",modules}};
        {std::ofstream output(destination/"config.json");output<<config.dump(2);}
        {std::ofstream output(destination/"precision-preparation.json");output<<Json{{"source_component",source.string()},{"policy",policy},{"converted_tensors",convertedCount},{"retained_tensors",keptCount},{"tensor_bytes",total}}.dump(2);}
        std::cout<<"Converted tensors "<<convertedCount<<" retained tensors "<<keptCount<<" total tensor bytes "<<total<<std::endl;
    }catch(const std::exception& failure){std::cerr<<failure.what()<<std::endl;return 1;}
}
