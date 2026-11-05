// SPDX-License-Identifier: Apache-2.0
// CPU-only selected-row dequantization from a real pinned checkpoint.
#include "../../src/model/mlx_qwen35_checkpoint.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("usage: mistral_affine_rows CHECKPOINT REQUEST_JSON NEW_OUTPUT_FOLDER");
        Garnet::SafeTensorsIndex index;std::string error;
        if(!index.Open(argv[1],error))throw std::runtime_error(error);
        Garnet::MlxQwen35Checkpoint adapter;adapter.Open(index);
        if(!adapter.DenseMistral())throw std::runtime_error("Mistral affine checkpoint required");
        std::ifstream requestFile(argv[2]);nlohmann::json requests;requestFile>>requests;
        std::filesystem::path folder(argv[3]);
        if(!std::filesystem::create_directory(folder))throw std::runtime_error("new output directory required");
        nlohmann::json records=nlohmann::json::array();size_t number=0;
        for(const auto& request:requests) {
            auto name=request.at("tensor").get<std::string>();
            auto metadata=Garnet::AffineInt4Metadata::Resolve(index,adapter,name);
            Garnet::AffineInt4Mapping mapping;mapping.Open(metadata);auto view=mapping.View();
            auto rows=request.at("rows").get<std::vector<size_t>>();
            std::vector<float> raw(rows.size()*metadata.layout.columns),rounded(raw.size());
            for(size_t i=0;i<rows.size();++i) {
                view.DecodeRow(rows[i],raw.data()+i*metadata.layout.columns,metadata.layout.columns);
                view.DecodeEmbeddingRow(rows[i],rounded.data()+i*metadata.layout.columns,metadata.layout.columns);
            }
            auto rawPath=folder/(std::to_string(number)+"-raw.f32");
            auto roundedPath=folder/(std::to_string(number)+"-rounded.f32");
            std::ofstream a(rawPath,std::ios::binary),b(roundedPath,std::ios::binary);
            a.write(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(float));
            b.write(reinterpret_cast<char*>(rounded.data()),rounded.size()*sizeof(float));
            if(!a||!b)throw std::runtime_error("incomplete output");
            records.push_back({{"tensor",name},{"rows",rows},{"columns",metadata.layout.columns},
                {"dtype",metadata.layout.coefficientType==Garnet::AffineInt4CoefficientType::FP16?"F16":"BF16"},
                {"raw",rawPath.string()},{"rounded",roundedPath.string()}});++number;
        }
        std::ofstream report(folder/"native.json");report<<records.dump(2)<<'\n';
        if(!report)throw std::runtime_error("cannot save native report");
        std::cout<<"mistral_affine_rows matrices="<<number<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
