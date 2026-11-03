// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "affine_int4_codec.h"
#include "mlx_vision_patch.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <array>
#include <fstream>
#include <string>

namespace Garnet {
// Explicit checkpoint representation adapter. Canonical xModel tensor/operator
// names stay unchanged; packed dimensions never masquerade as logical shapes.
class MlxQwen35Checkpoint {
    bool enabled_=false;
public:
    void Open(const SafeTensorsIndex& index) {
        enabled_=false;
        const auto* embedding=index.Find("language_model.model.embed_tokens.weight");
        if(!embedding)return;
        if(index.Find("model.language_model.embed_tokens.weight"))
            throw std::invalid_argument("Ambiguous original/MLX Qwen checkpoint roots");
        std::ifstream stream(index.FilePath().parent_path()/"config.json");
        nlohmann::json config;
        if(!stream)throw std::invalid_argument("MLX Qwen checkpoint requires config.json");
        stream>>config;
        if(config.value("model_type","")!="qwen3_5"||!config.contains("quantization"))
            throw std::invalid_argument("Unsupported MLX checkpoint architecture/quantization");
        const auto& quant=config.at("quantization");
        if(quant.value("bits",0)!=4||quant.value("group_size",0)!=64||quant.value("mode","")!="affine")
            throw std::invalid_argument("Native MLX Qwen profile requires affine INT4/group64");
        const auto* conv=index.Find("language_model.model.layers.0.linear_attn.conv1d.weight");
        if(!conv||conv->shape.size()!=3||conv->shape[0]<=0||conv->shape[1]!=4||conv->shape[2]!=1)
            throw std::invalid_argument("MLX Qwen conversion layout is not verified");
        if(embedding->dataType!="U32")
            throw std::invalid_argument("MLX Qwen affine profile requires packed embeddings");
        enabled_=true;
    }
    bool Enabled() const {return enabled_;}
    std::string PhysicalName(const std::string& canonical) const {
        if(!enabled_)return canonical;
        const std::string prefix="model.language_model.";
        if(canonical.compare(0,prefix.size(),prefix)==0)
            return "language_model.model."+canonical.substr(prefix.size());
        if(canonical.compare(0,8,"lm_head.")==0)return "language_model."+canonical;
        if(canonical.compare(0,13,"model.visual.")==0)
            return "vision_tower."+canonical.substr(13);
        return canonical;
    }
    const SafeTensorMetadata* Find(const SafeTensorsIndex& index,const std::string& name) const {
        return index.Find(PhysicalName(name));
    }
    bool ConvertedConvolution(const std::string& name,const SafeTensorMetadata& metadata) const {
        const std::string suffix=".linear_attn.conv1d.weight";
        return enabled_&&name.size()>=suffix.size()&&
            name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0&&
            metadata.shape.size()==3&&metadata.shape[1]==4&&metadata.shape[2]==1;
    }
    bool ConvertedVisionPatch(const std::string& name,const SafeTensorMetadata& metadata) const {
        if(!enabled_||name!="model.visual.patch_embed.proj.weight")return false;
        if(metadata.dataType!="BF16"||metadata.shape.size()!=5||metadata.shape[0]<=0||
           metadata.shape[1]!=2||metadata.shape[2]!=16||metadata.shape[3]!=16||metadata.shape[4]!=3)
            throw std::invalid_argument("Unsupported converted Qwen vision patch layout");
        return true;
    }
};

struct AffineInt4Metadata {
    AffineInt4Layout layout;
    std::array<const SafeTensorMetadata*,3> tensors{};

    static AffineInt4Metadata Resolve(const SafeTensorsIndex& index,
                                     const MlxQwen35Checkpoint& adapter,const std::string& name) {
        const std::string suffix=".weight";
        if(!adapter.Enabled()||name.size()<suffix.size()||
           name.compare(name.size()-suffix.size(),suffix.size(),suffix)!=0)
            throw std::invalid_argument("Affine INT4 requires an explicit converted checkpoint");
        const std::string base=name.substr(0,name.size()-suffix.size());
        AffineInt4Metadata result;
        result.tensors={adapter.Find(index,name),adapter.Find(index,base+".scales"),
                        adapter.Find(index,base+".biases")};
        const auto* packed=result.tensors[0];
        if(!packed||packed->dataType!="U32"||packed->shape.size()!=2||
           packed->shape[0]<=0||packed->shape[1]<=0||uint64_t(packed->shape[1])>SIZE_MAX/8||
           packed->shape[1]>INT64_MAX/8||
           uint64_t(packed->shape[0])>SIZE_MAX)
            throw std::invalid_argument("Invalid affine INT4 packed matrix");
        result.layout=AffineInt4Layout::Create(size_t(packed->shape[0]),size_t(packed->shape[1])*8);
        if(packed->dataSize!=result.layout.weightBytes)
            throw std::invalid_argument("Affine INT4 packed byte count mismatch");
        for(size_t i=1;i<3;++i) {
            const auto* coefficient=result.tensors[i];
            if(!coefficient||coefficient->dataType!="BF16"||coefficient->shape.size()!=2||
               coefficient->shape[0]!=packed->shape[0]||
               uint64_t(coefficient->shape[1])!=result.layout.groupsPerRow||
               coefficient->dataSize!=result.layout.coefficientBytes)
                throw std::invalid_argument("Affine INT4 BF16 scale/bias contract mismatch");
        }
        return result;
    }
};

// Mappings own the bytes for their complete use interval. Embedding lookup
// decodes only selected rows; this helper never expands a complete matrix.
class AffineInt4Mapping {
    std::array<SafeTensorsMappedFile,3> mappings_;
    std::array<const void*,3> data_{};
    AffineInt4Metadata metadata_;
public:
    void Close() {data_={};metadata_={};for(auto& mapping:mappings_)mapping.Close();}
    void Open(const AffineInt4Metadata& metadata) {
        Close();
        try {
            for(size_t i=0;i<3;++i) {
                const auto& tensor=*metadata.tensors[i];std::string error;
                if(!mappings_[i].Open(tensor.filePath,error))throw std::runtime_error(error);
                data_[i]=mappings_[i].DataAt(tensor.dataOffset,tensor.dataSize);
                if(!data_[i])throw std::invalid_argument("Affine INT4 tensor exceeds checkpoint bounds");
            }
            metadata_=metadata;
        }catch(...){Close();throw;}
    }
    const void* Data(size_t index) const {
        if(index>=3||!data_[index])throw std::logic_error("Affine INT4 mapping is not open");
        return data_[index];
    }
    AffineInt4View View() const {
        return AffineInt4View(metadata_.layout,Data(0),metadata_.layout.weightBytes,
                             Data(1),metadata_.layout.coefficientBytes,Data(2),metadata_.layout.coefficientBytes);
    }
};
}
