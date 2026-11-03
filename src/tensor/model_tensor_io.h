// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include "lowering_context.h"
#ifdef GARNET_METAL_MLX_AFFINE_INT4
#include "mlx_qwen35_checkpoint.h"
#endif
#include <cstring>
#include <limits>
#include <cmath>
#include <fstream>
namespace Garnet {
// Host-side token lookup and state movement. Neural projections belong to the graph.
class ModelTensorIO {
    SafeTensorsIndex index_;
    SafeTensorsMappedFile mapping_;
    const SafeTensorMetadata* embedding_=nullptr;
    size_t embeddingWidth_=0;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    MlxQwen35Checkpoint mlxCheckpoint_;
    AffineInt4Mapping affineEmbedding_;
    bool affineEmbeddingReady_=false;
#endif
    static std::vector<int64_t> Shape(X::Value dimensions) {
        if(!dimensions.IsList()||!dimensions.Size())throw X::Error("shape must be a nonempty list");
        std::vector<int64_t> shape;
        for(uint64_t i=0;i<dimensions.Size();++i){auto d=dimensions.Get(i).ToInt64();
            if(d<=0)throw X::Error("shape dimensions must be positive");shape.push_back(d);}
        return shape;
    }
    static size_t Count(const std::vector<int64_t>& shape) {
        size_t count=1;for(auto d:shape){if(d<=0||uint64_t(d)>SIZE_MAX/count)throw X::Error("tensor shape overflow");count*=size_t(d);}
        if(count>SIZE_MAX/sizeof(float))throw X::Error("tensor byte length overflow");return count;
    }
    static void RequireHost(const X::Tensor& tensor) {
        auto info=tensor.Info();
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||!info.data||!IsContiguousTensor(tensor))
            throw X::Error("expected contiguous host FP32 tensor");
    }
public:
    void Close(){embedding_=nullptr;embeddingWidth_=0;mapping_.Close();
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        affineEmbedding_.Close();affineEmbeddingReady_=false;
#endif
    }
    std::string CheckpointFormat() const {
        if(!embedding_)return "closed";
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        if(affineEmbeddingReady_)return "mlx_qwen35_affine_int4_g64_bf16";
#endif
        return "bf16";
    }
    bool Open(const std::string& path) {
        Close();std::string error;if(!index_.Open(path,error))throw X::Error(error);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        mlxCheckpoint_.Open(index_);
        if(mlxCheckpoint_.Enabled()) {
            const auto metadata=AffineInt4Metadata::Resolve(index_,mlxCheckpoint_,"model.language_model.embed_tokens.weight");
            affineEmbedding_.Open(metadata);embedding_=metadata.tensors[0];
            embeddingWidth_=metadata.layout.columns;affineEmbeddingReady_=true;return true;
        }
#endif
        const auto* entry=index_.Find("model.language_model.embed_tokens.weight");
        if(!entry)entry=index_.Find("model.embed_tokens.weight");
        if(!entry||entry->dataType!="BF16"||entry->shape.size()!=2||!entry->shape[0]||!entry->shape[1]||
            entry->shape[0]>INT64_MAX||entry->shape[1]>INT64_MAX||entry->shape[0]>UINT64_MAX/2/entry->shape[1]||
            entry->shape[0]*entry->shape[1]*2!=entry->dataSize)throw X::Error("invalid BF16 embedding table");
        if(!mapping_.Open(entry->filePath,error))throw X::Error(error);
        if(!mapping_.DataAt(entry->dataOffset,entry->dataSize))throw X::Error("embedding exceeds checkpoint");
        embedding_=entry;embeddingWidth_=size_t(entry->shape[1]);return true;
    }
    X::Value Embedding(X::Value tokens) {
        if(!embedding_||!tokens.IsList()||!tokens.Size())throw X::Error("open checkpoint and supply token IDs");
        const auto width=embeddingWidth_;std::vector<int64_t> shape{int64_t(tokens.Size()),int64_t(width)};
        std::vector<float> values(Count(shape));
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        if(affineEmbeddingReady_) {
            const auto view=affineEmbedding_.View();
            for(uint64_t b=0;b<tokens.Size();++b){const auto id=tokens.Get(b).ToInt64();
                if(id<0||uint64_t(id)>=embedding_->shape[0])throw X::Error("embedding token out of range");
                view.DecodeEmbeddingRow(size_t(id),values.data()+b*width,width);}
            return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
        }
#endif
        const auto* bytes=static_cast<const unsigned char*>(mapping_.DataAt(embedding_->dataOffset,embedding_->dataSize));
        for(uint64_t b=0;b<tokens.Size();++b){auto id=tokens.Get(b).ToInt64();
            if(id<0||uint64_t(id)>=embedding_->shape[0])throw X::Error("embedding token out of range");
            for(uint64_t d=0;d<width;++d){uint16_t bf16;std::memcpy(&bf16,bytes+2*(uint64_t(id)*width+d),2);
                uint32_t bits=uint32_t(bf16)<<16;std::memcpy(&values[b*width+d],&bits,4);}}
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value Zeros(X::Value dimensions) {
        auto shape=Shape(dimensions);std::vector<float> values(Count(shape),0);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value Slice(X::Value source,int64_t start,X::Value dimensions) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();auto shape=Shape(dimensions);
        const auto count=Count(shape),width=count/size_t(shape[0]);
        if(info.rank!=2||start<0||shape[0]!=info.shape[0]||start>info.shape[1]||width>uint64_t(info.shape[1]-start))
            throw X::Error("packed tensor slice shape/bounds invalid");
        auto lease=tensor.Acquire();auto* input=static_cast<const float*>(info.data);std::vector<float> values(count);
        for(size_t b=0;b<size_t(shape[0]);++b)std::memcpy(values.data()+b*width,input+b*info.shape[1]+start,width*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value ReadF32(const std::string& path,X::Value dimensions) {
        auto shape=Shape(dimensions);std::vector<float> values(Count(shape));
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file||file.tellg()!=std::streamoff(values.size()*4))throw X::Error("FP32 file length mismatch");
        file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),values.size()*4);
        if(!file)throw X::Error("FP32 file read failed");
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value StackTokens(X::Value tokens) {
        if(!tokens.IsList()||!tokens.Size())throw X::Error("nonempty token tensor list required");
        X::Tensor first(tokens.Get(uint64_t(0)));RequireHost(first);auto info=first.Info();
        if(info.rank!=2)throw X::Error("token tensors require batch x hidden shape");
        std::vector<int64_t> shape{info.shape[0],int64_t(tokens.Size()),info.shape[1]};std::vector<float> values(Count(shape));
        for(uint64_t i=0;i<tokens.Size();++i){X::Tensor tensor(tokens.Get(i));RequireHost(tensor);auto t=tensor.Info();
            if(t.rank!=2||t.shape[0]!=shape[0]||t.shape[1]!=shape[2])throw X::Error("token tensor stack shape mismatch");
            auto lease=tensor.Acquire();auto data=static_cast<const float*>(t.data);
            for(int64_t b=0;b<shape[0];++b)std::memcpy(values.data()+(b*shape[1]+i)*shape[2],data+b*shape[2],shape[2]*4);}
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value VisualToken(X::Value source,int64_t token) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(info.rank!=3||token<0||token>=info.shape[1])throw X::Error("visual token index/shape invalid");
        auto lease=tensor.Acquire();const auto* input=static_cast<const float*>(info.data);
        std::vector<int64_t> shape{info.shape[0],info.shape[2]};std::vector<float> values(Count(shape));
        for(int64_t b=0;b<info.shape[0];++b)std::memcpy(values.data()+b*info.shape[2],
            input+(b*info.shape[1]+token)*info.shape[2],info.shape[2]*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    // Compatibility with the serving caller: these logits are already projected on GPU.
    X::Value Logits(X::Value source) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(!embedding_||info.rank!=2||uint64_t(info.shape[1])!=embedding_->shape[0])throw X::Error("expected GPU-projected vocabulary logits");
        return source;
    }
    X::Value Argmax(X::Value source) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(info.rank!=2||info.shape[0]<=0||info.shape[1]<=0)throw X::Error("argmax requires nonempty rows");
        auto lease=tensor.Acquire();const auto* data=static_cast<const float*>(info.data);auto result=X::Value::List(Host());
        for(int64_t b=0;b<info.shape[0];++b){int64_t best=0;const auto* row=data+b*info.shape[1];
            for(int64_t d=0;d<info.shape[1];++d){if(!std::isfinite(row[d]))throw X::Error("nonfinite logits");if(row[d]>row[best])best=d;}
            result.Append(X::Value(best));}return result;
    }
    BEGIN_PACKAGE(ModelTensorIO)
        APISET().AddFunc<1>("open",&ModelTensorIO::Open);
        APISET().AddFunc<0>("close",&ModelTensorIO::Close);
        APISET().AddProp("checkpoint_format",&ModelTensorIO::CheckpointFormat);
        APISET().AddFunc<1>("embedding",&ModelTensorIO::Embedding);
        APISET().AddFunc<1>("zeros",&ModelTensorIO::Zeros);
        APISET().AddFunc<3>("slice",&ModelTensorIO::Slice);
        APISET().AddFunc<2>("read_f32",&ModelTensorIO::ReadF32);
        APISET().AddFunc<1>("stack_tokens",&ModelTensorIO::StackTokens);
        APISET().AddFunc<2>("visual_token",&ModelTensorIO::VisualToken);
        APISET().AddFunc<1>("logits",&ModelTensorIO::Logits);
        APISET().AddFunc<1>("argmax",&ModelTensorIO::Argmax);
    END_PACKAGE
};
}
