// SPDX-License-Identifier: Apache-2.0
#include "compiled_graph.h"
#include "float16_codec.h"
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
#include "../../runtime/memory/scratch_alias_group_plan.h"
#include <type_traits>
#endif
#include "graph_capture.h"
#include "lowering_context.h"
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
#include "../../runtime/memory/scratch_lifetime_plan.h"
#endif
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <map>
#include <cstdlib>
#include <cstdio>
namespace Garnet {
const SafeTensorMetadata* MetalCompiledGraph::FindWeightMetadata(const std::string& name) const {
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    return mlxCheckpoint_.Find(index_,name);
#else
    return index_.Find(name);
#endif
}
#ifdef GARNET_METAL_MLX_AFFINE_INT4
size_t MetalCompiledGraph::AffineInt4Weight(const std::string& name) {
    auto found=weights_.find(name);if(found!=weights_.end())return found->second;
    const auto metadata=AffineInt4Metadata::Resolve(index_,mlxCheckpoint_,name);
    const auto& layout=metadata.layout;
#ifdef GARNET_METAL_AFFINE_FP16_COEFFICIENTS
    if(layout.coefficientType!=AffineInt4CoefficientType::FP16)
        throw std::invalid_argument("This Metal affine engine requires FP16 coefficients; select the BF16 engine profile for this checkpoint");
#else
    if(layout.coefficientType!=AffineInt4CoefficientType::BF16)
        throw std::invalid_argument("This Metal affine engine supports BF16 coefficients only; an FP16 kernel profile is required");
#endif
    std::vector<CheckpointDependency> dependencies;
    for(const auto* tensor:metadata.tensors)
        dependencies.push_back({*tensor,std::filesystem::last_write_time(tensor->filePath),std::filesystem::file_size(tensor->filePath)});
    auto unchanged=[](const CheckpointDependency& a,const CheckpointDependency& b){
        return a.metadata.filePath==b.metadata.filePath&&a.metadata.dataType==b.metadata.dataType&&
            a.metadata.shape==b.metadata.shape&&a.metadata.dataOffset==b.metadata.dataOffset&&
            a.metadata.dataSize==b.metadata.dataSize&&a.modified==b.modified&&a.bytes==b.bytes;
    };
    auto shared=sharedWeights_.find(name);
    if(shared!=sharedWeights_.end()) {
        const auto& record=shared->second;
        if(!record.slot.affineInt4.Valid()||record.dependencies.size()!=dependencies.size())
            throw std::runtime_error("Shared affine INT4 representation mismatch");
        for(size_t i=0;i<dependencies.size();++i)
            if(!unchanged(record.dependencies[i],dependencies[i]))throw std::runtime_error("Shared affine INT4 checkpoint changed");
        const auto id=slots_.size();slots_.push_back(record.slot);weights_[name]=id;weightRecords_[name]=record;return id;
    }
    AffineInt4Mapping mapped;mapped.Open(metadata);
    auto weight=gpu_->UploadAffineInt4(mapped.Data(0),layout.weightBytes,mapped.Data(1),layout.coefficientBytes,
                                      mapped.Data(2),layout.coefficientBytes,layout.rows,layout.columns,
                                      layout.coefficientType==AffineInt4CoefficientType::FP16);
    for(const auto& dependency:dependencies)
        if(dependency.modified!=std::filesystem::last_write_time(dependency.metadata.filePath)||
           dependency.bytes!=std::filesystem::file_size(dependency.metadata.filePath))
            throw std::runtime_error("Affine INT4 checkpoint changed during upload");
    const auto id=slots_.size();
    slots_.push_back({MetalMatrix(),layout.rows,layout.columns,{int64_t(layout.rows),int64_t(layout.columns)},MetalBF16Weight(),std::move(weight)});
    weightRecords_[name]={slots_[id],*metadata.tensors[0],dependencies[0].modified,dependencies[0].bytes,std::move(dependencies)};
    weights_[name]=id;return id;
}
#endif
size_t MetalCompiledGraph::Allocate(size_t rows,size_t columns) {
    auto id=slots_.size();slots_.push_back({gpu_->Allocate(rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;
}
size_t MetalCompiledGraph::Weight(const std::string& name) {
    auto found=weights_.find(name);if(found!=weights_.end())return found->second;
    auto entry=FindWeightMetadata(name);
    if(!entry)throw std::runtime_error("Metal checkpoint weight missing: "+name);
    if(name.rfind("mtp.",0)==0) {
        if(!mtpProgram_||entry->dataType!="BF16")throw std::invalid_argument("MTP weights require an explicit raw-BF16 MTP program");
        if(entry->shape.size()==2)return BF16Weight(name);
    }
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    if(entry->dataType=="U32")return AffineInt4Weight(name);
    const bool convertedConv=mlxCheckpoint_.ConvertedConvolution(name,*entry);
    const bool convertedPatch=mlxCheckpoint_.ConvertedVisionPatch(name,*entry);
#else
    const bool convertedConv=false;
#endif
#ifdef GARNET_METAL_VISION_GLOBAL_ALPHA
    auto ends=[&](const std::string& suffix){return name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;};
    const bool visionProjection=name.rfind("model.visual.blocks.",0)==0&&
        (ends(".attn.qkv.weight")||ends(".attn.proj.weight")||ends(".mlp.linear_fc1.weight")||ends(".mlp.linear_fc2.weight"));
    if(entry->dataType=="BF16"&&entry->shape.size()==2&&visionProjection) {
        const auto id=BF16Weight(name,true);
        if(slots_[id].bf16.GlobalHalfReady())return id;
        // Inexact common scales retain the original FP32-expanded MPS path.
    }
#endif
    if(bf16Projections_&&entry->dataType=="BF16"&&entry->shape.size()==2&&
        (name.rfind("model.language_model.layers.",0)==0||name.rfind("model.layers.",0)==0))return BF16Weight(name);
    const auto modified=std::filesystem::last_write_time(entry->filePath);
    const auto bytes=std::filesystem::file_size(entry->filePath);
    auto shared=sharedWeights_.find(name);
    if(shared!=sharedWeights_.end()) {
        const auto& record=shared->second;const auto& metadata=record.metadata;
        if(metadata.filePath!=entry->filePath||metadata.dataType!=entry->dataType||metadata.shape!=entry->shape||
            metadata.dataOffset!=entry->dataOffset||metadata.dataSize!=entry->dataSize||record.modified!=modified||record.bytes!=bytes)
            throw std::runtime_error("Shared Metal checkpoint changed or does not match");
        auto id=slots_.size();slots_.push_back(record.slot);weights_[name]=id;weightRecords_[name]=record;return id;
    }
    const bool patch=entry&&name=="model.visual.patch_embed.proj.weight"&&entry->shape.size()==5;
    if(!entry||entry->shape.empty()||(!patch&&(entry->shape.size()>3||
        (entry->shape.size()==3&&entry->shape[1]!=1&&!convertedConv))))
        throw std::runtime_error("Metal matrix weight missing/invalid: "+name);
    for(auto dim:entry->shape)if(dim<=0)throw std::runtime_error("Invalid Metal checkpoint dimension");
    const size_t rows=entry->shape.size()==1?1:entry->shape[0];
    size_t columns=convertedConv?size_t(entry->shape[1]):size_t(entry->shape.back());
    if(patch){columns=1;for(size_t i=1;i<entry->shape.size();++i){
        if(uint64_t(entry->shape[i])>std::numeric_limits<size_t>::max()/columns)throw std::runtime_error("Metal patch dimension overflow");
        columns*=entry->shape[i];}}
    if(rows>std::numeric_limits<size_t>::max()/columns || rows*columns>std::numeric_limits<size_t>::max()/4)
        throw std::runtime_error("Metal checkpoint matrix size overflow");
    SafeTensorsMappedFile file;std::string error;
    if(!file.Open(entry->filePath,error))throw std::runtime_error(error);
    std::vector<float> data(rows*columns);
    auto raw=file.DataAt(entry->dataOffset,entry->dataSize);
    if(!raw)throw std::runtime_error("Metal checkpoint data is out of file bounds");
    if(entry->dataType=="BF16") {
        if(entry->dataSize!=data.size()*2)throw std::runtime_error("Invalid BF16 weight length");
        for(size_t i=0;i<data.size();++i){uint16_t half;std::memcpy(&half,static_cast<const char*>(raw)+2*i,2);
            uint32_t bits=uint32_t(half)<<16;std::memcpy(&data[i],&bits,4);}
    }else if(entry->dataType=="F16") {
        if(entry->dataSize!=data.size()*2)throw std::runtime_error("Invalid FP16 weight length");
        for(size_t i=0;i<data.size();++i){uint16_t half;std::memcpy(&half,static_cast<const char*>(raw)+2*i,2);
            data[i]=Float16ToFloat32(half);}
    }else if(entry->dataType=="F32") {
        if(entry->dataSize!=data.size()*4)throw std::runtime_error("Invalid FP32 weight length");
        std::memcpy(data.data(),raw,entry->dataSize);
    }else throw std::runtime_error("Unsupported initial Metal weight dtype: "+entry->dataType);
    auto logicalShape=std::vector<int64_t>(entry->shape.begin(),entry->shape.end());
    if(convertedConv)logicalShape={int64_t(rows),1,int64_t(columns)};
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    if(convertedPatch) {
        data=MlxVisionPatchToCanonical(data,rows,size_t(entry->shape[1]),size_t(entry->shape[2]),
                                        size_t(entry->shape[3]),size_t(entry->shape[4]));
        logicalShape={int64_t(rows),entry->shape[4],entry->shape[1],entry->shape[2],entry->shape[3]};
    }
#endif
    const auto id=slots_.size();slots_.push_back({gpu_->Upload(data,rows,columns),rows,columns,std::move(logicalShape)});
    if(modified!=std::filesystem::last_write_time(entry->filePath)||bytes!=std::filesystem::file_size(entry->filePath))
        throw std::runtime_error("Metal checkpoint changed while loading weights");
    weightRecords_[name]={slots_[id],*entry,modified,bytes};
    weights_[name]=id;return id;
}
size_t MetalCompiledGraph::BF16Weight(const std::string& name,bool visionPrefill) {
    const auto key=name+(visionPrefill?"|VisionGlobalBF16":"|BF16");auto found=weights_.find(key);if(found!=weights_.end())return found->second;
    auto entry=FindWeightMetadata(name);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    if(entry&&entry->dataType=="U32")return AffineInt4Weight(name);
#endif
    if(!entry||entry->shape.size()!=2||entry->shape[0]<=0||entry->shape[1]<=0)
        throw std::runtime_error("Invalid vocabulary checkpoint matrix");
    // F32 fixtures retain their exact original representation through MPS.
    if(entry->dataType=="F32")return Weight(name);
    if(entry->dataType!="BF16")throw std::runtime_error("BF16 vocabulary head requires BF16/F32 checkpoint weights");
    const auto modified=std::filesystem::last_write_time(entry->filePath);
    const auto bytes=std::filesystem::file_size(entry->filePath);
    auto shared=sharedWeights_.find(key);
    if(shared!=sharedWeights_.end()) {
        const auto& record=shared->second;const auto& metadata=record.metadata;
        if(metadata.filePath!=entry->filePath||metadata.dataType!=entry->dataType||metadata.shape!=entry->shape||
            metadata.dataOffset!=entry->dataOffset||metadata.dataSize!=entry->dataSize||record.modified!=modified||record.bytes!=bytes)
            throw std::runtime_error("Shared BF16 checkpoint changed or does not match");
        const auto id=slots_.size();slots_.push_back(record.slot);weights_[key]=id;weightRecords_[key]=record;return id;
    }
    SafeTensorsMappedFile file;std::string error;
    if(!file.Open(entry->filePath,error))throw std::runtime_error(error);
    auto data=file.DataAt(entry->dataOffset,entry->dataSize);
    if(!data)throw std::runtime_error("Vocabulary checkpoint exceeds file bounds");
    bool floatPrefill=false;
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
    const std::string keySuffix=".self_attn.k_proj.weight";
    floatPrefill=name.rfind("model.layers.",0)==0&&name.size()>=keySuffix.size()&&
        name.compare(name.size()-keySuffix.size(),keySuffix.size(),keySuffix)==0;
#endif
    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill);
    auto id=slots_.size();slots_.push_back({MetalMatrix(),size_t(entry->shape[0]),size_t(entry->shape[1]),std::vector<int64_t>(entry->shape.begin(),entry->shape.end()),std::move(weight)});
    if(modified!=std::filesystem::last_write_time(entry->filePath)||bytes!=std::filesystem::file_size(entry->filePath))throw std::runtime_error("BF16 checkpoint changed while loading");
    weightRecords_[key]={slots_[id],*entry,modified,bytes};weights_[key]=id;return id;
}
size_t MetalCompiledGraph::LowerVision(size_t input,const X::Value& attrs) {
    auto integer=[&](const char* name,size_t limit){auto v=attrs.Get(name).ToInt64();
        if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal vision profile: ")+name);return size_t(v);};
    const auto gh=integer("grid_h",128),gw=integer("grid_w",128),hidden=integer("hidden_size",4096),heads=integer("num_heads",128),
        depth=integer("depth",64),patch=integer("patch_size",64),temporal=integer("temporal_patch_size",8),channels=integer("in_channels",16),
        merge=integer("spatial_merge_size",8),positions=integer("num_position_embeddings",65536),outHidden=integer("out_hidden_size",65536);
    const auto theta=attrs.Get("rope_theta").ToDouble();
    const size_t n=gh*gw,unit=merge*merge,feature=channels*temporal*patch*patch,side=size_t(std::sqrt(double(positions)));
    const auto shape=slots_[input].shape;
    if(gh%merge||gw%merge||hidden%heads||(hidden/heads)%4||side*side!=positions||!(theta>0)||!std::isfinite(theta)||
       shape.size()!=3||shape[1]!=int64_t(n)||shape[2]!=int64_t(feature))throw std::runtime_error("Metal still-image profile mismatch");
    const size_t batch=shape[0],dim=hidden/heads;
    auto constant=[&](const std::vector<float>& values,size_t rows,size_t columns){auto id=slots_.size();
        slots_.push_back({gpu_->Upload(values,rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;};
    auto vision=[&](unsigned mode,size_t a,size_t b,size_t c,size_t rows,size_t columns){
        auto out=Allocate(rows,columns);Op op{Kind::Vision,a,b,out};op.arguments={a,b,c};op.dimensions={mode,n,hidden,heads,unit};ops_.push_back(std::move(op));return out;};
    auto project=[&](size_t a,const std::string& prefix){
        auto w=Weight(prefix+".weight"),bias=Weight(prefix+".bias");
        if(slots_[w].columns!=slots_[a].columns||slots_[bias].rows!=1||slots_[bias].columns!=slots_[w].rows)
            throw std::runtime_error("Metal vision projection checkpoint mismatch: "+prefix);
        auto product=Allocate(slots_[a].rows,slots_[w].rows);ops_.push_back({Kind::MatMul,a,w,product,0,0,true});
        return vision(0,product,bias,bias,slots_[product].rows,slots_[product].columns);
    };
    auto norm=[&](size_t a,const std::string& prefix){auto w=Weight(prefix+".weight"),bias=Weight(prefix+".bias");
        if(slots_[w].rows!=1||slots_[bias].rows!=1||slots_[w].columns!=slots_[a].columns||slots_[bias].columns!=slots_[a].columns)
            throw std::runtime_error("Metal vision norm checkpoint mismatch");
        return vision(1,a,w,bias,slots_[a].rows,slots_[a].columns);};
    auto add=[&](size_t a,size_t b){if(slots_[a].rows!=slots_[b].rows||slots_[a].columns!=slots_[b].columns)
        throw std::runtime_error("Metal vision residual dimensions mismatch");
        auto out=Allocate(slots_[a].rows,slots_[a].columns);ops_.push_back({Kind::Add,a,b,out});return out;};
    // Flatten only input batch/token rows; the copy respects native row padding.
    auto flat=vision(9,input,input,input,batch*n,feature);
    auto patchWeight=Weight("model.visual.patch_embed.proj.weight");
    const std::vector<int64_t> patchShape={int64_t(hidden),int64_t(channels),int64_t(temporal),int64_t(patch),int64_t(patch)};
    if(slots_[patchWeight].shape!=patchShape)throw std::runtime_error("Metal vision patch weight shape mismatch");
    auto x=project(flat,"model.visual.patch_embed.proj");
    if(slots_[x].columns!=hidden)throw std::runtime_error("Metal vision patch hidden size mismatch");
    auto table=Weight("model.visual.pos_embed.weight");
    if(slots_[table].rows!=positions||slots_[table].columns!=hidden)throw std::runtime_error("Metal vision position table mismatch");
    std::vector<float> interpolation,angles;
    interpolation.reserve(n*8);angles.reserve(n*dim);
    for(size_t bh=0;bh<gh;bh+=merge)for(size_t bw=0;bw<gw;bw+=merge)
      for(size_t ih=0;ih<merge;++ih)for(size_t iw=0;iw<merge;++iw){
        const size_t h=bh+ih,w=bw+iw;
        const float ph=gh>1?float(h)*float(side-1)/float(gh-1):0,pw=gw>1?float(w)*float(side-1)/float(gw-1):0;
        const size_t h0=size_t(ph),w0=size_t(pw),h1=std::min(h0+1,side-1),w1=std::min(w0+1,side-1);
        const float dh=ph-h0,dw=pw-w0;
        interpolation.insert(interpolation.end(),{float(h0*side+w0),float(h0*side+w1),float(h1*side+w0),float(h1*side+w1),
            (1-dh)*(1-dw),(1-dh)*dw,dh*(1-dw),dh*dw});
        for(size_t repeat=0;repeat<2;++repeat)for(size_t axis=0;axis<2;++axis)for(size_t d=0;d<dim/4;++d)
            angles.push_back(float(double(axis?w:h)*std::pow(theta,-double(2*d)/double(dim/2))));
      }
    const auto metadata=constant(interpolation,n,8),rope=constant(angles,n,dim);
    x=vision(5,x,table,metadata,batch*n,hidden);
    for(size_t layer=0;layer<depth;++layer){
        auto prefix="model.visual.blocks."+std::to_string(layer);
        auto qkv=project(norm(x,prefix+".norm1"),prefix+".attn.qkv");
        if(slots_[qkv].columns!=3*hidden)throw std::runtime_error("Metal vision QKV shape mismatch");
        auto rotated=vision(6,qkv,rope,rope,batch*n,3*hidden);
        size_t context;
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
        if(batch*n<=1024&&n<=512&&heads<=16&&dim<=128) {
            context=vision(10,rotated,rotated,rotated,batch*n,hidden);
        } else
#endif
        {
            auto probability=vision(7,rotated,rotated,rotated,batch*n,heads*n);
            context=vision(8,probability,rotated,rotated,batch*n,hidden);
        }
        x=add(x,project(context,prefix+".attn.proj"));
        auto intermediate=project(norm(x,prefix+".norm2"),prefix+".mlp.linear_fc1");
        auto activated=vision(2,intermediate,intermediate,intermediate,slots_[intermediate].rows,slots_[intermediate].columns);
        x=add(x,project(activated,prefix+".mlp.linear_fc2"));
    }
    x=norm(x,"model.visual.merger.norm");
    x=vision(4,x,x,x,batch*n/unit,hidden*unit);
    auto fc=project(x,"model.visual.merger.linear_fc1");
    x=vision(3,fc,fc,fc,slots_[fc].rows,slots_[fc].columns);
    x=project(x,"model.visual.merger.linear_fc2");
    if(slots_[x].columns!=outHidden)throw std::runtime_error("Metal vision merger output width mismatch");
    auto output=vision(4,x,x,x,batch,n/unit*outHidden);
    ops_.back().dimensions[4]=n/unit;
    slots_[output].shape={int64_t(batch),int64_t(n/unit),int64_t(outHidden)};
    return output;
}
void MetalCompiledGraph::Clear() {
    mtpProgram_=mtpPrefill_=false;
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
    nonresidentPackSlots_=0;
#endif
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    prefillScratch_.reset();sharedPrefillScratchBytes_=0;
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
    sharedPrefillAliasGroups_=sharedPrefillAliasSlots_=0;
#ifdef GARNET_METAL_PREFILL_ALIAS_TEST_FAULTS
    prefillAliasFailNextCommit_=false;
#endif
#endif
#endif
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    lazyDefaultPlanReady_=false;lazyDefaultSlots_.clear();lazyRetiredBytes_=lazyRehydratedBytes_=0;
#endif
    visionParallelDispatches_=0;
    visionPrefetchDispatches_=0;visionVectorDispatches_=0;mlpGraphDispatches_=0;
    resetKVPrefix_[0]=resetKVPrefix_[1]=false;
    verificationTokens_=0;
    visionGlobalAlphaDispatches_=0;++generation_;ready_=false;bf16Projections_=false;outputParts_.clear();outputStorageGroups_.clear();residentOps_.clear();controlSources_.clear();ops_.clear();inputs_.clear();inputShapes_.clear();tensors_.clear();weights_.clear();sharedWeights_.clear();weightRecords_.clear();bindings_.clear();slots_.clear();gpu_.reset();
#ifdef GARNET_METAL_MTP_DEFER_PACKS
    mtpDeferredPackSlots_=0;
#endif
}
void MetalCompiledGraph::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
bool MetalCompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{});
}
bool MetalCompiledGraph::BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{},true);
}
#ifdef GARNET_METAL_MTP
bool MetalCompiledGraph::BuildMTP(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{},true,false,true);
}
bool MetalCompiledGraph::BuildMTPPrefill(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source) {
    if(!source.IsValid()||source.IsNone()) {
        std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{},true,false,true,true);
    }
    if(source.runtime()!=Host()->runtime)throw std::invalid_argument("MTP prefill source runtime mismatch");
    auto peer=source.NativeData<MetalCompiledGraph>();
    if(!peer||peer==this)throw std::invalid_argument("MTP prefill requires another raw MTP source");
    std::scoped_lock lock(mutex_,peer->mutex_);
    if(!peer->ready_||!peer->mtpProgram_)throw std::invalid_argument("MTP prefill source must be ready raw MTP");
    return BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,true,false,true,true);
}
#endif
bool MetalCompiledGraph::BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source) {
    if(source.runtime()!=Host()->runtime)throw std::invalid_argument("Shared Metal graph runtime mismatch");
    auto peer=source.NativeData<MetalCompiledGraph>();
    if(!peer||peer==this)throw std::invalid_argument("Shared Metal weights require another compiled graph");
    std::scoped_lock lock(mutex_,peer->mutex_);
    if(!peer->ready_)throw std::invalid_argument("Shared Metal weight source is closed");
    return BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_,false,peer->mtpProgram_,peer->mtpPrefill_);
}
#if defined(GARNET_METAL_VERIFY_ATTENTION)||defined(GARNET_METAL_COMPACT_OWNER_PREFILL)
bool MetalCompiledGraph::SharesVerificationWeights(const MetalCompiledGraph& source) const {
    if(weightRecords_.size()!=source.weightRecords_.size())return false;
    auto buffers=[](const Slot& slot) {
        std::vector<uintptr_t> ids;
        auto add=[&](const MetalBufferInfo& info){if(info.identity)ids.push_back(info.identity);};
        add(MetalContext::MatrixBufferInfo(slot.matrix));
        for(const auto& info:MetalContext::WeightBufferInfo(slot.bf16))add(info);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        for(const auto& info:MetalContext::AffineInt4BufferInfo(slot.affineInt4))add(info);
#endif
        return ids;
    };
    for(const auto& entry:weightRecords_) {
        auto found=source.weightRecords_.find(entry.first);
        if(found==source.weightRecords_.end()||buffers(entry.second.slot)!=buffers(found->second.slot))return false;
    }
    return !weightRecords_.empty();
}
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
bool MetalCompiledGraph::BuildVerificationShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source) {
#if !defined(GARNET_METAL_MLX_AFFINE_INT4)||!defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)||!defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)||!defined(GARNET_METAL_INDEXED_DECODE)||!defined(GARNET_METAL_AFFINE_PROFILE)
    throw std::logic_error("Compiled verification requires prepared FP32 affine kernels");
#else
    if(source.runtime()!=Host()->runtime)throw std::invalid_argument("Shared Metal graph runtime mismatch");
    auto peer=source.NativeData<MetalCompiledGraph>();
    if(!peer||peer==this)throw std::invalid_argument("Verification requires another compiled decoder graph");
    std::scoped_lock lock(mutex_,peer->mutex_);
    if(!peer->ready_||peer->verificationTokens_||(peer->inputs_.size()!=4&&peer->inputs_.size()!=6)||
       peer->inputShapes_[0].size()!=2||peer->outputParts_.size()!=3)
        throw std::invalid_argument("Verification source must be a ready ordinary decoder layer");
    const auto batch=peer->inputShapes_[0][0];
    const bool linearSource=peer->inputs_.size()==4&&peer->inputShapes_[1].size()==3&&peer->inputShapes_[2].size()==4&&
        peer->inputShapes_[3]==std::vector<int64_t>{batch};
    const bool sixSource=peer->inputs_.size()==6&&peer->inputShapes_[3]==std::vector<int64_t>{batch,3}&&
        peer->inputShapes_[4]==std::vector<int64_t>{batch}&&peer->inputShapes_[5]==peer->inputShapes_[4];
    if(!linearSource&&!sixSource)
        throw std::invalid_argument("Verification source control profile mismatch");
#if !defined(GARNET_METAL_LINEAR_CHUNK_CONV)||!defined(GARNET_METAL_LINEAR_CHUNK_DELTA)
    if(linearSource)throw std::invalid_argument("Recurrent verification requires packed convolution and delta kernels");
#endif
    // The factory's BF16 preference does not affect converted affine matrices.
    // Certify actual projection representations rather than that preference.
    for(const auto& op:peer->ops_)if((op.kind==Kind::MatMul||op.kind==Kind::BF16MatVec)&&
        (op.kind!=Kind::MatMul||!op.transpose||!peer->slots_[op.b].affineInt4.Valid()))
        throw std::invalid_argument("Verification source projections must be affine FP32 decode");
    if(symbols.runtime()!=Host()->runtime||!symbols.IsList()||symbols.Size()!=6)
        throw std::invalid_argument("Verification requires six symbolic inputs");
    for(size_t i=0;i<3;++i) {
        X::Tensor symbol(symbols.Get(i));auto info=symbol.Info();const auto& expected=peer->inputShapes_[i];
        if(!info.symbolic||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=(i?expected.size():3)||
           (i?!std::equal(expected.begin(),expected.end(),info.shape):
                (info.shape[0]!=expected[0]||info.shape[2]!=expected[1])))
            throw std::invalid_argument("Verification source hidden/state profile mismatch");
    }
    const bool built=BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,false,true);
    if(!SharesVerificationWeights(*peer)){Clear();throw std::invalid_argument("Verification must share all source decoder weights");}
    return built;
#endif
}
#endif
void MetalCompiledGraph::ValidateMTPProgram(bool metadataOnly) const {
    const auto width=inputShapes_[0][2],chunk=inputShapes_[0][1];
    const auto kv=inputShapes_[2][1],dim=inputShapes_[2][3];
    auto matrix=[&](const std::string& name,std::vector<long long> shape) {
        if(metadataOnly) {
            const auto* entry=index_.Find(name);
            if(!entry||entry->dataType!="BF16"||entry->shape!=shape)
                throw std::invalid_argument("MTP BF16 checkpoint dimensions mismatch: "+name);
            return;
        }
        const auto found=weightRecords_.find(name+"|BF16");
        if(found==weightRecords_.end()||found->second.metadata.dataType!="BF16"||found->second.metadata.shape!=shape||
           !found->second.slot.bf16.Valid())throw std::invalid_argument("MTP BF16 projection contract mismatch: "+name);
    };
    auto metadata=[&](const char* name){const auto* entry=index_.Find(name);
        if(!entry||entry->shape.size()!=2)return int64_t(0);return int64_t(entry->shape[0]);};
    const auto q=metadata("mtp.layers.0.self_attn.q_proj.weight"),middle=metadata("mtp.layers.0.mlp.gate_proj.weight");
    if(q<=0||q%(2*dim)||q/(2*dim)>32||q/(2*dim)%kv||middle<=0||middle>17408)
        throw std::invalid_argument("MTP attention/MLP dimensions exceed the prepared profile");
    matrix("mtp.fc.weight",{width,2*width});
    for(auto name:{"k_proj","v_proj"})matrix(std::string("mtp.layers.0.self_attn.")+name+".weight",{kv*dim,width});
    matrix("mtp.layers.0.self_attn.q_proj.weight",{q,width});
    matrix("mtp.layers.0.self_attn.o_proj.weight",{width,q/2});
    for(auto name:{"gate_proj","up_proj"})matrix(std::string("mtp.layers.0.mlp.")+name+".weight",{middle,width});
    matrix("mtp.layers.0.mlp.down_proj.weight",{width,middle});
    for(const auto& name:{"mtp.pre_fc_norm_embedding.weight","mtp.pre_fc_norm_hidden.weight","mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight","mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight","mtp.layers.0.self_attn.k_norm.weight"}) {
        const std::string key=name;const bool qk=key.find("self_attn.")!=std::string::npos;
        const auto* entry=index_.Find(key);
        if(!entry||entry->dataType!="BF16"||entry->shape!=std::vector<long long>{qk?dim:width})
            throw std::invalid_argument("MTP raw checkpoint norm dimensions mismatch: "+key);
    }
    if(index_.TensorCount()!=15)throw std::invalid_argument("MTP requires a separate fifteen-tensor auxiliary checkpoint");
    if(metadataOnly)return;
    for(const auto& entry:weightRecords_)if(entry.first.find("|BF16")==std::string::npos) {
        const auto& m=entry.second.metadata;
        const bool qk=entry.first=="mtp.layers.0.self_attn.q_norm.weight"||entry.first=="mtp.layers.0.self_attn.k_norm.weight";
        const bool norm=entry.first=="mtp.pre_fc_norm_embedding.weight"||entry.first=="mtp.pre_fc_norm_hidden.weight"||
            entry.first=="mtp.layers.0.input_layernorm.weight"||entry.first=="mtp.layers.0.post_attention_layernorm.weight"||entry.first=="mtp.norm.weight";
        if((!qk&&!norm)||m.dataType!="BF16"||m.shape!=std::vector<long long>{qk?dim:width})
            throw std::invalid_argument("MTP raw norm contract mismatch: "+entry.first);
    }
    size_t projections=0,norms=0,ropes=0,attention=0,updates=0;
    for(const auto& op:ops_) {
        if(op.kind==Kind::MatMul||op.kind==Kind::BF16MatVec){
            if(op.kind!=Kind::MatMul||!op.transpose||!slots_[op.b].bf16.Valid())throw std::invalid_argument("MTP requires eight native BF16 projections");++projections;}
        if(op.kind==Kind::Norm){if(op.offset!=1)throw std::invalid_argument("MTP raw norms require +1 offsets");++norms;}
        if(op.kind==Kind::NormRope){if(op.offset!=1||op.dimensions.back()!=1)throw std::invalid_argument("MTP Q/K raw norms require +1 offsets");++ropes;}
        if(op.kind==Kind::AttentionChunk){if(op.dimensions.back()!=1)throw std::invalid_argument("MTP requires gated full attention");++attention;}
        if(op.kind==Kind::KVChunk){
            if(op.arguments[1]!=(updates?inputs_[3]:inputs_[2])||op.arguments[2]!=inputs_[5]||op.arguments[3]!=inputs_[6]||op.arguments[4]!=inputs_[4])
                throw std::invalid_argument("MTP requires explicit independent K/V and controls");++updates;}
    }
    if(weightRecords_.size()!=15||projections!=8||norms!=5||ropes!=2||attention!=1||updates!=2||
       outputParts_.size()!=3||slots_[outputParts_[0]].columns!=size_t(width*chunk)||
       slots_[outputParts_[1]].columns!=size_t(kv*dim*inputShapes_[2][2])||slots_[outputParts_[2]].columns!=slots_[outputParts_[1]].columns)
        throw std::invalid_argument("MTP must return normalized hidden plus K/V from exactly one complete auxiliary layer");
}
bool MetalCompiledGraph::BuildInternal(X::Value graph,X::Value symbols,const std::string& checkpoint,
    std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections,bool verification,bool mtp,bool mtpPrefill) {
    Clear();sharedWeights_=std::move(shared);bf16Projections_=bf16Projections;
    auto start=std::chrono::steady_clock::now();
    try {
#ifndef GARNET_METAL_MTP
        if(mtp)throw std::logic_error("MTP graph extension is disabled");
#endif
        mtpProgram_=mtp;mtpPrefill_=mtpPrefill;
        if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
            throw std::runtime_error("Metal graph inputs must belong to this runtime");
        std::string error;if(!index_.Open(checkpoint,error))throw std::runtime_error(error);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        mlxCheckpoint_.Open(index_);
        if(mtp&&mlxCheckpoint_.Enabled())throw std::invalid_argument("MTP auxiliary checkpoint must retain raw BF16 norm offsets");
        if(verification&&!mlxCheckpoint_.Enabled())throw std::invalid_argument("Verification requires converted affine INT4/group64 weights");
#endif
        if(verification) {
            if(symbols.Size()!=6)throw std::invalid_argument("Verification requires six symbolic inputs");
            X::Tensor hidden(symbols.Get(uint64_t(0)));auto info=hidden.Info();
            if(info.rank!=3||info.shape[0]<1||info.shape[0]>4||info.shape[1]<1||info.shape[1]>4/info.shape[0])
                throw std::invalid_argument("Verification supports at most four physical token rows");
            verificationTokens_=size_t(info.shape[1]);
        }
        if(mtpProgram_) {
            if(symbols.Size()!=7)throw std::invalid_argument("MTP requires seven explicit inputs");
            for(uint64_t i=0;i<symbols.Size();++i) {
                X::Tensor symbol(symbols.Get(i));const auto info=symbol.Info();
                if(!info.symbolic||info.dtype!=X3_TENSOR_FLOAT32||info.rank<1||info.rank>4)
                    throw std::invalid_argument("MTP requires symbolic FP32 inputs");
                inputShapes_.emplace_back(info.shape,info.shape+info.rank);
                for(auto d:inputShapes_.back())if(d<=0)throw std::invalid_argument("MTP input dimensions must be positive");
            }
            const auto& h=inputShapes_[0];const auto& k=inputShapes_[2];
            if(h.size()!=3||h[0]>4||h[1]>(mtpPrefill_?128:4)/h[0]||h[2]>5120||inputShapes_[1]!=h||
               k.size()!=4||k[0]!=h[0]||k[1]>8||k[2]>4096||k[3]>256||inputShapes_[3]!=k||
               inputShapes_[4]!=std::vector<int64_t>{h[0],h[1],3}||
               inputShapes_[5]!=std::vector<int64_t>{h[0],h[1]}||inputShapes_[6]!=inputShapes_[5])
                throw std::invalid_argument("MTP profile requires bounded FP32 KV/control inputs and at most four decode or128 explicit prefill rows");
            // Reject oversized auxiliary profiles/weights before any GPU allocation.
            ValidateMTPProgram(true);
        }
        gpu_=std::make_unique<MetalContext>();
        for(uint64_t i=0;i<symbols.Size();++i) {
            X::Tensor symbol(symbols.Get(i));auto info=symbol.Info();
            if(!info.symbolic||info.rank<1||info.rank>4||info.dtype!=X3_TENSOR_FLOAT32)
                throw std::runtime_error("Metal graph input must be symbolic FP32 rank one through four");
            size_t columns=1;
            for(size_t axis=0;axis<info.rank;++axis) {
                if(info.shape[axis]<=0)throw std::runtime_error("Metal graph input dimensions must be positive");
                if(axis) {
                    if(uint64_t(info.shape[axis])>std::numeric_limits<size_t>::max()/columns)
                        throw std::runtime_error("Metal graph input dimensions overflow");
                    columns*=info.shape[axis];
                }
            }
            if(!mtpProgram_)inputShapes_.emplace_back(info.shape,info.shape+info.rank);
            const auto id=Allocate(info.shape[0],columns);inputs_.push_back(id);slots_[id].shape=inputShapes_[i];
            if(!tensors_.emplace(info.id,id).second)throw std::runtime_error("Duplicate Metal symbolic input");
        }
        if(verificationTokens_) {
            const auto batch=inputShapes_[0][0],chunk=int64_t(verificationTokens_);
            if(inputShapes_[3]!=std::vector<int64_t>{batch,chunk,3}||
               inputShapes_[4]!=std::vector<int64_t>{batch,chunk}||inputShapes_[5]!=inputShapes_[4])
                throw std::invalid_argument("Verification control profile mismatch");
        }
        size_t verificationBlocks=0;
        TensorGraphCapture capture(graph);
        auto lookup=[&](const X::Value& value){auto found=tensors_.find(TensorId(value));
            if(found==tensors_.end())throw std::runtime_error("Unknown Metal graph operand");
            return found->second;};
        std::map<size_t,std::vector<size_t>> packedParts;
        std::map<size_t,size_t> aliases;
        std::vector<size_t> deferredPacks, lazyPacks;
        auto allocatePack=[&](size_t rows,size_t columns) {
#ifdef GARNET_METAL_MTP_DEFER_PACKS
            if(mtpProgram_) {
                const auto id=slots_.size();
                slots_.push_back({MetalMatrix(),rows,columns,{int64_t(rows),int64_t(columns)}});
                deferredPacks.push_back(id);return id;
            }
#endif
            return Allocate(rows,columns);
        };
        auto slice=[&](size_t input,size_t groups,size_t sourceWidth,size_t start,size_t count,
            std::vector<int64_t> shape=std::vector<int64_t>{}) {
            if(!groups||!count||start>sourceWidth||count>sourceWidth-start||groups>std::numeric_limits<size_t>::max()/sourceWidth||
                slots_[input].columns!=groups*sourceWidth)throw std::runtime_error("Metal graph slice shape mismatch");
            auto parts=packedParts.find(input);
            if(groups==1&&parts!=packedParts.end()) {
                size_t offset=0;
                for(auto part:parts->second) {
                    if(start==offset&&count==slots_[part].columns) {
                        // A complete packed segment already exists as an independent matrix.
                        // Alias its storage; preserve this view's logical shape separately.
                        auto view=slots_[part];view.shape=shape.empty()?std::vector<int64_t>{int64_t(view.rows),int64_t(view.columns)}:shape;
                        const auto id=slots_.size();slots_.push_back(std::move(view));aliases[id]=part;return id;
                    }
                    offset+=slots_[part].columns;
                }
            }
            auto out=Allocate(slots_[input].rows,groups*count);
            if(!shape.empty())slots_[out].shape=std::move(shape);
            Op op{Kind::Slice,input,input,out};op.dimensions={groups,sourceWidth,start,count};ops_.push_back(std::move(op));return out;
        };
        // Input projections are independent across chunk tokens; recurrent attention stays sequential.
        std::map<size_t,std::pair<size_t,size_t>> chunkOrigins;
        std::map<size_t,size_t> flatChunks;
        std::map<std::pair<size_t,size_t>,size_t> chunkProjections;
        auto matmul=[&](size_t a,size_t b){
            if(slots_[a].columns!=slots_[b].columns)throw std::runtime_error("Metal checkpoint projection shape mismatch");
            auto origin=chunkOrigins.find(a);
            if(origin!=chunkOrigins.end()&&slots_[origin->second.first].shape.size()==3) {
                const auto source=origin->second.first,index=origin->second.second;
                const auto shape=slots_[source].shape;
                const size_t batch=shape[0],chunk=shape[1],width=shape[2],outputWidth=slots_[b].rows;
                auto flat=flatChunks.find(source);
                if(flat==flatChunks.end()) {
                    auto id=Allocate(batch*chunk,width);
                    Op op{Kind::Layout,source,source,id};op.dimensions={0};ops_.push_back(std::move(op));
                    flat=flatChunks.emplace(source,id).first;
                }
                const auto key=std::make_pair(source,b);
                auto projected=chunkProjections.find(key);
                if(projected==chunkProjections.end()) {
                    auto matrix=Allocate(batch*chunk,outputWidth);
                    ops_.push_back({Kind::MatMul,flat->second,b,matrix,0,0,true});
                    auto packed=Allocate(batch,chunk*outputWidth);
                    Op op{Kind::Layout,matrix,matrix,packed};op.dimensions={0};ops_.push_back(std::move(op));
                    projected=chunkProjections.emplace(key,packed).first;
                }
                return slice(projected->second,1,chunk*outputWidth,index*outputWidth,outputWidth);
            }
            auto out=Allocate(slots_[a].rows,slots_[b].rows);ops_.push_back({Kind::MatMul,a,b,out,0,0,true});return out;
        };
        for(const auto& node:capture.Operations()) {
            if(node.provider.empty())continue;
            if(node.provider!="garnet"||node.inputs.empty()||node.operandCount<1||node.operandCount>2)
                throw std::runtime_error("Unsupported initial Metal graph operation: "+node.name);
            auto a=lookup(node.inputs[0]);size_t out;
            if(node.operandCount==2) {
                auto b=lookup(node.inputs[1]);
                if(node.name.rfind("qwen35_conv_bind_",0)==0||node.name.rfind("qwen35_delta_bind_",0)==0||node.name.rfind("qwen35_attention_bind_",0)==0) {
                    static const std::vector<std::string> allowed={"qwen35_conv_bind_weight","qwen35_conv_bind_state",
                        "qwen35_conv_bind_active","qwen35_delta_bind_key","qwen35_delta_bind_value",
                        "qwen35_delta_bind_decay","qwen35_delta_bind_beta","qwen35_delta_bind_state","qwen35_delta_bind_active",
                        "qwen35_attention_bind_key","qwen35_attention_bind_value","qwen35_attention_bind_positions",
                        "qwen35_attention_bind_slots","qwen35_attention_bind_active"};
                    if(std::find(allowed.begin(),allowed.end(),node.name)==allowed.end())
                        throw std::runtime_error("Unknown Metal attention binding: "+node.name);
                    auto bound=bindings_[TensorId(node.inputs[0])];bound[node.name]=b;
                    bindings_[TensorId(node.output)]=std::move(bound);tensors_[TensorId(node.output)]=a;continue;
                }
                if(node.name=="qwen35_concat_sequence") {
                    auto as=slots_[a].shape,bs=slots_[b].shape;
                    if(as.size()!=bs.size()||(as.size()!=2&&as.size()!=3)||as[0]!=bs[0]||(as.size()==3&&as[2]!=bs[2]))
                        throw std::runtime_error("Metal chunk concat shape mismatch");
                    out=allocatePack(slots_[a].rows,slots_[a].columns+slots_[b].columns);as[1]+=bs[1];slots_[out].shape=as;
                    Op op{Kind::Concat,a,b,out};op.arguments={a,b,b};op.transpose=false;ops_.push_back(std::move(op));
                    for(auto part:{a,b}) {
                        auto parts=packedParts.find(part);
                        if(parts==packedParts.end())packedParts[out].push_back(part);
                        else packedParts[out].insert(packedParts[out].end(),parts->second.begin(),parts->second.end());
                    }
                    tensors_[TensorId(node.output)]=out;continue;
                }
                if(node.name=="qwen35_mask_chunk") {
                    const auto as=slots_[a].shape,bs=slots_[b].shape;
                    if(as.size()!=3||bs!=std::vector<int64_t>{as[0],as[1]})throw std::runtime_error("Metal chunk mask shape mismatch");
                    out=Allocate(slots_[a].rows,slots_[a].columns);slots_[out].shape=as;
                    Op op{Kind::Layout,a,b,out};op.dimensions={size_t(as[2])};ops_.push_back(std::move(op));
                    tensors_[TensorId(node.output)]=out;continue;
                }
                if(node.name=="qwen35_replace_packed_hidden") {
                    auto hiddenDim=node.attributes.Get("hidden_dim").ToInt64();
                    if(hiddenDim<=0||slots_[a].rows!=slots_[b].rows||slots_[b].columns!=uint64_t(hiddenDim)||
                        slots_[a].columns<=uint64_t(hiddenDim))throw std::runtime_error("Metal packed hidden replacement shape mismatch");
                    const auto packedWidth=slots_[a].columns;
                    auto parts=packedParts.find(a);
                    if(parts!=packedParts.end()&&parts->second.size()==3&&slots_[parts->second[0]].columns==uint64_t(hiddenDim)) {
                        out=Allocate(slots_[a].rows,packedWidth);
                        Op op{Kind::Concat,b,parts->second[1],out};op.arguments={b,parts->second[1],parts->second[2]};op.transpose=true;
                        packedParts[out]=op.arguments;ops_.push_back(std::move(op));
                        tensors_[TensorId(node.output)]=out;continue;
                    }
                    auto tail=slice(a,1,packedWidth,hiddenDim,packedWidth-hiddenDim);
                    out=Allocate(slots_[a].rows,packedWidth);
                    Op op{Kind::Concat,b,tail,out};op.arguments={b,tail,tail};op.transpose=false;ops_.push_back(std::move(op));
                    tensors_[TensorId(node.output)]=out;continue;
                }
                const bool mask=node.name=="qwen35_mask_rows";
                if(node.name!="add"&&!mask)throw std::runtime_error("Metal binary lowering not implemented: "+node.name);
                if(slots_[a].rows!=slots_[b].rows||slots_[b].columns!=(mask?1:slots_[a].columns))
                    throw std::runtime_error("Metal binary shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({mask?Kind::Mask:Kind::Add,a,b,out});slots_[out].shape=slots_[a].shape;
                tensors_[TensorId(node.output)]=out;continue;
            }
            if(node.name=="reshape") {
                auto dimensions=node.attributes.Get("shape");std::vector<int64_t> shape;size_t total=slots_[a].rows*slots_[a].columns,known=1;int inferred=-1;
                if(!dimensions.IsList()||!dimensions.Size()||dimensions.Size()>4)throw std::runtime_error("Invalid Metal reshape rank");
                for(uint64_t i=0;i<dimensions.Size();++i){auto d=dimensions.Get(i).ToInt64();shape.push_back(d);
                    if(d==-1){if(inferred!=-1)throw std::runtime_error("Multiple inferred reshape dimensions");inferred=int(i);}
                    else{if(d<=0||uint64_t(d)>total/known)throw std::runtime_error("Invalid Metal reshape dimension");known*=size_t(d);}}
                if(inferred>=0){if(total%known)throw std::runtime_error("Indivisible Metal reshape");shape[inferred]=total/known;}
                else if(known!=total)throw std::runtime_error("Metal reshape element count mismatch");
                bool aliasCache=false;size_t root=a;
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
                while(aliases.count(root))root=aliases.at(root);
                // Only complete row-preserving views of a fresh chunk-cache
                // writer can share storage. Inputs and arbitrary reshapes keep
                // the existing copy path and immutable host-input contract.
                if(slots_[a].rows==size_t(shape[0])&&slots_[a].columns==total/size_t(shape[0])&&
                   gpu_->IsDefaultLayout(slots_[a].matrix,slots_[a].rows,slots_[a].columns))
                    for(const auto& producer:ops_) {
                        if(producer.kind==Kind::KVChunk&&producer.out==root)aliasCache=true;
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
                        if((producer.kind==Kind::ConvChunk||producer.kind==Kind::DeltaChunk)&&producer.auxiliaryOutput==root)aliasCache=true;
#endif
                    }
#endif
                if(aliasCache){auto view=slots_[a];view.shape=shape;out=slots_.size();slots_.push_back(std::move(view));aliases[out]=root;}
                else {
                    out=Allocate(shape[0],total/size_t(shape[0]));slots_[out].shape=shape;
                    Op op{Kind::Layout,a,a,out};op.dimensions={0};ops_.push_back(std::move(op));
                }
            }else if(node.name=="qwen35_chunk_token") {
                auto shape=slots_[a].shape;auto index=node.attributes.Get("index").ToInt64();
                if((shape.size()!=2&&shape.size()!=3)||index<0||index>=shape[1])throw std::runtime_error("Invalid Metal chunk token");
                size_t width=shape.size()==3?size_t(shape[2]):1;
                std::vector<int64_t> outputShape=shape.size()==3?std::vector<int64_t>{shape[0],shape[2]}:std::vector<int64_t>{shape[0]};
                out=slice(a,1,slots_[a].columns,size_t(index)*width,width,outputShape);
                chunkOrigins[out]={a,size_t(index)};
                if(std::find(inputs_.begin(),inputs_.end(),a)!=inputs_.end())controlSources_[out]={a,size_t(index)};
            }else if(node.name=="qwen35_vision_encoder") {
                out=LowerVision(a,node.attributes);
            }else if(node.name=="qwen35_attention_step"||node.name=="qwen_text_attention_step"||node.name=="qwen35_attention_chunk"||node.name=="qwen_text_attention_chunk") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal attention dimension: ")+name);
                    return size_t(v);};
                const auto heads=integer("num_heads",128),kv=integer("num_kv_heads",128),dim=integer("head_dim",4096),rotary=integer("rotary_dim",4096);
                const auto theta=node.attributes.Get("rope_theta").ToDouble(),epsilon=node.attributes.Get("eps").ToDouble();
                auto t=node.attributes.Get("mrope_t").ToInt64(),h=node.attributes.Get("mrope_h").ToInt64(),w=node.attributes.Get("mrope_w").ToInt64();
                if(heads%kv||rotary%2||rotary>dim||t<0||h<0||w<0||t>int64_t(rotary/2)||h>int64_t(rotary/2)||w>int64_t(rotary/2)||
                    uint64_t(t+h+w)!=rotary/2||!(theta>0)||!std::isfinite(theta)||!(epsilon>0)||!std::isfinite(float(epsilon)))
                    throw std::runtime_error("Metal attention RoPE configuration invalid");
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal attention missing binding: ")+name);
                    return it->second;};
                auto key=require("qwen35_attention_bind_key"),value=require("qwen35_attention_bind_value"),positions=require("qwen35_attention_bind_positions"),
                    slot=require("qwen35_attention_bind_slots"),active=require("qwen35_attention_bind_active");
                const bool isChunk=node.name=="qwen35_attention_chunk"||node.name=="qwen_text_attention_chunk";
                if(verificationTokens_) {
                    if(!isChunk||heads>32||kv>8||dim>256)
                        throw std::invalid_argument("Verification requires one bounded packed attention block");
                    ++verificationBlocks;
                }
                const auto hs=slots_[a].shape,cs=slots_[key].shape;
                const size_t chunk=isChunk&&hs.size()==3?size_t(hs[1]):1;
                if(hs.size()!=(isChunk?3:2)||cs.size()!=4||cs[0]!=hs[0]||uint64_t(cs[1])!=kv||cs[2]<=0||cs[2]>8192||uint64_t(cs[3])!=dim||
                    slots_[value].shape!=cs||slots_[positions].shape!=(isChunk?std::vector<int64_t>{hs[0],int64_t(chunk),3}:std::vector<int64_t>{hs[0],3})||
                    slots_[slot].shape!=(isChunk?std::vector<int64_t>{hs[0],int64_t(chunk)}:std::vector<int64_t>{hs[0]})||slots_[active].shape!=slots_[slot].shape||
                    (isChunk&&(chunk>kMetalPrefillChunkLimit||cs[2]>int64_t(kMetalAttentionChunkCapacityLimit))))
                    throw std::runtime_error("Metal full attention state shapes invalid");
                // Control tensors must be explicit host inputs for pre-dispatch bounds checks.
                for(auto control:{positions,slot,active})if(std::find(inputs_.begin(),inputs_.end(),control)==inputs_.end()&&!controlSources_.count(control))
                    throw std::runtime_error("Metal attention control tensors must be graph inputs");
                const size_t batch=hs[0],capacity=cs[2],cacheWidth=kv*capacity*dim,rows=batch*chunk,width=hs.back();
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
                if(isChunk&&(batch>4||width>4096||rows>GARNET_METAL_DENSE_PREFILL_MAX_TOKENS))
                    throw std::invalid_argument("Dense prefill exceeds the compiled physical token-row profile");
#endif
                if(verificationTokens_&&(chunk!=verificationTokens_||rows>4||capacity>4096))
                    throw std::invalid_argument("Verification attention exceeds prepared bounds");
                auto flatten=[&](size_t input,size_t columns){auto output=Allocate(rows,columns);Op op{Kind::Layout,input,input,output};op.dimensions={0};ops_.push_back(std::move(op));return output;};
                if(isChunk)a=flatten(a,width);
                const auto ropePositions=isChunk?flatten(positions,3):positions;
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                const bool text=node.name=="qwen_text_attention_step"||node.name=="qwen_text_attention_chunk",normalize=!text||node.attributes.Get("qk_norm").ToInt64()!=0;
                auto qg=matmul(a,Weight(prefix+".q_proj.weight")),kp=matmul(a,Weight(prefix+".k_proj.weight")),vp=matmul(a,Weight(prefix+".v_proj.weight"));
                if(text&&node.attributes.Get("attention_bias").ToInt64()!=0) {
                    auto bias=[&](size_t input,const std::string& name){auto weight=Weight(name);
                        if(slots_[weight].rows!=1||slots_[weight].columns!=slots_[input].columns)throw std::runtime_error("Metal projection bias shape mismatch");
                        auto result=Allocate(slots_[input].rows,slots_[input].columns);ops_.push_back({Kind::Bias,input,weight,result});return result;};
                    qg=bias(qg,prefix+".q_proj.bias");kp=bias(kp,prefix+".k_proj.bias");vp=bias(vp,prefix+".v_proj.bias");
                }
                size_t qw,kw;
                if(normalize){qw=Weight(prefix+".q_norm.weight");kw=Weight(prefix+".k_norm.weight");}
                else{std::vector<float> ones(dim,1);qw=slots_.size();slots_.push_back({gpu_->Upload(ones,1,dim),1,dim,{1,int64_t(dim)}});kw=qw;}
                if(slots_[qg].columns!=(text?1:2)*heads*dim||slots_[kp].columns!=kv*dim||slots_[vp].columns!=kv*dim||
                    slots_[qw].rows!=1||slots_[qw].columns!=dim||slots_[kw].rows!=1||slots_[kw].columns!=dim)
                    throw std::runtime_error("Metal full attention checkpoint dimensions mismatch");
                auto q=text?qg:slice(qg,heads,2*dim,0,dim),gate=text?q:slice(qg,heads,2*dim,dim,dim);
                std::vector<float> frequencies(rotary);
                for(size_t i=0;i<rotary/2;++i) {
                    frequencies[i]=float(std::pow(theta,-2.0*double(i)/double(rotary)));
                    frequencies[rotary/2+i]=(i%3==1&&i<size_t(h)*3)?1.0f:((i%3==2&&i<size_t(w)*3)?2.0f:0.0f);
                }
                auto freq=slots_.size();slots_.push_back({gpu_->Upload(frequencies,2,rotary/2),2,rotary/2,{2,int64_t(rotary/2)}});
                auto qr=Allocate(rows,heads*dim),kr=Allocate(rows,kv*dim);
                float normOffset=text?0.f:1.f;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
                if(mlxCheckpoint_.Enabled())normOffset=0;
#endif
                Op qrope{Kind::NormRope,q,qw,qr,float(epsilon)};qrope.arguments={q,qw,ropePositions,freq};qrope.dimensions={heads,dim,rotary,size_t(normalize)};qrope.offset=normOffset;ops_.push_back(std::move(qrope));
                Op krope{Kind::NormRope,kp,kw,kr,float(epsilon)};krope.arguments={kp,kw,ropePositions,freq};krope.dimensions={kv,dim,rotary,size_t(normalize)};krope.offset=normOffset;ops_.push_back(std::move(krope));
                auto nextKey=Allocate(batch,cacheWidth),nextValue=Allocate(batch,cacheWidth);
                auto updateCaches=[&]() {
                    for(auto pair:{std::make_pair(kr,std::make_pair(key,nextKey)),std::make_pair(vp,std::make_pair(value,nextValue))}) {
                        Op update{isChunk?Kind::KVChunk:Kind::KVUpdate,pair.first,pair.second.first,pair.second.second};
                        update.arguments={pair.first,pair.second.first,slot,active,positions};update.dimensions={kv,dim,capacity,chunk};ops_.push_back(std::move(update));
                    }
                };
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
                if(!isChunk)updateCaches();
#else
                updateCaches();
#endif
                auto context=Allocate(rows,heads*dim);
                if(isChunk) {
                    Op attend{verificationTokens_?Kind::VerificationAttention:Kind::AttentionChunk,qr,key,context};
                    attend.arguments={qr,kr,vp,key,value,slot,active,gate};
                    if(verificationTokens_)attend.auxiliaryOutput=Allocate(rows,heads*capacity);
                    attend.dimensions={heads,kv,dim,capacity,chunk,size_t(!text)};ops_.push_back(std::move(attend));
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
                    // Attention must see the old cache for earlier queries if
                    // a later token overwrites that slot. Commit updates only
                    // after all queries have consumed old-cache + chunk K/V.
                    updateCaches();
#endif
                } else {
                    auto probabilities=Allocate(batch,heads*capacity);
                    Op scores{Kind::Scores,qr,nextKey,probabilities};scores.arguments={qr,nextKey,slot,active};scores.dimensions={heads,kv,dim,capacity};ops_.push_back(std::move(scores));
                    Op attend{Kind::AttentionContext,probabilities,nextValue,context};attend.arguments={probabilities,nextValue,gate,slot};attend.dimensions={heads,kv,dim,capacity,size_t(!text)};ops_.push_back(std::move(attend));
                }
                auto hidden=matmul(context,Weight(prefix+".o_proj.weight"));
                if(slots_[hidden].columns!=width)throw std::runtime_error("Metal attention output projection shape mismatch");
                if(isChunk) {auto packed=Allocate(batch,chunk*width);Op op{Kind::Layout,hidden,hidden,packed};op.dimensions={0};ops_.push_back(std::move(op));hidden=packed;}
                out=allocatePack(batch,chunk*width+2*cacheWidth);
                Op pack{Kind::Concat,hidden,nextKey,out};pack.arguments={hidden,nextKey,nextValue};pack.transpose=true;packedParts[out]=pack.arguments;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_attention_output"||node.name=="qwen35_attention_key_cache"||node.name=="qwen35_attention_value_cache") {
                auto dimension=[&](const char* name,size_t limit){auto n=node.attributes.Get(name).ToInt64();
                    if(n<=0||uint64_t(n)>limit)throw std::runtime_error("Invalid Metal attention unpack dimension");
                    return size_t(n);};
                auto hidden=dimension("hidden_dim",kMetalPrefillPackedHiddenLimit),kv=dimension("num_kv_heads",128),dim=dimension("head_dim",4096),capacity=dimension("capacity",8192);
                const size_t cache=kv*dim*capacity,width=hidden+2*cache;const auto batch=int64_t(slots_[a].rows);
                if(slots_[a].columns!=width)throw std::runtime_error("Metal attention packed shape mismatch");
                if(node.name=="qwen35_attention_output")out=slice(a,1,width,0,hidden,{batch,int64_t(hidden)});
                else out=slice(a,1,width,hidden+(node.name=="qwen35_attention_value_cache"?cache:0),cache,
                    {batch,int64_t(kv),int64_t(capacity),int64_t(dim)});
            }else if(node.name=="qwen35_linear_step"||node.name=="qwen35_linear_chunk") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal linear dimension: ")+name);
                    return size_t(v);};
                const auto kh=integer("key_heads",128),vh=integer("value_heads",128),kd=integer("key_dim",4096),vd=integer("value_dim",4096);
                if(vh%kh)throw std::runtime_error("Metal linear head grouping mismatch");
                const auto hiddenShape=slots_[a].shape;
                const bool isChunk=node.name=="qwen35_linear_chunk";
                if(verificationTokens_) {
                    if(!isChunk)throw std::invalid_argument("Verification requires one packed recurrent block");
                    ++verificationBlocks;
                }
                const size_t chunk=isChunk&&hiddenShape.size()==3?size_t(hiddenShape[1]):1;
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal linear missing binding: ")+name);
                    return it->second;};
                auto convState=require("qwen35_conv_bind_state"),state=require("qwen35_delta_bind_state"),active=require("qwen35_conv_bind_active");
                const auto cs=slots_[convState].shape;
                const size_t channels=2*kh*kd+vh*vd;
                if(hiddenShape.size()!=(isChunk?3:2)||chunk>kMetalPrefillChunkLimit||cs.size()!=3||cs[0]!=hiddenShape[0]||uint64_t(cs[1])!=channels||cs[2]<=0||cs[2]>64||
                    slots_[state].shape!=std::vector<int64_t>{hiddenShape[0],int64_t(vh),int64_t(kd),int64_t(vd)}||
                    slots_[active].shape!=(isChunk?std::vector<int64_t>{hiddenShape[0],int64_t(chunk)}:std::vector<int64_t>{hiddenShape[0]}))throw std::runtime_error("Metal linear state logical shape mismatch");
                const auto batch=size_t(hiddenShape[0]),kernel=size_t(cs[2]);
                const size_t width=size_t(hiddenShape.back());
                auto layout=[&](size_t source,size_t rows,size_t columns) {
                    auto result=Allocate(rows,columns);Op op{Kind::Layout,source,source,result};op.dimensions={0};ops_.push_back(std::move(op));return result;
                };
                auto source=isChunk?layout(a,batch*chunk,width):a;
                const auto prefix=node.attributes.Get("weight_prefix").ToString();
                const auto epsilon=float(node.attributes.Get("eps").ToDouble());
                if(!(epsilon>0)||!std::isfinite(epsilon))throw std::runtime_error("Invalid Metal linear epsilon");
                auto qkv=matmul(source,Weight(prefix+".in_proj_qkv.weight"));
                auto z=matmul(source,Weight(prefix+".in_proj_z.weight"));
                auto ap=matmul(source,Weight(prefix+".in_proj_a.weight")),bp=matmul(source,Weight(prefix+".in_proj_b.weight"));
                auto convWeight=Weight(prefix+".conv1d.weight"),aLog=Weight(prefix+".A_log"),dt=Weight(prefix+".dt_bias");
                if(slots_[qkv].columns!=channels||slots_[z].columns!=vh*vd||slots_[ap].columns!=vh||slots_[bp].columns!=vh||
                    slots_[convWeight].rows!=channels||slots_[convWeight].columns!=kernel||slots_[aLog].columns!=vh||slots_[dt].columns!=vh)
                    throw std::runtime_error("Metal linear checkpoint projection shape mismatch");
                // Projections share weights across all tokens. State updates
                // retain causal order and honor every row's active mask.
                auto qkvTokens=isChunk?layout(qkv,batch,chunk*channels):qkv;
                auto zTokens=isChunk?layout(z,batch,chunk*vh*vd):z;
                auto aTokens=isChunk?layout(ap,batch,chunk*vh):ap;
                auto bTokens=isChunk?layout(bp,batch,chunk*vh):bp;
                auto norm=Weight(prefix+".norm.weight");
                if(slots_[norm].rows!=1||slots_[norm].columns!=vd)throw std::runtime_error("Metal linear norm weight shape mismatch");
                const size_t qw=vh*kd,vw=vh*vd,pw=2*qw+vw+2*vh;
                bool chunkConv=false;size_t preparedTokens=0,preparedFlat=0;
#ifdef GARNET_METAL_LINEAR_CHUNK_CONV
                chunkConv=isChunk&&kernel<=8;
                if(chunkConv) {
                    auto convolution=Allocate(batch*chunk,channels),nextConv=Allocate(batch,channels*kernel);
                    Op conv{Kind::ConvChunk,qkv,convWeight,convolution};conv.arguments={qkv,convWeight,convState,active};
                    conv.dimensions={chunk,kernel};conv.auxiliaryOutput=nextConv;ops_.push_back(std::move(conv));convState=nextConv;
                    auto prepared=Allocate(batch*chunk,pw);
                    preparedFlat=prepared;
                    Op prepare{Kind::Prepare,convolution,ap,prepared};prepare.arguments={convolution,ap,bp,aLog,dt};
                    prepare.dimensions={kh,vh,kd,vd,0};ops_.push_back(std::move(prepare));
                    preparedTokens=layout(prepared,batch,chunk*pw);
                }
#endif
                bool chunkDelta=false;size_t gatedFlat=0;
#ifdef GARNET_METAL_LINEAR_CHUNK_DELTA
                chunkDelta=chunkConv&&kd==128&&vd%16==0;
                if(chunkDelta) {
                    auto cores=Allocate(batch*chunk,vw),nextState=Allocate(batch,vh*kd*vd);
                    Op delta{Kind::DeltaChunk,preparedFlat,state,cores};delta.arguments={preparedFlat,state,active};
                    delta.dimensions={chunk,vh,kd,vd};delta.auxiliaryOutput=nextState;ops_.push_back(std::move(delta));state=nextState;
                    gatedFlat=Allocate(batch*chunk,vw);
                    Op gate{Kind::GateNorm,cores,z,gatedFlat,epsilon};gate.arguments={cores,z,norm};gate.dimensions={vh,vd};ops_.push_back(std::move(gate));
                }
#endif
                if(verificationTokens_&&(!chunkConv||!chunkDelta||chunk!=verificationTokens_||batch*chunk>4))
                    throw std::invalid_argument("Verification recurrent block requires packed convolution and delta kernels");
                size_t accumulated=0;
                bool directWrite=false;
#ifdef GARNET_METAL_LINEAR_CHUNK_DIRECT_WRITE
                directWrite=isChunk;
                if(directWrite&&!chunkDelta)accumulated=Allocate(batch,chunk*vw);
#endif
                for(size_t token=0;token<chunk&&!chunkDelta;++token) {
                    auto select=[&](size_t projected,size_t columns) {
                        return isChunk?slice(projected,1,chunk*columns,token*columns,columns):projected;
                    };
                    auto tokenActive=isChunk?slice(active,1,chunk,token,1,{int64_t(batch)}):active;
                    auto tokenZ=select(zTokens,vw);size_t convPacked=0,prepared=0;
                    if(chunkConv)prepared=select(preparedTokens,pw);
                    else {
                        auto tokenQkv=select(qkvTokens,channels),tokenA=select(aTokens,vh),tokenB=select(bTokens,vh);
                        convPacked=Allocate(batch,channels*(kernel+1));
                        Op conv{Kind::Conv,tokenQkv,convWeight,convPacked};conv.arguments={tokenQkv,convWeight,convState,tokenActive};conv.dimensions={kernel};ops_.push_back(std::move(conv));
                        prepared=Allocate(batch,pw);
                        Op prepare{Kind::Prepare,convPacked,tokenA,prepared};prepare.arguments={convPacked,tokenA,tokenB,aLog,dt};
                        prepare.dimensions={kh,vh,kd,vd,kernel};ops_.push_back(std::move(prepare));
                    }
                    auto q=slice(prepared,1,pw,0,qw),k=slice(prepared,1,pw,qw,qw),v=slice(prepared,1,pw,2*qw,vw),
                        g=slice(prepared,1,pw,2*qw+vw,vh),beta=slice(prepared,1,pw,2*qw+vw+vh,vh);
                    auto deltaPacked=Allocate(batch,vw*(kd+1));
                    Op delta{Kind::Delta,q,k,deltaPacked};delta.arguments={q,k,v,g,beta,state,tokenActive};delta.dimensions={vh,kd,vd};ops_.push_back(std::move(delta));
                    auto core=slice(deltaPacked,vh,(kd+1)*vd,0,vd);
                    auto gated=directWrite?accumulated:Allocate(batch,vw);
                    Op gate{Kind::GateNorm,core,tokenZ,gated,epsilon};gate.arguments={core,tokenZ,norm};gate.dimensions={vh,vd};ops_.push_back(std::move(gate));
                    if(directWrite)ops_.back().dimensions.push_back(token*vw);
                    else if(token==0)accumulated=gated;
                    else {
                        auto joined=Allocate(batch,(token+1)*vw);Op join{Kind::Concat,accumulated,gated,joined};
                        join.arguments={accumulated,gated,gated};join.transpose=false;ops_.push_back(std::move(join));accumulated=joined;
                    }
                    if(!chunkConv)convState=slice(convPacked,channels,kernel+1,1,kernel);
                    state=slice(deltaPacked,vh,(kd+1)*vd,vd,kd*vd);
                }
                auto gated=chunkDelta?gatedFlat:(isChunk?layout(accumulated,batch*chunk,vw):accumulated);
                auto hidden=matmul(gated,Weight(prefix+".out_proj.weight"));
                if(slots_[hidden].columns!=width)throw std::runtime_error("Metal linear output projection shape mismatch");
                if(isChunk)hidden=layout(hidden,batch,chunk*width);
                out=Allocate(batch,slots_[hidden].columns+slots_[convState].columns+slots_[state].columns);
                Op pack{Kind::Concat,hidden,convState,out};pack.arguments={hidden,convState,state};pack.transpose=true;packedParts[out]=pack.arguments;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_linear_output"||node.name=="qwen35_linear_conv_state"||node.name=="qwen35_linear_recurrent_state") {
                auto dim=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error("Invalid Metal linear unpack dimension");
                    return size_t(v);};
                const auto h=dim("hidden_dim",kMetalPrefillPackedHiddenLimit),kh=dim("key_heads",128),vh=dim("value_heads",128),kd=dim("key_dim",4096),
                    vd=dim("value_dim",4096),kernel=dim("kernel_size",64),channels=2*kh*kd+vh*vd,conv=channels*kernel,state=vh*kd*vd;
                if(vh%kh||slots_[a].columns!=h+conv+state)throw std::runtime_error("Metal linear packed dimensions mismatch");
                const auto batch=int64_t(slots_[a].rows);
                if(node.name=="qwen35_linear_output")out=slice(a,1,h+conv+state,0,h,{batch,int64_t(h)});
                else if(node.name=="qwen35_linear_conv_state")out=slice(a,1,h+conv+state,h,conv,{batch,int64_t(channels),int64_t(kernel)});
                else out=slice(a,1,h+conv+state,h+conv,state,{batch,int64_t(vh),int64_t(kd),int64_t(vd)});
            }else if(node.name=="qwen35_conv_output"||node.name=="qwen35_conv_state"||
                node.name=="qwen35_delta_output"||node.name=="qwen35_delta_state") {
                const auto shape=slots_[a].shape;
                if(shape.size()!=3)throw std::runtime_error("Metal attention unpack requires rank three packed input");
                const bool delta=node.name.rfind("qwen35_delta_",0)==0,wantState=node.name.find("_state")!=std::string::npos;
                size_t start=0,count=1;std::vector<int64_t> outShape;
                if(delta) {
                    const auto kd=node.attributes.Get("key_dim").ToInt64(),vd=node.attributes.Get("value_dim").ToInt64();
                    if(kd<=0||vd<=0||uint64_t(kd)>=std::numeric_limits<size_t>::max()/uint64_t(vd)||
                        uint64_t(shape[2])!=(uint64_t(kd)+1)*uint64_t(vd))throw std::runtime_error("Metal delta unpack dimensions invalid");
                    start=wantState?vd:0;count=wantState?size_t(kd)*size_t(vd):size_t(vd);
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kd,vd}:std::vector<int64_t>{shape[0],shape[1],vd};
                }else {
                    const auto kernel=node.attributes.Get("kernel_size").ToInt64();
                    if(kernel<=0||uint64_t(kernel)==std::numeric_limits<size_t>::max()||uint64_t(shape[2])!=uint64_t(kernel)+1)
                        throw std::runtime_error("Metal convolution unpack dimensions invalid");
                    start=wantState?1:0;count=wantState?kernel:1;
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kernel}:std::vector<int64_t>{shape[0],shape[1]};
                }
                out=Allocate(slots_[a].rows,size_t(shape[1])*count);slots_[out].shape=std::move(outShape);
                Op op{Kind::Slice,a,a,out};op.dimensions={size_t(shape[1]),size_t(shape[2]),start,count};ops_.push_back(std::move(op));
            }else if(node.name=="qwen35_conv_step"||node.name=="qwen35_delta_step") {
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const std::string& name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error("Metal attention missing binding: "+name);
                    return it->second;};
                const auto shape=slots_[a].shape;
                if(node.name=="qwen35_conv_step") {
                    auto w=require("qwen35_conv_bind_weight"),s=require("qwen35_conv_bind_state"),active=require("qwen35_conv_bind_active");
                    const auto ss=slots_[s].shape;
                    if(shape.size()!=2||ss.size()!=3||ss[0]!=shape[0]||ss[1]!=shape[1]||
                        slots_[w].shape!=std::vector<int64_t>{shape[1],ss[2]}||slots_[active].shape!=std::vector<int64_t>{shape[0]})
                        throw std::runtime_error("Metal causal convolution logical shape mismatch");
                    const size_t kernel=ss[2];
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[a].columns)
                        throw std::runtime_error("Metal convolution packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[a].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t(kernel+1)};
                    Op op{Kind::Conv,a,w,out};op.arguments={a,w,s,active};op.dimensions={kernel};ops_.push_back(std::move(op));
                }else {
                    auto k=require("qwen35_delta_bind_key"),v=require("qwen35_delta_bind_value"),g=require("qwen35_delta_bind_decay"),
                        beta=require("qwen35_delta_bind_beta"),s=require("qwen35_delta_bind_state"),active=require("qwen35_delta_bind_active");
                    const auto vs=slots_[v].shape;
                    if(shape.size()!=3||vs.size()!=3||vs[0]!=shape[0]||vs[1]!=shape[1]||slots_[k].shape!=shape||
                        slots_[s].shape!=std::vector<int64_t>{shape[0],shape[1],shape[2],vs[2]}||
                        slots_[g].shape!=std::vector<int64_t>{shape[0],shape[1]}||slots_[beta].shape!=slots_[g].shape||
                        slots_[active].shape!=std::vector<int64_t>{shape[0]})throw std::runtime_error("Metal gated delta logical shape mismatch");
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[v].columns)
                        throw std::runtime_error("Metal delta packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[v].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t((size_t(shape[2])+1)*size_t(vs[2]))};
                    Op op{Kind::Delta,a,k,out};op.arguments={a,k,v,g,beta,s,active};
                    op.dimensions={size_t(shape[1]),size_t(shape[2]),size_t(vs[2])};ops_.push_back(std::move(op));
                }
            }else if(node.name=="qwen35_rms_norm"||node.name=="qwen_text_norm") {
                auto weight=Weight(node.attributes.Get("weight_name").ToString());
                if(slots_[weight].rows!=1||slots_[weight].columns!=slots_[a].columns)
                    throw std::runtime_error("Metal normalization shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                float offset=node.name=="qwen_text_norm"?0.f:1.f;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
                if(mlxCheckpoint_.Enabled())offset=0;
#endif
                ops_.push_back({Kind::Norm,a,weight,out,float(node.attributes.Get("eps").ToDouble()),offset,false});
            }else if(node.name=="qwen35_mtp_fc"&&mtpProgram_) {
                out=matmul(a,Weight("mtp.fc.weight"));
            }else if(node.name=="medusa_residual_head") {
                // Explicit raw-BF16 draft head. Target weights/state are not
                // modified; accepted tokens still require target verification.
                const auto head=node.attributes.Get("head_index").ToInt64();
                const auto features=node.attributes.Get("emit_features").ToInt64();
                if(head<0||head>=4||slots_[a].shape.size()!=2||!slots_[a].rows||slots_[a].rows>2||
                   slots_[a].columns!=4096||mlxCheckpoint_.Enabled()||(features!=0&&features!=1))
                    throw std::invalid_argument("Medusa head requires raw BF16 weights and a bounded1/2x4096 input");
                const auto prefix=std::to_string(head);
                auto check=[&](const std::string& name,const std::vector<long long>& shape) {
                    const auto* metadata=FindWeightMetadata(name);
                    if(!metadata||metadata->dataType!="BF16"||metadata->shape!=shape)
                        throw std::invalid_argument("Medusa BF16 weight shape/type mismatch: "+name);
                };
                check(prefix+".0.linear.weight",{4096,4096});check(prefix+".0.linear.bias",{4096});
                check(prefix+".1.weight",{32768,4096});
                // Reserve a conservative bound for optional retained BF16
                // layouts as well as raw weights and the small FP32 workspace.
                const auto memory=MetalContext::MemoryStats();
                constexpr uint64_t margin=256ull*1024*1024;
                constexpr uint64_t extra=3ull*(4096*4096+32768*4096)*2+8ull*1024*1024;
                if(memory.recommended<=margin+extra||memory.allocated>memory.recommended-margin-extra)
                    throw std::runtime_error("Medusa head exceeds the native working-set guard");
                auto linear=matmul(a,BF16Weight(prefix+".0.linear.weight"));
                auto biased=Allocate(slots_[a].rows,4096),activated=Allocate(slots_[a].rows,4096),residual=Allocate(slots_[a].rows,4096);
                ops_.push_back({Kind::Bias,linear,Weight(prefix+".0.linear.bias"),biased});
                const auto ones=slots_.size();std::vector<float> values(slots_[a].rows*4096,1.f);
                slots_.push_back({gpu_->Upload(values,slots_[a].rows,4096),slots_[a].rows,4096,{int64_t(slots_[a].rows),4096}});
                ops_.push_back({Kind::Silu,biased,ones,activated});ops_.push_back({Kind::Add,a,activated,residual});
                out=matmul(residual,BF16Weight(prefix+".1.weight"));
                if(features) {
                    auto packed=Allocate(slots_[a].rows,4096+32768);
                    Op capture{Kind::Concat,residual,out,packed};capture.arguments={residual,out,out};
                    ops_.push_back(std::move(capture));out=packed;
                }
            }else if(node.name=="qwen35_mlp") {
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                auto gate=matmul(a,Weight(prefix+".gate_proj.weight"));
                auto up=matmul(a,Weight(prefix+".up_proj.weight"));
                if(slots_[gate].columns!=slots_[up].columns)throw std::runtime_error("Metal gate/up shape mismatch");
                auto activated=Allocate(slots_[gate].rows,slots_[gate].columns);
                ops_.push_back({Kind::Silu,gate,up,activated});
                out=matmul(activated,Weight(prefix+".down_proj.weight"));
            }else if(node.name=="qwen35_tied_head"||node.name=="qwen_text_head") {
                const auto name=node.name=="qwen35_tied_head"?"model.language_model.embed_tokens.weight":node.attributes.Get("weight_name").ToString();
                const bool bf16=node.attributes.Get("bf16_weight").ToInt64()!=0;
                auto weight=bf16?BF16Weight(name):Weight(name);
#ifdef GARNET_METAL_SCREENED_HEAD
                if(bf16&&slots_[weight].bf16.Valid())gpu_->PrepareScreenedHead(slots_[weight].bf16);
#endif
#ifdef GARNET_METAL_MLX_AFFINE_INT4
                if(slots_[weight].affineInt4.Valid())out=matmul(a,weight);
                else
#endif
                if(bf16&&FindWeightMetadata(name)->dataType=="BF16") {
                    if(slots_[a].columns!=slots_[weight].columns||slots_[a].rows>2)throw std::runtime_error("BF16 vocabulary profile requires batch 1/2");
                    out=Allocate(slots_[a].rows,slots_[weight].rows);ops_.push_back({Kind::BF16MatVec,a,weight,out});
                }else out=matmul(a,weight);
            }
            else throw std::runtime_error("Metal lowering not implemented: "+node.name);
            tensors_[TensorId(node.output)]=out;
        }
        auto output=capture.Outputs();
        if((output.IsList()||x3_value_object_kind(output.raw())==X3_OBJECT_KIND_TUPLE)&&output.Size()==1)
            output=output.Get(uint64_t(0));
        if(!IsTensor(output))throw std::runtime_error("Initial Metal graph requires one tensor output");
        output_=lookup(output);
        if(packedParts.count(output_))outputParts_=packedParts.at(output_);
        if(verificationTokens_) {
            if(verificationBlocks!=1||outputParts_.size()!=3||
               slots_[outputParts_[0]].rows!=size_t(inputShapes_[0][0])||
               slots_[outputParts_[0]].columns!=verificationTokens_*size_t(inputShapes_[0][2])||
               slots_[outputParts_[1]].columns!=slots_[inputs_[1]].columns||
               slots_[outputParts_[2]].columns!=slots_[inputs_[2]].columns)
                throw std::invalid_argument("Verification must return one complete decoder layer");
            // Every weight must have been shared from this layer's source;
            // another layer with coincidentally equal shapes is not its verifier.
            for(const auto& entry:weightRecords_)if(!sharedWeights_.count(entry.first))
                throw std::invalid_argument("Verification used a weight outside its source decoder layer");
        }
        // All operations write fresh slots. Trace only matrices needed by the
        // returned value, resolving storage aliases before removing unused packs.
        auto canonical=[&](size_t id){while(aliases.count(id))id=aliases.at(id);return id;};
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
        // Propagate request-row identity from a 2D decoder/head hidden input.
        // Stateful operations must use this graph's active controls; unrelated
        // matrices with the same row count do not establish that identity.
        if(!inputShapes_.empty()&&inputShapes_[0].size()==2) {
            const auto rows=size_t(inputShapes_[0][0]);size_t activeId=SIZE_MAX;
            const bool head=inputs_.size()==1;
            if(inputs_.size()==4&&inputShapes_[3]==std::vector<int64_t>{int64_t(rows)})activeId=canonical(inputs_[3]);
            if(inputs_.size()==6&&inputShapes_[3]==std::vector<int64_t>{int64_t(rows),3}&&
               inputShapes_[4]==std::vector<int64_t>{int64_t(rows)}&&inputShapes_[5]==inputShapes_[4])activeId=canonical(inputs_[5]);
            if(rows&&rows<=4&&(head||activeId!=SIZE_MAX)) {
                std::vector<bool> layout(slots_.size(),false);layout[canonical(inputs_[0])]=true;
                auto rowwise=[&](size_t id){return id<slots_.size()&&layout[canonical(id)]&&slots_[id].rows==rows;};
                auto active=[&](size_t id){return activeId!=SIZE_MAX&&canonical(id)==activeId;};
                for(auto& op:ops_) {
                    bool known=false;
                    switch(op.kind) {
                    case Kind::MatMul:
                    case Kind::BF16MatVec:
                        known=rowwise(op.a);
                        if(known&&op.kind==Kind::MatMul&&op.transpose&&slots_[op.b].affineInt4.Valid())op.indexedDecodeProjection=true;
                        break;
                    case Kind::Norm:case Kind::Bias:case Kind::Mask:case Kind::NormRope:case Kind::Slice:
                        known=rowwise(op.a);break;
                    case Kind::Layout:
                        known=rowwise(op.a)&&op.dimensions.size()==1&&slots_[op.out].columns==slots_[op.a].columns&&
                            (!op.dimensions[0]||(active(op.b)&&op.dimensions[0]==slots_[op.a].columns));break;
                    case Kind::Add:case Kind::Silu:
                        known=rowwise(op.a)&&rowwise(op.b);break;
                    case Kind::Conv:
                        known=rowwise(op.a)&&op.arguments.size()==4&&active(op.arguments[3]);break;
                    case Kind::Prepare:
                        known=op.arguments.size()==5&&rowwise(op.arguments[0])&&rowwise(op.arguments[1])&&rowwise(op.arguments[2]);break;
                    case Kind::Delta:
                        known=op.arguments.size()==7&&rowwise(op.arguments[0])&&rowwise(op.arguments[1])&&rowwise(op.arguments[2])&&
                            rowwise(op.arguments[3])&&rowwise(op.arguments[4])&&active(op.arguments[6]);break;
                    case Kind::GateNorm:
                        known=rowwise(op.a)&&rowwise(op.b)&&op.dimensions.size()==2&&slots_[op.out].columns==op.dimensions[0]*op.dimensions[1];break;
                    case Kind::Scores:
                        known=rowwise(op.a)&&op.arguments.size()==4&&active(op.arguments[3]);break;
                    case Kind::AttentionContext:
                        known=op.arguments.size()>=3&&rowwise(op.arguments[0])&&rowwise(op.arguments[2]);break;
                    default:break;
                    }
                    if(known&&slots_[op.out].rows==rows)layout[canonical(op.out)]=true;
                }
            }
        }
#endif
#ifdef GARNET_METAL_INDEXED_TOKEN_PROJECTIONS
        // Prove token-row order from the hidden input through known lowering
        // operations. Shape coincidence alone never enables indexed dispatch.
        if(inputs_.size()==6&&inputShapes_[0].size()==3) {
            const auto batch=size_t(inputShapes_[0][0]),chunk=size_t(inputShapes_[0][1]);
            if(batch&&batch<=512&&(chunk>1||verificationTokens_)&&chunk<=512/batch&&
               inputShapes_[3]==std::vector<int64_t>{int64_t(batch),int64_t(chunk),3}&&
               inputShapes_[4]==std::vector<int64_t>{int64_t(batch),int64_t(chunk)}&&inputShapes_[5]==inputShapes_[4]) {
                const size_t rows=batch*chunk;
                // A one-token block has both physical layouts. Retain both
                // proven identities instead of arbitrarily dropping one.
                std::vector<unsigned char> layout(slots_.size(),0); // bit1 flat,bit2 packed;0 unknown/state.
                layout[canonical(inputs_[0])]=2;
                auto flat=[&](size_t id){return id<slots_.size()&&(layout[canonical(id)]&1)&&slots_[id].rows==rows;};
                auto packed=[&](size_t id){return id<slots_.size()&&(layout[canonical(id)]&2)&&slots_[id].rows==batch&&slots_[id].columns%chunk==0;};
                auto active=[&](size_t id){return canonical(id)==canonical(inputs_[5]);};
                for(auto& op:ops_) {
                    bool outFlat=false,outPacked=false;
                    switch(op.kind) {
                    case Kind::Layout:
                        if(op.dimensions.size()==1&&!op.dimensions[0]) {
                            outFlat=(flat(op.a)&&slots_[op.out].columns==slots_[op.a].columns)||
                                (packed(op.a)&&slots_[op.out].columns==slots_[op.a].columns/chunk);
                            outPacked=(packed(op.a)&&slots_[op.out].columns==slots_[op.a].columns)||
                                (flat(op.a)&&slots_[op.out].columns/chunk==slots_[op.a].columns&&slots_[op.out].columns%chunk==0);
                        }else if(op.dimensions.size()==1&&packed(op.a)&&active(op.b)&&
                            op.dimensions[0]==slots_[op.a].columns/chunk&&slots_[op.out].columns==slots_[op.a].columns)
                            outPacked=true;
                        break;
                    case Kind::MatMul:
                    case Kind::BF16MatVec:
                        outFlat=flat(op.a);
                        if(outFlat&&op.kind==Kind::MatMul&&op.transpose&&slots_[op.b].affineInt4.Valid())
                            {op.indexedTokenProjection=!verificationTokens_;op.indexedDecodeProjection=verificationTokens_!=0;}
                        break;
                    case Kind::Norm:
                    case Kind::Bias:
                    case Kind::Mask:
                    case Kind::NormRope:
                    case Kind::Slice:
                        outFlat=flat(op.a);break;
                    case Kind::Add:
                    case Kind::Silu:
                        outFlat=flat(op.a)&&flat(op.b);
                        outPacked=packed(op.a)&&packed(op.b)&&slots_[op.a].columns==slots_[op.b].columns;
                        break;
                    case Kind::ConvChunk:
                        outFlat=flat(op.a)&&op.arguments.size()==4&&active(op.arguments[3])&&
                            op.dimensions.size()==2&&op.dimensions[0]==chunk;break;
                    case Kind::Prepare:
                        outFlat=op.arguments.size()==5&&flat(op.arguments[0])&&flat(op.arguments[1])&&flat(op.arguments[2]);break;
                    case Kind::DeltaChunk:
                        outFlat=flat(op.a)&&op.arguments.size()==3&&active(op.arguments[2])&&
                            op.dimensions.size()==4&&op.dimensions[0]==chunk;break;
                    case Kind::GateNorm:
                        outFlat=flat(op.a)&&flat(op.b)&&op.dimensions.size()>=2&&
                            (op.dimensions.size()==2||op.dimensions[2]==0)&&
                            slots_[op.out].columns==op.dimensions[0]*op.dimensions[1];break;
                    case Kind::AttentionChunk:
                    case Kind::VerificationAttention:
                        outFlat=op.arguments.size()==8&&flat(op.arguments[0])&&flat(op.arguments[1])&&flat(op.arguments[2])&&
                            active(op.arguments[6])&&op.dimensions.size()==6&&op.dimensions[4]==chunk;break;
                    default:break;
                    }
                    unsigned char proven=0;
                    if(outFlat&&slots_[op.out].rows==rows)proven|=1;
                    if(outPacked&&slots_[op.out].rows==batch)proven|=2;
                    if(proven)layout[canonical(op.out)]=proven;
                }
            }
        }
#endif
        if(verificationTokens_)for(const auto& op:ops_)if(op.kind==Kind::MatMul||op.kind==Kind::BF16MatVec) {
#ifdef GARNET_METAL_MLX_AFFINE_INT4
            if(op.kind!=Kind::MatMul||!op.transpose||!slots_[op.b].affineInt4.Valid()||
               !op.indexedDecodeProjection||op.indexedTokenProjection||
               slots_[op.a].rows!=size_t(inputShapes_[0][0])*verificationTokens_)
#endif
                throw std::invalid_argument("Verification projection lacks proven FP32 token-row identity at slot "+std::to_string(op.a));
        }
        // A delta temporary consumed only by its complete output/state slices
        // can write those independent matrices directly, without packing/copying.
        std::unordered_set<size_t> removedSlices;
        for(size_t index=0;index<ops_.size();++index) {
            auto& delta=ops_[index];if(delta.kind!=Kind::Delta)continue;
            const auto packed=canonical(delta.out),heads=delta.dimensions[0],kd=delta.dimensions[1],vd=delta.dimensions[2];
            if(packed==canonical(output_))continue;
            size_t core=SIZE_MAX,state=SIZE_MAX;bool eligible=true;
            for(size_t consumer=0;consumer<ops_.size();++consumer) {
                if(consumer==index)continue;
                const auto& op=ops_[consumer];
                bool uses=canonical(op.a)==packed||canonical(op.b)==packed;
                for(auto arg:op.arguments)uses=uses||canonical(arg)==packed;
                if(!uses)continue;
                if(op.kind!=Kind::Slice||consumer<=index||op.dimensions.size()!=4||
                    op.dimensions[0]!=heads||op.dimensions[1]!=(kd+1)*vd){eligible=false;break;}
                if(op.dimensions[2]==0&&op.dimensions[3]==vd&&core==SIZE_MAX)core=consumer;
                else if(op.dimensions[2]==vd&&op.dimensions[3]==kd*vd&&state==SIZE_MAX)state=consumer;
                else {eligible=false;break;}
            }
            if(eligible&&core!=SIZE_MAX&&state!=SIZE_MAX) {
                delta.kind=Kind::DeltaSplit;delta.out=ops_[core].out;delta.auxiliaryOutput=ops_[state].out;
                removedSlices.insert(core);removedSlices.insert(state);
            }
        }
        if(!removedSlices.empty()) {
            std::vector<Op> fused;
            for(size_t index=0;index<ops_.size();++index)if(!removedSlices.count(index))fused.push_back(std::move(ops_[index]));
            ops_=std::move(fused);
        }
        std::unordered_set<size_t> live={canonical(output_)};
        std::vector<Op> retained;
        for(auto it=ops_.rbegin();it!=ops_.rend();++it) {
            if(!live.count(canonical(it->out))&&(it->auxiliaryOutput==SIZE_MAX||!live.count(canonical(it->auxiliaryOutput))))continue;
            live.insert(canonical(it->out));if(it->auxiliaryOutput!=SIZE_MAX)live.insert(canonical(it->auxiliaryOutput));
            live.insert(canonical(it->a));live.insert(canonical(it->b));
            for(auto arg:it->arguments)live.insert(canonical(arg));
            retained.push_back(std::move(*it));
        }
        std::reverse(retained.begin(),retained.end());ops_=std::move(retained);
        if(outputParts_.size()==3) {
            std::unordered_set<size_t> needed;
            for(auto part:outputParts_)needed.insert(canonical(part));
            residentOps_.assign(ops_.size(),0);
            for(size_t i=ops_.size();i-->0;) {
                auto& op=ops_[i];if(!needed.count(canonical(op.out))&&(op.auxiliaryOutput==SIZE_MAX||!needed.count(canonical(op.auxiliaryOutput))))continue;
                residentOps_[i]=1;needed.insert(canonical(op.a));needed.insert(canonical(op.b));
                for(auto argument:op.arguments)needed.insert(canonical(argument));
            }
        }
        for(auto id:inputs_)live.insert(canonical(id));
        for(size_t id=0;id<slots_.size();++id)if(!live.count(canonical(id)))slots_[id].matrix=MetalMatrix();
#ifdef GARNET_METAL_MTP_DEFER_PACKS
        for(auto id:deferredPacks) {
            if(!live.count(canonical(id)))continue;
            // Only independent complete concat buffers absent from every
            // resident operation may stay unallocated. Native execution
            // returns three independent parts; standalone Run rehydrates.
            bool resident=outputParts_.size()!=3,aliased=false;size_t writers=0;
            auto uses=[&](const Op& op){
                if(op.a==id||op.b==id||op.out==id||op.auxiliaryOutput==id)return true;
                return std::find(op.arguments.begin(),op.arguments.end(),id)!=op.arguments.end();
            };
            for(const auto& alias:aliases)if(canonical(alias.first)==id)aliased=true;
            for(size_t i=0;i<ops_.size();++i) {
                const auto& op=ops_[i];
                if(op.out==id){if(op.kind!=Kind::Concat)resident=true;++writers;}
                if(i<residentOps_.size()&&residentOps_[i]&&uses(op))resident=true;
            }
            if(!resident&&!aliased&&writers==1)lazyPacks.push_back(id);
            else {
                auto& slot=slots_[id];slot.matrix=gpu_->Allocate(slot.rows,slot.columns);
                for(const auto& alias:aliases)if(canonical(alias.first)==id) {
                    auto& view=slots_[alias.first];
                    if(view.rows!=slot.rows||view.columns!=slot.columns)throw std::logic_error("Deferred pack alias layout mismatch");
                    view.matrix=slot.matrix;
                }
            }
        }
#endif
        // Exchange only fresh, independently written output storage. Keep
        // aliases together and retain the copy path for inputs/constants or
        // storage shared by two returned parts.
        outputStorageGroups_.resize(outputParts_.size());
        for(size_t part=0;part<outputParts_.size();++part) {
            const auto& matrix=slots_[outputParts_[part]].matrix;std::vector<size_t> group;
            bool eligible=true,writer=false;
            for(size_t id=0;id<slots_.size();++id)if(matrix.SharesStorage(slots_[id].matrix)) {
                group.push_back(id);
                if(!matrix.CompatibleLayout(slots_[id].matrix)||std::find(inputs_.begin(),inputs_.end(),id)!=inputs_.end())eligible=false;
            }
            for(size_t other=0;other<outputParts_.size();++other)if(other!=part&&matrix.SharesStorage(slots_[outputParts_[other]].matrix))eligible=false;
            for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
                (std::find(group.begin(),group.end(),ops_[i].out)!=group.end()||std::find(group.begin(),group.end(),ops_[i].auxiliaryOutput)!=group.end()))writer=true;
            if(eligible&&writer)outputStorageGroups_[part]=std::move(group);
        }
        resetKVPrefix_[0]=ComputeResetKVPrefix(1);resetKVPrefix_[1]=ComputeResetKVPrefix(2);
#ifdef GARNET_METAL_CHUNK_ATTENTION_MPS
        for(const auto& op:ops_)if(op.kind==Kind::AttentionChunk) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu_->PrepareAttentionChunk(args,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
        }
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
        // Profiles are upper bounds: valid row counts/masks vary at execution.
        // Prepare before readiness and before singleton scratch is retired.
        for(const auto& op:ops_)if(op.kind==Kind::MatMul&&slots_[op.b].affineInt4.Valid())
            if(!gpu_->PrepareAffineInt4Profile(slots_[op.a].matrix,slots_[op.b].affineInt4,
                slots_[op.out].matrix,slots_[op.a].rows))
                throw std::runtime_error("Cannot prepare bounded affine kernel profile");
#endif
#ifdef GARNET_METAL_BODY_SHAPE
        // Prepare before readiness/compile timing ends. Decode never lazily compiles.
        for(const auto& op:ops_)if((op.kind==Kind::BF16MatVec||(op.kind==Kind::MatMul&&op.transpose))&&slots_[op.b].bf16.Valid())
            gpu_->PrepareBF16BodyShape(slots_[op.a].matrix,slots_[op.b].bf16,slots_[op.out].matrix);
#endif
        if(mtpProgram_) {
            ValidateMTPProgram();const auto memory=MetalContext::MemoryStats();
            if(memory.recommended<=256*1024*1024||memory.allocated>memory.recommended-256*1024*1024)
                throw std::runtime_error("MTP graph exceeds the native working-set guard");
        }
        sharedWeights_.clear();ready_=true;
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
        // Metadata is enough to construct private session state. Retire proved
        // resident-unused defaults before the next layer is built, avoiding a
        // model-wide construction peak. Standalone Run rehydrates the plan.
        RetireResidentDefaults();
#endif
#ifdef GARNET_METAL_MTP_DEFER_PACKS
        if(!lazyPacks.empty()&&!lazyDefaultPlanReady_)throw std::logic_error("Deferred packs require standalone rehydration plan");
        lazyDefaultSlots_.insert(lazyDefaultSlots_.end(),lazyPacks.begin(),lazyPacks.end());
        mtpDeferredPackSlots_=lazyPacks.size();
#endif
        compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return true;
    }catch(...){Clear();throw;}
}
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
bool MetalCompiledGraph::CanPoolResidentHidden() const {
    return !mtpProgram_&&inputs_.size()==6&&inputShapes_[0].size()==3&&inputShapes_[0][1]>1&&
        outputParts_.size()==3&&outputStorageGroups_.size()==3&&!outputStorageGroups_[0].empty()&&
        slots_[outputParts_[0]].rows==size_t(inputShapes_[0][0])&&
        slots_[outputParts_[0]].columns==size_t(inputShapes_[0][1])*size_t(inputShapes_[0][2]);
}
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
bool MetalCompiledGraph::CanUpdateLinearStateInPlace(size_t part) const {
    if(part<1||part>2||mtpProgram_||inputs_.size()!=6||inputShapes_[0].size()!=3||inputShapes_[0][1]<=1||
       outputStorageGroups_.size()!=3||outputStorageGroups_[part].empty()||
       inputShapes_[1].size()!=3||inputShapes_[2].size()!=4)return false;
    const auto batch=inputShapes_[0][0],chunk=inputShapes_[0][1];
    if(inputShapes_[5]!=std::vector<int64_t>{batch,chunk})return false;
    const auto& group=outputStorageGroups_[part];size_t writer=SIZE_MAX;
    auto member=[&](size_t id){return std::find(group.begin(),group.end(),id)!=group.end();};
    for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
        (member(ops_[i].out)||member(ops_[i].auxiliaryOutput))) {
        const auto& op=ops_[i];
        if(writer!=SIZE_MAX||!member(op.auxiliaryOutput)||op.dimensions.empty()||op.dimensions[0]!=size_t(chunk))return false;
        if(part==1) {
            if(op.kind!=Kind::ConvChunk||op.arguments.size()!=4||op.arguments[2]!=inputs_[1]||op.arguments[3]!=inputs_[5]||
               op.dimensions.size()!=2||op.dimensions[1]<1||op.dimensions[1]>8||
               inputShapes_[1][2]!=int64_t(op.dimensions[1]))return false;
        }else {
            if(op.kind!=Kind::DeltaChunk||op.arguments.size()!=3||op.arguments[1]!=inputs_[2]||op.arguments[2]!=inputs_[5]||
               op.dimensions.size()!=4||op.dimensions[2]!=128||!op.dimensions[3]||op.dimensions[3]>4096||op.dimensions[3]%16||
               inputShapes_[2]!=std::vector<int64_t>{batch,int64_t(op.dimensions[1]),128,int64_t(op.dimensions[3])})return false;
        }
        writer=i;
    }
    if(writer==SIZE_MAX)return false;
    const auto& source=slots_[inputs_[part]].matrix;
    for(size_t i=writer+1;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]) {
        const auto& op=ops_[i];
        if(op.a==inputs_[part]||op.b==inputs_[part]||source.SharesStorage(slots_[op.a].matrix)||source.SharesStorage(slots_[op.b].matrix))return false;
        for(auto id:op.arguments)if(id==inputs_[part]||source.SharesStorage(slots_[id].matrix))return false;
    }
    return true;
}
#endif
bool MetalCompiledGraph::CanUpdateKVInPlace(size_t part) const {
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
    if(CanUpdateLinearStateInPlace(part))return true;
#endif
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
    if(inputShapes_.empty())return false;
    const bool chunk=inputShapes_[0].size()==3;
    bool admitted=inputShapes_[0].size()==2;
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
    admitted=admitted||chunk;
#endif
    size_t cacheInput=part;
    bool inputCount=inputs_.size()==6;
#ifdef GARNET_METAL_MTP_KV_INPLACE
    // The validated raw auxiliary graph has two hidden inputs before K/V.
    if(mtpProgram_&&inputs_.size()==7){inputCount=true;cacheInput=part+1;}
#endif
    if(part<1||part>2||!inputCount||!admitted||
        part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto& group=outputStorageGroups_[part];size_t writer=SIZE_MAX;
    auto member=[&](size_t id){return std::find(group.begin(),group.end(),id)!=group.end();};
    for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
        (member(ops_[i].out)||member(ops_[i].auxiliaryOutput))) {
        const auto& op=ops_[i];
        if(writer!=SIZE_MAX||op.kind!=(chunk?Kind::KVChunk:Kind::KVUpdate)||op.arguments.size()!=5||op.arguments[1]!=inputs_[cacheInput])return false;
        writer=i;
    }
    if(writer==SIZE_MAX)return false;
    // Later operations must consume the updated cache, not its old input view.
    const auto& source=slots_[inputs_[cacheInput]].matrix;
    for(size_t i=writer+1;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]) {
        const auto& op=ops_[i];
        if(op.a==inputs_[cacheInput]||op.b==inputs_[cacheInput]||source.SharesStorage(slots_[op.a].matrix)||source.SharesStorage(slots_[op.b].matrix))return false;
        for(auto id:op.arguments)if(id==inputs_[cacheInput]||source.SharesStorage(slots_[id].matrix))return false;
    }
    return true;
#else
    return false;
#endif
}
bool MetalCompiledGraph::ComputeResetKVPrefix(size_t part) const {
    if(part<1||part>2||inputs_.size()!=6||part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto& shape=inputShapes_[part];
    if(shape.size()!=4||std::any_of(shape.begin(),shape.end(),[](int64_t n){return n<=0;}))return false;
    const auto& group=outputStorageGroups_[part];
    auto member=[&](size_t id){return std::find(group.begin(),group.end(),id)!=group.end();};
    auto target=slots_[outputParts_[part]].matrix;std::unordered_set<size_t> visited;
    for(;;) {
        size_t writer=SIZE_MAX;
        auto writes=[&](size_t id){return id!=SIZE_MAX&&target.SharesStorage(slots_[id].matrix);};
        for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
            (writes(ops_[i].out)||writes(ops_[i].auxiliaryOutput))) {
            if(writer!=SIZE_MAX)return false;writer=i;
        }
        if(writer==SIZE_MAX||!visited.insert(writer).second)return false;
        const auto& op=ops_[writer];
        if(op.kind==Kind::Layout) {
            // Prefill returns the cache through a fresh reshape copy. Admit
            // only a complete, row-preserving copy, then prove its producer.
            const auto& input=slots_[op.a];const auto& output=slots_[op.out];
            if(op.a!=op.b||op.dimensions!=std::vector<size_t>{0}||op.auxiliaryOutput!=SIZE_MAX||
                input.rows!=size_t(shape[0])||output.rows!=input.rows||input.columns!=slots_[inputs_[part]].columns||output.columns!=input.columns)
                return false;
            target=input.matrix;continue;
        }
        if((op.kind!=Kind::KVUpdate&&op.kind!=Kind::KVChunk)||op.arguments.size()!=5||
            op.arguments[1]!=inputs_[part]||op.arguments[2]!=inputs_[4]||op.arguments[3]!=inputs_[5]||op.dimensions.size()!=4||
            op.dimensions[0]!=size_t(shape[1])||op.dimensions[1]!=size_t(shape[3])||op.dimensions[2]!=size_t(shape[2]))return false;
        break;
    }
    // Reject a writer aliasing the input cache outside the proved output group.
    const auto& source=slots_[inputs_[part]].matrix;
    for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]) {
        const auto& op=ops_[i];
        if(!member(op.out)&&source.SharesStorage(slots_[op.out].matrix))return false;
        if(op.auxiliaryOutput!=SIZE_MAX&&!member(op.auxiliaryOutput)&&source.SharesStorage(slots_[op.auxiliaryOutput].matrix))return false;
    }
    return true;
}
void MetalCompiledGraph::PrepareInputs(X::Value inputs,const std::unordered_set<size_t>& skip) {
    if(!ready_||inputs.runtime()!=Host()->runtime||!inputs.IsList()||inputs.Size()!=inputs_.size())
        throw std::runtime_error("Metal graph not ready or input count invalid");
    const char* profile=std::getenv("GARNET_METAL_PROFILE_OPS");
    if(verificationTokens_&&profile&&std::strcmp(profile,"1")==0)
        throw std::invalid_argument("Verification does not support serialized operation profiling");
    std::unordered_map<size_t,std::vector<float>> hostInputs;
    std::unordered_set<size_t> controls;
    if(verificationTokens_)for(size_t i=3;i<6;++i) {
        if(skip.count(i))throw std::invalid_argument("Verification controls must be validated host inputs");
        controls.insert(inputs_[i]);
    }
    for(const auto& op:ops_)if(op.kind==Kind::KVUpdate||op.kind==Kind::KVChunk)
        for(auto i:{2,3,4}){auto id=op.arguments[i];controls.insert(controlSources_.count(id)?controlSources_.at(id).first:id);}
    for(const auto& op:ops_)if(op.kind==Kind::Layout&&op.dimensions[0])controls.insert(op.b);
    for(size_t i=0;i<inputs_.size();++i) {
        if(skip.count(i))continue;
        X::Tensor tensor(inputs.Get(i));auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=inputShapes_[i].size()||
            !std::equal(inputShapes_[i].begin(),inputShapes_[i].end(),info.shape)||!info.data||!IsContiguousTensor(tensor))
            throw std::runtime_error("Metal runtime input shape/storage mismatch");
        auto lease=tensor.Acquire();auto data=static_cast<const float*>(info.data);
        if(controls.count(inputs_[i]))hostInputs[inputs_[i]]=std::vector<float>(data,data+slot.rows*slot.columns);
    }
    if(mtpProgram_) {
        const auto& flags=hostInputs.at(inputs_[6]);const auto& slots=hostInputs.at(inputs_[5]);
        const auto chunk=size_t(inputShapes_[0][1]);
        for(size_t base=0;base<flags.size();base+=chunk) {
            bool ended=false;
            for(size_t t=0;t<chunk;++t) {
                if(flags[base+t]==0)ended=true;
                else if(ended||slots[base+t]!=slots[base]+float(t))
                    throw std::invalid_argument("MTP requires contiguous active prefixes and increasing slots");
            }
        }
    }
    if(verificationTokens_) {
        // Validate linear layers too, before ANY layer is encoded by a session.
        // Partial blocks occupy a contiguous prefix of each physical row.
        const auto& active=hostInputs.at(inputs_[5]);const auto& slots=hostInputs.at(inputs_[4]);
        const auto& positions=hostInputs.at(inputs_[3]);
        for(size_t base=0;base<active.size();base+=verificationTokens_) {
            bool ended=false;
            for(size_t t=0;t<verificationTokens_;++t) {
                const auto i=base+t;
                if(!std::isfinite(active[i])||(active[i]!=0&&active[i]!=1)||
                   !std::isfinite(slots[i])||slots[i]<0||std::floor(slots[i])!=slots[i])
                    throw std::invalid_argument("Invalid verification active mask/slot");
                for(size_t axis=0;axis<3;++axis)if(!std::isfinite(positions[i*3+axis]))
                    throw std::invalid_argument("Verification positions must be finite");
                if(active[i]==0)ended=true;
                else if(ended||slots[i]>=4096||slots[i]!=slots[base]+float(t))
                    throw std::invalid_argument("Verification requires contiguous active prefixes and increasing slots");
            }
        }
    }
    for(const auto& op:ops_)if(op.kind==Kind::Layout&&op.dimensions[0]) {
        const auto& flags=hostInputs.at(op.b);
        for(float value:flags)if(!std::isfinite(value)||(value!=0&&value!=1))throw std::runtime_error("Invalid Metal chunk active flags");
    }
    for(const auto& entry:controlSources_)if(controls.count(entry.second.first)) {
        const auto& source=slots_[entry.second.first];const auto& values=hostInputs.at(entry.second.first);
        const auto& output=slots_[entry.first];std::vector<float> selected(output.rows*output.columns);
        for(size_t row=0;row<output.rows;++row)std::copy_n(values.data()+row*source.columns+entry.second.second*output.columns,
            output.columns,selected.data()+row*output.columns);
        hostInputs[entry.first]=std::move(selected);
    }
    for(auto& op:ops_)if(op.kind==Kind::KVUpdate||op.kind==Kind::KVChunk) {
        const auto& positions=hostInputs.at(op.arguments[4]);const auto& slots=hostInputs.at(op.arguments[2]);
        const auto& active=hostInputs.at(op.arguments[3]);
        for(size_t row=0;row<active.size();++row) {
            if(!std::isfinite(active[row])||(active[row]!=0&&active[row]!=1)||!std::isfinite(slots[row])||slots[row]<0||
                std::floor(slots[row])!=slots[row]||(active[row]!=0&&slots[row]>=op.dimensions[2]))
                throw std::runtime_error("Metal attention active row/slot invalid");
            for(size_t axis=0;axis<3;++axis)if(!std::isfinite(positions[row*3+axis]))
                throw std::runtime_error("Metal RoPE position must be finite");
        }
        if(op.kind==Kind::KVChunk) {
            // Direct slot lookup requires a contiguous active prefix in every
            // row. Overwrites, gaps and paused/resumed rows keep the scan.
            const size_t chunk=op.dimensions[3];op.contiguousSlots=true;
            for(size_t base=0;base<active.size();base+=chunk) {
                bool paused=false;
                for(size_t t=0;t<chunk;++t) {
                    if(active[base+t]==0)paused=true;
                    else if(paused||slots[base+t]!=slots[base]+float(t))op.contiguousSlots=false;
                }
            }
        }
    }
    for(size_t i=0;i<inputs_.size();++i){
        if(skip.count(i))continue;
        X::Tensor tensor(inputs.Get(i));auto lease=tensor.Acquire();auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        MaterializeDefault(inputs_[i]);
        gpu_->Update(slot.matrix,static_cast<const float*>(info.data),slot.rows*slot.columns);
    }
}
bool MetalCompiledGraph::SwapOutputStorage(size_t part,MetalMatrix& replacement) {
    if(part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto finished=slots_[outputParts_[part]].matrix;
    if(!finished.CompatibleLayout(replacement))return false;
    for(auto id:outputStorageGroups_[part])slots_[id].matrix=replacement;
    replacement=finished;return true;
}
void MetalCompiledGraph::EncodeGreedy(MetalContext& gpu,const MetalMatrix& prediction,const std::vector<size_t>* selectedRows) {
#if defined(GARNET_METAL_FUSED_HEAD_ARGMAX)||defined(GARNET_METAL_SCREENED_HEAD)
    Encode(gpu,true,&prediction,selectedRows);
#else
    Encode(gpu,true,nullptr,selectedRows);gpu.EncodeArgmax(slots_[output_].matrix,prediction);
#endif
}
void MetalCompiledGraph::Encode(MetalContext& gpu,bool packedOutput,const MetalMatrix* greedyOutput,
    const std::vector<size_t>* selectedTokenRows,VerificationBindings* verification) {
#ifndef GARNET_METAL_VERIFY_ATTENTION
    (void)verification;
#else
    if(verification&&!verificationTokens_)throw std::logic_error("Checkpoint bindings require a verification graph");
#endif
#if !defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)&&!defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)
    (void)selectedTokenRows;
#endif
    bool greedyEncoded=false;
    // Opt-in isolated-operation diagnostics deliberately split submissions.
    // They identify expensive kernels; their latency is not a serving metric.
    const char* profile=std::getenv("GARNET_METAL_PROFILE_OPS");
    const bool profileOps=profile&&std::strcmp(profile,"1")==0;
    static const char* names[]={"MatMul","BF16MatVec","Norm","Silu","Add","Bias","Mask","Conv","Delta","Slice","Prepare","GateNorm","Concat","NormRope","KVUpdate","KVChunk","AttentionChunk","Scores","AttentionContext","Vision","Layout","DeltaSplit","ConvChunk","DeltaChunk","VerificationAttention"};
    for(size_t i=0;i<ops_.size();++i) {
        if(!packedOutput&&!residentOps_[i])continue;
        const auto& op=ops_[i];
        auto a=slots_[op.a].matrix,b=slots_[op.b].matrix,out=slots_[op.out].matrix;
#ifdef GARNET_METAL_SCREENED_HEAD
        if(greedyOutput&&!profileOps&&i+1==ops_.size()&&op.out==output_&&op.kind==Kind::BF16MatVec&&
            slots_[op.b].bf16.Valid()&&gpu.EncodeScreenedArgmax(a,slots_[op.b].bf16,*greedyOutput)){
            greedyEncoded=true;continue;
        }
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
        if(greedyOutput&&!profileOps&&i+1==ops_.size()&&op.out==output_&&
            (op.kind==Kind::BF16MatVec||(op.kind==Kind::MatMul&&op.transpose))&&
            slots_[op.b].bf16.Valid()&&gpu.EncodeBF16Argmax(a,slots_[op.b].bf16,*greedyOutput)) {
            greedyEncoded=true;continue;
        }
#endif
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
        if(!profileOps&&op.kind==Kind::Add&&i+2<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]))) {
            const auto& layout=ops_[i+1];const auto& norm=ops_[i+2];
            const auto& packed=slots_[op.out];const auto& reshaped=slots_[layout.out];
            if(layout.kind==Kind::Layout&&layout.a==op.out&&layout.dimensions.size()==1&&!layout.dimensions[0]&&
                norm.kind==Kind::Norm&&norm.a==layout.out&&packed.rows<=2&&reshaped.rows>packed.rows&&
                reshaped.columns&&packed.columns%reshaped.columns==0&&
                packed.columns/reshaped.columns<=kMetalPrefillChunkLimit&&a.CompatibleLayout(b)&&a.CompatibleLayout(out)&&
                reshaped.matrix.CompatibleLayout(slots_[norm.out].matrix)) {
                gpu.EncodePackedPrefillNorm(a,b,slots_[norm.b].matrix,out,reshaped.matrix,slots_[norm.out].matrix,norm.epsilon,norm.offset);
                i+=2;continue;
            }
        }
#endif
#ifdef GARNET_METAL_RESIDUAL_NORM
        if(!profileOps&&op.kind==Kind::Add&&i+1<ops_.size()&&(packedOutput||residentOps_[i+1])) {
            const auto& next=ops_[i+1];
            if(next.kind==Kind::Norm&&next.a==op.out&&a.CompatibleLayout(b)&&a.CompatibleLayout(out)&&
                a.CompatibleLayout(slots_[next.out].matrix)) {
                gpu.EncodeResidualRMSNorm(a,b,slots_[next.b].matrix,out,slots_[next.out].matrix,next.epsilon,next.offset);
                ++i;continue;
            }
        }
#endif
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
        if(!profileOps&&slots_[op.a].rows>2&&i+3<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]&&residentOps_[i+3]))) {
            const auto& up=ops_[i+1];const auto& silu=ops_[i+2];const auto& down=ops_[i+3];
            auto projection=[&](const Op& value){return (value.kind==Kind::BF16MatVec||
                (value.kind==Kind::MatMul&&value.transpose))&&slots_[value.b].bf16.Valid();};
            if(projection(op)&&projection(up)&&projection(down)&&op.a==up.a&&op.out!=up.out&&
                silu.kind==Kind::Silu&&silu.a==op.out&&silu.b==up.out&&down.a==silu.out&&
                gpu.EncodeMlpPrefillGraph(a,slots_[op.b].bf16,slots_[up.b].bf16,
                    slots_[down.b].bf16,slots_[down.out].matrix)) {
                i+=3;continue;
            }
        }
#endif
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
        if(!profileOps&&slots_[op.a].rows>2&&i+2<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]))) {
            const auto& next=ops_[i+1];const auto& silu=ops_[i+2];
            auto projection=[&](const Op& v){return (v.kind==Kind::BF16MatVec||
                (v.kind==Kind::MatMul&&v.transpose))&&slots_[v.b].bf16.Valid();};
            if(projection(op)&&projection(next)&&op.a==next.a&&op.out!=next.out&&
                silu.kind==Kind::Silu&&silu.a==op.out&&silu.b==next.out&&
                gpu.EncodeJoinedGlobalMlp(a,slots_[op.b].bf16,slots_[next.b].bf16,
                    out,slots_[next.out].matrix,slots_[silu.out].matrix)) {
                i+=2;continue;
            }
        }
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
        if(!profileOps&&slots_[op.a].rows>2&&i+1<ops_.size()&&(packedOutput||residentOps_[i+1])) {
            const auto& next=ops_[i+1];
            auto projection=[&](const Op& value){return (value.kind==Kind::BF16MatVec||
                (value.kind==Kind::MatMul&&value.transpose))&&slots_[value.b].bf16.Valid();};
            if(projection(op)&&projection(next)&&op.a==next.a&&op.out!=next.out&&
                gpu.EncodeJoinedBF16Prefill(a,slots_[op.b].bf16,slots_[next.b].bf16,out,slots_[next.out].matrix)) {
                ++i;continue;
            }
        }
#endif
        if(op.kind==Kind::MatMul) {
#ifdef GARNET_METAL_MLX_AFFINE_INT4
            if(slots_[op.b].affineInt4.Valid()){
                if(!op.transpose)throw std::logic_error("Affine INT4 projection must transpose checkpoint rows");
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
                if(selectedTokenRows&&op.indexedDecodeProjection&&selectedTokenRows->size()<slots_[op.a].rows)
                    gpu.EncodeAffineInt4IndexedDecode(a,slots_[op.b].affineInt4,out,*selectedTokenRows);
                else
#endif
#ifdef GARNET_METAL_INDEXED_TOKEN_PROJECTIONS
                if(selectedTokenRows&&op.indexedTokenProjection&&selectedTokenRows->size()<slots_[op.a].rows)
                    gpu.EncodeAffineInt4IndexedMatMul(a,slots_[op.b].affineInt4,out,*selectedTokenRows);
                else
#endif
                    gpu.EncodeAffineInt4MatMul(a,slots_[op.b].affineInt4,out);
            }else
#endif
            if(slots_[op.b].bf16.Valid()){if(!op.transpose)throw std::logic_error("BF16 projection must transpose checkpoint rows");gpu.EncodeBF16MatVec(a,slots_[op.b].bf16,out);}
            else gpu.EncodeMatMul(a,b,out,op.transpose);
        }
        else if(op.kind==Kind::BF16MatVec)gpu.EncodeBF16MatVec(a,slots_[op.b].bf16,out);
        else if(op.kind==Kind::Layout)gpu.EncodeLayout(a,b,out,op.dimensions[0]);
        else if(op.kind==Kind::Norm)gpu.EncodeRMSNorm(a,b,out,op.epsilon,op.offset);
        else if(op.kind==Kind::Silu)gpu.EncodeSiluMul(a,b,out);
        else if(op.kind==Kind::Add)gpu.EncodeAdd(a,b,out);
        else if(op.kind==Kind::Bias)gpu.EncodeBias(a,b,out);
        else if(op.kind==Kind::Mask)gpu.EncodeMaskRows(a,b,out);
        else if(op.kind==Kind::Conv) {
            auto& args=op.arguments;gpu.EncodeCausalConv(slots_[args[0]].matrix,slots_[args[1]].matrix,
                slots_[args[2]].matrix,slots_[args[3]].matrix,out,op.dimensions[0]);
        }else if(op.kind==Kind::ConvChunk) {
            auto& args=op.arguments;
#ifdef GARNET_METAL_VERIFY_ATTENTION
            if(verification&&verification->conv==i) {
                gpu.EncodeLayout(slots_[args[0]].matrix,slots_[args[0]].matrix,verification->qkv,0);
                gpu.EncodeVerificationConvCheckpoint(slots_[args[0]].matrix,slots_[args[1]].matrix,
                    slots_[args[2]].matrix,slots_[args[3]].matrix,out,slots_[op.auxiliaryOutput].matrix,
                    verification->convS1,op.dimensions[0],op.dimensions[1]);
            }else
#endif
            gpu.EncodeCausalConvChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,
                slots_[args[2]].matrix,slots_[args[3]].matrix,out,slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1]);
        }else if(op.kind==Kind::DeltaChunk) {
            auto& args=op.arguments;
#ifdef GARNET_METAL_VERIFY_ATTENTION
            if(verification&&verification->delta==i)
                gpu.EncodeVerificationDeltaCheckpoint(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                    out,slots_[op.auxiliaryOutput].matrix,verification->deltaS1,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
            else
#endif
            gpu.EncodeGatedDeltaChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                out,slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }else if(op.kind==Kind::Delta) {
            auto& args=op.arguments;gpu.EncodeGatedDelta(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,slots_[args[5]].matrix,slots_[args[6]].matrix,out,
                op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::DeltaSplit) {
            auto& args=op.arguments;gpu.EncodeGatedDeltaSplit(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,slots_[args[5]].matrix,slots_[args[6]].matrix,out,
                slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::Prepare) {
            auto& args=op.arguments;gpu.EncodeLinearPrepare(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
#ifdef GARNET_METAL_VERIFY_ATTENTION
            if(verification&&verification->prepare==i)gpu.EncodeLayout(out,out,verification->prepared,0);
#endif
        }else if(op.kind==Kind::GateNorm) {
            auto& args=op.arguments;gpu.EncodeGatedNorm(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                out,op.dimensions[0],op.dimensions[1],op.epsilon,op.dimensions.size()>2?op.dimensions[2]:0);
        }else if(op.kind==Kind::Concat) {
            auto& args=op.arguments;gpu.EncodeConcat(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,out,op.transpose);
        }else if(op.kind==Kind::NormRope) {
            auto& args=op.arguments;gpu.EncodeNormRope(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.epsilon,op.offset,op.dimensions[3]!=0);
        }else if(op.kind==Kind::KVChunk) {
            auto& args=op.arguments;gpu.EncodeKVChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.contiguousSlots);
        }else if(op.kind==Kind::AttentionChunk) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu.EncodeAttentionChunk(args,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4],op.dimensions[5]!=0);
#ifdef GARNET_METAL_VERIFY_ATTENTION
        }else if(op.kind==Kind::VerificationAttention) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu.EncodeVerificationAttention(args,slots_[op.auxiliaryOutput].matrix,out,
                op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4],op.dimensions[5]!=0);
#endif
        }else if(op.kind==Kind::KVUpdate) {
            auto& args=op.arguments;gpu.EncodeKVUpdate(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::Scores) {
            auto& args=op.arguments;gpu.EncodeAttentionScores(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }else if(op.kind==Kind::Vision) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu.EncodeVision(op.dimensions[0],args,out,op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
        }else if(op.kind==Kind::AttentionContext) {
            auto& args=op.arguments;gpu.EncodeAttentionContext(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]!=0);
        }else {
            gpu.EncodeSliceGroups(a,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }
        if(profileOps) {
            gpu.SubmitAndWait();
            std::fprintf(stderr,"metal_op_profile {\"kind\":\"%s\",\"a_rows\":%zu,\"a_cols\":%zu,\"out_rows\":%zu,\"out_cols\":%zu,\"bf16\":%s,\"exact_half_prefill\":%s,\"gpu_ms\":%.9g,\"serialized_diagnostic\":true}\n",
                names[size_t(op.kind)],slots_[op.a].rows,slots_[op.a].columns,
                slots_[op.out].rows,slots_[op.out].columns,
                slots_[op.b].bf16.Valid()?"true":"false",
                slots_[op.a].rows>2&&slots_[op.b].bf16.ExactHalfReady()?"true":"false",gpu.LastGPUMs());
            gpu.Begin();
        }
    }
    if(greedyOutput&&!greedyEncoded)gpu.EncodeArgmax(slots_[output_].matrix,*greedyOutput);
}
X::Value MetalCompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::unique_lock<std::mutex> scratchLock;
    if(prefillScratch_)scratchLock=std::unique_lock<std::mutex>(prefillScratch_->mutex);
#endif
#ifdef GARNET_METAL_RETIRE_INPLACE_DEFAULTS
    // Standalone packed execution owns its outputs rather than receiving the
    // session's private KV bindings. Restore every retired default first.
    for(size_t part=
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
        0
#else
        1
#endif
        ;part<outputStorageGroups_.size();++part) {
        const auto& group=outputStorageGroups_[part];
        if(group.empty()||slots_[group.front()].matrix.Valid()||
           std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),group.front())==lazyDefaultSlots_.end())continue;
        MaterializeDefault(group.front());
        for(auto id:group)slots_[id].matrix=slots_[group.front()].matrix;
    }
    for(auto id:lazyDefaultSlots_)MaterializeDefault(id);
#endif
    visionParallelDispatches_=0;
    visionPrefetchDispatches_=0;visionVectorDispatches_=0;mlpGraphDispatches_=0;
    visionGlobalAlphaDispatches_=0;
    auto start=std::chrono::steady_clock::now();
    PrepareInputs(inputs);
    MaterializeDefault(output_);
    gpu_->Begin();
    try {
        Encode(*gpu_);
        gpu_->SubmitAndWait();
        visionGlobalAlphaDispatches_=gpu_->LastVisionGlobalAlphaDispatches();
        visionParallelDispatches_=gpu_->LastVisionParallelDispatches();
        visionPrefetchDispatches_=gpu_->LastVisionPrefetchDispatches();
        visionVectorDispatches_=gpu_->LastVisionVectorDispatches();
        mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();
    }catch(...){gpu_->Abort();throw;}
    auto result=gpu_->Download(slots_[output_].matrix);auto& output=slots_[output_];
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,output.shape,result.data(),result.size()*4);
}
}

namespace Garnet {
std::vector<MetalResourceReference> MetalCompiledGraph::MemoryReferences() {
    std::lock_guard<std::mutex> lock(mutex_);if(!ready_)throw std::logic_error("Inventory requires a ready graph");
    std::vector<MetalResourceReference> result;std::vector<bool> active(slots_.size(),false),input(slots_.size(),false),output(slots_.size(),false);
    auto mark=[&](size_t i){if(i<active.size())active[i]=true;};
    std::vector<bool> anyWriter(slots_.size(),false),residentWriter(slots_.size(),false);
    for(size_t i=0;i<ops_.size();++i){bool resident=residentOps_.size()!=ops_.size()||residentOps_[i];auto write=[&](size_t slot){if(slot<slots_.size()){anyWriter[slot]=true;if(resident)residentWriter[slot]=true;}};write(ops_[i].out);write(ops_[i].auxiliaryOutput);}

    for(size_t i=0;i<ops_.size();++i)if(residentOps_.size()!=ops_.size()||residentOps_[i]) {
        const auto& op=ops_[i];mark(op.a);mark(op.b);mark(op.out);mark(op.auxiliaryOutput);for(auto j:op.arguments)mark(j);
    }
    for(auto i:inputs_)if(i<input.size())input[i]=true;for(auto i:outputParts_)if(i<output.size())output[i]=true;
    std::vector<std::string> names(slots_.size());std::vector<std::pair<std::string,size_t>> sorted(weights_.begin(),weights_.end());std::sort(sorted.begin(),sorted.end());
    for(const auto& entry:sorted)if(entry.second<names.size()){if(!names[entry.second].empty())names[entry.second]+=";";names[entry.second]+=entry.first;}
    auto append=[&](const Slot& slot,size_t i,const std::string& role,const std::string& name,bool weight,bool in,bool out,bool resident,bool legacy){
        auto emit=[&](const MetalBufferInfo& info){if(!info.identity)return;MetalResourceReference r;r.buffer=info;r.slot=i;r.role=role;r.name=name;r.shape=slot.shape;r.weight=weight;r.input=in;r.outputPart=out;r.residentOperand=resident;r.anyWriter=i<anyWriter.size()&&anyWriter[i];r.residentWriter=i<residentWriter.size()&&residentWriter[i];r.legacyOutput=legacy;result.push_back(std::move(r));};
        emit(MetalContext::MatrixBufferInfo(slot.matrix));for(const auto& info:MetalContext::WeightBufferInfo(slot.bf16))emit(info);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        for(const auto& info:MetalContext::AffineInt4BufferInfo(slot.affineInt4))emit(info);
#endif
    };
    for(size_t i=0;i<slots_.size();++i)append(slots_[i],i,"slot",names[i],!names[i].empty(),input[i],output[i],active[i],i==output_);
    auto held=[&](const auto& records,const char* role){std::vector<std::string> keys;for(const auto& entry:records)keys.push_back(entry.first);std::sort(keys.begin(),keys.end());for(const auto& key:keys)append(records.at(key).slot,SIZE_MAX,role,key,true,false,false,false,false);};
    held(sharedWeights_,"shared_weight_record");held(weightRecords_,"weight_record");
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    if(prefillScratch_) {
        std::lock_guard<std::mutex> scratchLock(prefillScratch_->mutex);
        for(const auto& entry:prefillScratch_->buffers)for(const auto& matrix:entry.second) {
            MetalResourceReference reference;reference.buffer=MetalContext::MatrixBufferInfo(matrix);
            reference.slot=SIZE_MAX;reference.role="shared_prefill_scratch_pool";
            reference.shape={int64_t(entry.first.first),int64_t(entry.first.second)};
            result.push_back(std::move(reference));
        }
    }
#endif
    return result;
}
}

#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
namespace Garnet {
#ifdef GARNET_METAL_PREFILL_ALIAS_TEST_FAULTS
bool MetalCompiledGraph::FailNextPrefillAliasCommit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!ready_||prefillScratch_)return false;
    prefillAliasFailNextCommit_=true;return true;
}
#endif
void MetalCompiledGraph::SharePrefillScratch(const std::shared_ptr<MetalPrefillScratch>& owner) {
    if(!owner||!ready_||outputParts_.size()!=3||residentOps_.size()!=ops_.size())
        throw std::invalid_argument("Shared prefill scratch requires a ready resident graph");
    if(prefillScratch_==owner)return;
    if(prefillScratch_)throw std::logic_error("Prefill graph already belongs to another scratch owner");
    std::lock_guard<std::mutex> scratchLock(owner->mutex);
    std::vector<MetalBufferInfo> infos;
    std::unordered_map<uintptr_t,size_t> counts;
    std::unordered_set<uintptr_t> protectedStorage;
    auto protect=[&](const Slot& slot) {
        auto info=MetalContext::MatrixBufferInfo(slot.matrix);
        if(info.identity)protectedStorage.insert(info.identity);
        for(const auto& buffer:MetalContext::WeightBufferInfo(slot.bf16))protectedStorage.insert(buffer.identity);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        for(const auto& buffer:MetalContext::AffineInt4BufferInfo(slot.affineInt4))protectedStorage.insert(buffer.identity);
#endif
    };
    for(const auto& slot:slots_) {
        auto info=MetalContext::MatrixBufferInfo(slot.matrix);infos.push_back(info);
        if(info.identity)++counts[info.identity];
    }
    for(auto id:inputs_)protect(slots_[id]);
    protect(slots_[output_]);
    for(auto id:outputParts_)protect(slots_[id]);
    for(const auto& entry:weights_)protect(slots_[entry.second]);
    for(const auto& entry:weightRecords_)protect(entry.second.slot);
    for(const auto& entry:sharedWeights_)protect(entry.second.slot);
    // Prove initialization for both resident and standalone packed execution.
    // A prior read (including a same-op read/write) rejects the buffer.
    std::vector<size_t> firstRead(slots_.size(),SIZE_MAX),firstWrite(slots_.size(),SIZE_MAX);
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
    std::vector<size_t> lastUse(slots_.size(),0);
#endif
    std::vector<bool> residentWriter(slots_.size(),false),fullyInitialized(slots_.size(),false);
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
    std::vector<bool> residentFirstWrite(slots_.size(),false);
#endif
    for(size_t i=0;i<ops_.size();++i) {
        const auto& op=ops_[i];
        const bool fullWrite=op.kind==Kind::MatMul||op.kind==Kind::BF16MatVec||op.kind==Kind::Norm||
            op.kind==Kind::Silu||op.kind==Kind::Add||op.kind==Kind::Bias||op.kind==Kind::Slice||
            op.kind==Kind::Concat||op.kind==Kind::NormRope||op.kind==Kind::Layout||
            op.kind==Kind::Mask||op.kind==Kind::Prepare||op.kind==Kind::ConvChunk||op.kind==Kind::DeltaChunk||
            op.kind==Kind::VerificationAttention||
            (op.kind==Kind::GateNorm&&op.dimensions.size()>=2&&
             (op.dimensions.size()==2||op.dimensions[2]==0)&&op.out<slots_.size()&&
             slots_[op.out].columns==op.dimensions[0]*op.dimensions[1]);
        // Chunk kernels write every logical output element, including zeros
        // for inactive tokens. GateNorm may instead write a slice, so only its
        // zero-offset, whole-width form can initialize shared scratch.
        auto read=[&](size_t id){if(id<slots_.size()){
            firstRead[id]=std::min(firstRead[id],i);
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
            lastUse[id]=std::max(lastUse[id],i);
#endif
        }};
        auto write=[&](size_t id){if(id<slots_.size()){
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
            lastUse[id]=std::max(lastUse[id],i);
#endif
            if(firstWrite[id]==SIZE_MAX) {
                fullyInitialized[id]=fullWrite;
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
                residentFirstWrite[id]=residentOps_[i];
#endif
            }
            firstWrite[id]=std::min(firstWrite[id],i);residentWriter[id]=residentWriter[id]||residentOps_[i];}};
        read(op.a);read(op.b);for(auto id:op.arguments)read(id);
        write(op.out);write(op.auxiliaryOutput);
    }
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
    std::vector<ScratchLifetime> lifetimes;
#else
    std::map<std::pair<size_t,size_t>,size_t> next;
#endif
    size_t reused=0;
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
    std::vector<ScratchAliasMember> members;
    for(size_t id=0;id<slots_.size();++id) {
        const auto& slot=slots_[id];const auto& info=infos[id];
        if(!info.identity)continue;
        const bool allowed=!protectedStorage.count(info.identity)&&
            gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns);
        members.push_back({id,info.identity,slot.rows,slot.columns,firstRead[id],firstWrite[id],lastUse[id],
                           allowed,fullyInitialized[id],residentFirstWrite[id]});
    }
    const auto groups=PlanScratchAliasGroups(members);
    std::unordered_map<size_t,const ScratchAliasGroup*> byRepresentative;
    for(const auto& group:groups) {
        lifetimes.push_back(group.lifetime);
        byRepresentative.emplace(group.lifetime.slot,&group);
    }
    size_t aliasGroups=0,aliasSlots=0;
    // Preparation is transactional: allocating vectors/maps or detecting an
    // invalid view must leave BOTH graph slots and the existing pool intact.
    auto stagedBuffers=owner->buffers;
    std::vector<std::pair<size_t,MetalMatrix>> stagedBindings;
    for(const auto& assignment:PlanScratchLifetimes(std::move(lifetimes))) {
        const auto& group=*byRepresentative.at(assignment.slot);
        auto& slot=slots_[assignment.slot];const auto& info=infos[assignment.slot];
        const auto key=std::make_pair(slot.rows,slot.columns);const auto lane=assignment.lane;
        auto& buffers=stagedBuffers[key];
        if(lane>buffers.size())throw std::logic_error("Scratch alias lane allocation gap");
        if(lane==buffers.size())buffers.push_back(slot.matrix);
        else {
            if(reused>SIZE_MAX-info.allocatedBytes)throw std::overflow_error("Scratch alias accounting overflow");
            reused+=info.allocatedBytes;
        }
        // Validate every physical view before staging the shared Impl. All
        // aliases must share initialization/pending ownership after commit.
        for(auto id:group.slots)if(!slots_[id].matrix.CompatibleLayout(buffers[lane]))
            throw std::logic_error("Scratch alias physical layout mismatch");
        for(auto id:group.slots)stagedBindings.emplace_back(id,buffers[lane]);
        if(group.slots.size()>1){++aliasGroups;aliasSlots+=group.slots.size();}
    }
#ifdef GARNET_METAL_PREFILL_ALIAS_TEST_FAULTS
    if(prefillAliasFailNextCommit_) {
        prefillAliasFailNextCommit_=false;
        throw std::runtime_error("Injected prefill alias precommit failure");
    }
#endif
    // No allocation or layout check remains after the first graph mutation.
    // The preparing caller holds the graph lock and this pool's lock; existing
    // execution paths retain the pool lease through GPU completion.
    static_assert(std::is_nothrow_copy_assignable<MetalMatrix>::value,"Scratch binding commit must not throw");
    static_assert(noexcept(owner->buffers.swap(stagedBuffers)),"Scratch pool commit must not throw");
    for(const auto& binding:stagedBindings)slots_[binding.first].matrix=binding.second;
    owner->buffers.swap(stagedBuffers);
    sharedPrefillAliasGroups_=aliasGroups;sharedPrefillAliasSlots_=aliasSlots;
#else
    for(size_t id=0;id<slots_.size();++id) {
        auto& slot=slots_[id];const auto& info=infos[id];
        if(!info.identity||counts[info.identity]!=1||protectedStorage.count(info.identity)||
           !residentWriter[id]||!fullyInitialized[id]||firstWrite[id]==SIZE_MAX||firstRead[id]<=firstWrite[id]||
           !gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns))continue;
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
        lifetimes.push_back({id,slot.rows,slot.columns,firstWrite[id],lastUse[id]});
    }
    for(const auto& assignment:PlanScratchLifetimes(std::move(lifetimes))) {
        const auto id=assignment.slot;auto& slot=slots_[id];const auto& info=infos[id];
        const auto key=std::make_pair(slot.rows,slot.columns);const auto lane=assignment.lane;
#else
        const auto key=std::make_pair(slot.rows,slot.columns);const auto lane=next[key]++;
#endif
        auto& buffers=owner->buffers[key];
        if(lane==buffers.size())buffers.push_back(slot.matrix);
        else {
            if(!slot.matrix.CompatibleLayout(buffers[lane]))throw std::logic_error("Shared prefill scratch layout mismatch");
            if(reused>SIZE_MAX-info.allocatedBytes)throw std::overflow_error("Shared prefill scratch accounting overflow");
            reused+=info.allocatedBytes;slot.matrix=buffers[lane];
        }
    }
#endif
    prefillScratch_=owner;sharedPrefillScratchBytes_=reused;
}
}
#endif

namespace Garnet {
void MetalCompiledGraph::RetireResidentDefaults(bool preserveHiddenInput) {
#ifndef GARNET_METAL_RETIRE_PREFILL_INPUTS
    (void)preserveHiddenInput;
#endif
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(!lazyDefaultPlanReady_) {
        if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;
        std::vector<MetalBufferInfo> infos;std::unordered_map<uintptr_t,size_t> counts;std::unordered_set<uintptr_t> protectedWeights;
        auto protect=[&](const Slot& slot){
            for(const auto& b:MetalContext::WeightBufferInfo(slot.bf16))protectedWeights.insert(b.identity);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
            for(const auto& b:MetalContext::AffineInt4BufferInfo(slot.affineInt4))protectedWeights.insert(b.identity);
#endif
        };
        for(const auto& slot:slots_){auto info=MetalContext::MatrixBufferInfo(slot.matrix);infos.push_back(info);if(info.identity)++counts[info.identity];protect(slot);}
        for(const auto& record:weightRecords_){auto info=MetalContext::MatrixBufferInfo(record.second.slot.matrix);if(info.identity)protectedWeights.insert(info.identity);protect(record.second.slot);}
        auto eligible=[&](size_t id){const auto& slot=slots_[id];const auto& info=infos[id];return info.identity&&counts[info.identity]==1&&!protectedWeights.count(info.identity)&&gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns);};
        std::vector<size_t> plan;
#ifdef GARNET_METAL_RETIRE_PREFILL_INPUTS
        if(inputs_.size()==6&&inputShapes_[0].size()==3) {
            const auto batch=inputShapes_[0][0],chunk=inputShapes_[0][1];const auto id=inputs_[0];
            bool safe=batch>0&&chunk>1&&inputShapes_[3]==std::vector<int64_t>{batch,chunk,3}&&
                inputShapes_[4]==std::vector<int64_t>{batch,chunk}&&inputShapes_[5]==inputShapes_[4]&&
                id!=output_&&std::find(outputParts_.begin(),outputParts_.end(),id)==outputParts_.end()&&eligible(id);
            for(const auto& op:ops_)if(op.out==id||op.auxiliaryOutput==id)safe=false;
            if(safe)plan.push_back(id);
        }
#endif
        size_t cacheStart=1;
#ifdef GARNET_METAL_MTP_KV_INPLACE
        if(mtpProgram_&&inputs_.size()==7)cacheStart=2;
#endif
        for(size_t part=cacheStart;part<cacheStart+2;++part){auto id=inputs_[part];if(inputShapes_[part].size()<3||std::find(outputParts_.begin(),outputParts_.end(),id)!=outputParts_.end())continue;if(eligible(id))plan.push_back(id);}
        if(!ops_.empty()&&ops_.back().kind==Kind::Concat&&ops_.back().out==output_&&!residentOps_.back()&&std::find(inputs_.begin(),inputs_.end(),output_)==inputs_.end()&&std::find(outputParts_.begin(),outputParts_.end(),output_)==outputParts_.end()&&eligible(output_))plan.push_back(output_);
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
        // Only fresh complete concat outputs, never arbitrary "unused" slots.
        // Resolve ownership by storage identity so resident aliases protect the
        // same allocation even when the writer uses another slot number.
        std::unordered_set<uintptr_t> residentStorage;
        auto mark=[&](size_t id){if(id<infos.size()&&infos[id].identity)residentStorage.insert(infos[id].identity);};
        for(size_t i=0;i<ops_.size();++i)if(residentOps_[i]) {
            const auto& op=ops_[i];mark(op.a);mark(op.b);mark(op.out);mark(op.auxiliaryOutput);
            for(auto id:op.arguments)mark(id);
        }
        for(size_t i=0;i<ops_.size();++i) {
            const auto& op=ops_[i];const auto id=op.out;
            if(residentOps_[i]||op.kind!=Kind::Concat||id==output_||!eligible(id)||
               residentStorage.count(infos[id].identity)||
               std::find(inputs_.begin(),inputs_.end(),id)!=inputs_.end()||
               std::find(outputParts_.begin(),outputParts_.end(),id)!=outputParts_.end()||
               std::find(plan.begin(),plan.end(),id)!=plan.end())continue;
            // Build lowers concats into fresh output allocations. Reconfirm
            // that this is the only writer and no prior op consumes it.
            bool safe=true;
            for(size_t j=0;j<ops_.size();++j) {
                const auto& other=ops_[j];
                if(j!=i&&(other.out==id||other.auxiliaryOutput==id))safe=false;
                if(j<=i&&(other.a==id||other.b==id||std::find(other.arguments.begin(),other.arguments.end(),id)!=other.arguments.end()))safe=false;
            }
            if(safe){plan.push_back(id);++nonresidentPackSlots_;}
        }
#endif
#ifdef GARNET_METAL_RETIRE_INPLACE_DEFAULTS
        // In-place session execution binds every proven output alias to its
        // private cache. Retire only complete, exclusively owned alias groups;
        // hidden outputs, inputs and checkpoint owners must remain protected.
        for(size_t part=0;part<=2;++part)if((part>0&&CanUpdateKVInPlace(part))
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
            ||(part==0&&CanPoolResidentHidden())
#endif
            ) {
            const auto& group=outputStorageGroups_[part];
            const auto identity=infos[group.front()].identity;
            std::unordered_set<size_t> unique(group.begin(),group.end());
            bool safe=identity&&unique.size()==group.size()&&counts[identity]==group.size()&&!protectedWeights.count(identity);
            for(auto id:group) {
                const auto& slot=slots_[id];
                safe=safe&&infos[id].identity==identity&&id!=output_&&(part==0||id!=outputParts_[0])&&
                    std::find(inputs_.begin(),inputs_.end(),id)==inputs_.end()&&
                    gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns);
            }
            if(safe)plan.insert(plan.end(),group.begin(),group.end());
        }
#endif
        lazyDefaultSlots_=std::move(plan);lazyDefaultPlanReady_=true;
    }
    for(auto id:lazyDefaultSlots_)if(slots_[id].matrix.Valid()) {
#ifdef GARNET_METAL_RETIRE_PREFILL_INPUTS
        // The first layer just uploaded its host hidden input. Subsequent
        // layers skip that upload and borrow their predecessor's hidden output.
        if(preserveHiddenInput&&id==inputs_[0])continue;
#endif
        auto info=MetalContext::MatrixBufferInfo(slots_[id].matrix);lazyRetiredBytes_+=info.allocatedBytes;slots_[id].matrix=MetalMatrix();
    }
#endif
}
void MetalCompiledGraph::MaterializeDefault(size_t id) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(slots_[id].matrix.Valid())return;
    if(std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),id)==lazyDefaultSlots_.end())throw std::logic_error("Missing non-deferred graph storage");
    auto matrix=gpu_->Allocate(slots_[id].rows,slots_[id].columns);auto info=MetalContext::MatrixBufferInfo(matrix);slots_[id].matrix=std::move(matrix);lazyRehydratedBytes_+=info.allocatedBytes;
#else
    (void)id;
#endif
}
size_t MetalCompiledGraph::RetiredPrefillHiddenInputSlots() const {
#if defined(GARNET_METAL_RETIRE_PREFILL_INPUTS)&&defined(GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
    std::lock_guard<std::mutex> lock(mutex_);
    return !inputs_.empty()&&std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),inputs_[0])!=lazyDefaultSlots_.end()?1:0;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyRetiredBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyRetiredBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyRehydratedBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyRehydratedBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyPlannedSlots() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyDefaultSlots_.size();
#else
    return 0;
#endif
}
}
