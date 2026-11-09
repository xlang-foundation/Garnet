"""Install a default-off native down-projection specialization in an isolated source."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=R.parent;S=R/'source1';P=M/'mistral-tensor-view-prefill-screen1'
assert json.loads((P/'timing-audit3.json').read_text())['passed']
changes=[]
def edit(name,old,new,count=1):
 p=S/name;s=p.read_text();assert s.count(old)==count,(name,old[:80],s.count(old));p.write_text(s.replace(old,new));changes.append(name)
shader=(P/'shared3.metal').read_text().replace('shared_tensor_m256','packed_down_tensor_m256')
header='// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kPackedDownTensorShader=R"METAL(\n'+shader+')METAL";\n}\n'
(S/'src/backends/metal/packed_down_tensor_shader.h').write_text(header);changes.append('src/backends/metal/packed_down_tensor_shader.h')
option='''option(GARNET_METAL_MISTRAL_DOWN_TENSOR512 "Opt-in Apple10 dense512 Mistral down projection with tensor operands" OFF)
if(GARNET_METAL_MISTRAL_DOWN_TENSOR512)
    if(NOT GARNET_METAL_MISTRAL_PACKED_PREFILL OR NOT GARNET_METAL_AFFINE_FP16_COEFFICIENTS OR NOT GARNET_METAL_AFFINE_PROFILE_INDEX32)
        message(FATAL_ERROR "Tensor down512 requires packed prefill, FP16 coefficients and bounded affine indexes")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_MISTRAL_DOWN_TENSOR512=1)
endif()

'''
edit('cmake/GarnetMetal.cmake','option(GARNET_METAL_AFFINE_WEIGHT_RESIDENCY',option+'option(GARNET_METAL_AFFINE_WEIGHT_RESIDENCY')
edit('src/backends/metal/metal_context.h','    void SetPackedAffineEnabled(bool enabled);','    void SetPackedAffineEnabled(bool enabled);\n    void SetPackedTensorDownEnabled(bool enabled);\n    bool PackedTensorDownEnabled() const;\n    size_t LastPackedTensorDownDispatches() const;')
edit('src/backends/metal/metal_context.mm','#include "packed_affine_shader.h"','#include "packed_affine_shader.h"\n#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512\n#include "packed_down_tensor_shader.h"\n#endif')
edit('src/backends/metal/metal_context.mm','    struct PackedPlan {std::array<id<MTLComputePipelineState>,2> pipelines{};','    struct PackedPlan {std::array<id<MTLComputePipelineState>,2> pipelines{};\n        id<MTLComputePipelineState> tensorDown=nil;')
edit('src/backends/metal/metal_context.mm','    bool packedEnabled=true;','''    bool packedEnabled=true;
#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
    bool packedTensorDownEnabled=true;
#else
    bool packedTensorDownEnabled=false;
#endif
    size_t packedTensorDownDispatches=0;
    bool UsePackedTensorDown(const PackedPlan& plan,size_t rows,const std::vector<uint32_t>& indices) const {
        if(!packedTensorDownEnabled||!plan.tensorDown||rows!=512||indices.size()!=512)return false;
        for(size_t i=0;i<512;++i)if(indices[i]!=i)return false;
        return true;
    }''')
marker='bool MetalContext::PreparePackedAffineProfile('
methods='''void MetalContext::SetPackedTensorDownEnabled(bool enabled) {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Select tensor down between drained command batches");
#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
    if(enabled&&![impl_->device supportsFamily:MTLGPUFamilyApple10])throw std::logic_error("Tensor down requires Apple10");
#else
    if(enabled)throw std::logic_error("Tensor down requires its compiled native extension");
#endif
    impl_->packedTensorDownEnabled=enabled;
}
bool MetalContext::PackedTensorDownEnabled() const {
#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
    return impl_->packedTensorDownEnabled&&[impl_->device supportsFamily:MTLGPUFamilyApple10];
#else
    return false;
#endif
}
size_t MetalContext::LastPackedTensorDownDispatches() const {return impl_->packedTensorDownDispatches;}
'''
edit('src/backends/metal/metal_context.mm',marker,methods+marker)
edit('src/backends/metal/metal_context.mm','            source=[source stringByAppendingString:[NSString stringWithUTF8String:kPackedAffineShader]];','''            source=[source stringByAppendingString:[NSString stringWithUTF8String:kPackedAffineShader]];
#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
            if([impl_->device supportsFamily:MTLGPUFamilyApple10])source=[source stringByAppendingString:[NSString stringWithUTF8String:kPackedDownTensorShader]];
#endif''')
marker='''#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL
        for(size_t i=0;i<2;++i)'''
# Pipeline preparation is inserted immediately before paired-plan preparation.
p=S/'src/backends/metal/metal_context.mm';text=p.read_text();start=text.index('bool MetalContext::PreparePackedAffineProfile(');at=text.index('#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL',text.index('plan.pipelines[i]=pipeline;',start))
insert='''#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
        if(a.columns==14336&&o.columns==4096&&[impl_->device supportsFamily:MTLGPUFamilyApple10]) {
            auto function=[impl_->packedLibrary newFunctionWithName:@"packed_down_tensor_m256" constantValues:constants error:&error];
            if(!function)throw std::runtime_error("Cannot specialize tensor down profile");
            auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<256||pipeline.staticThreadgroupMemoryLength!=17408)
                throw std::runtime_error("Tensor down pipeline resource mismatch");
            plan.tensorDown=pipeline;
        }
#endif
'''
p.write_text(text[:at]+insert+text[at:]);changes.append('src/backends/metal/metal_context.mm')
edit('src/backends/metal/metal_context.mm','  auto gatePipeline=gp->second.fused[gateTile==256],downPipeline=dp->second.pipelines[downTile==256];','''  auto gatePipeline=gp->second.fused[gateTile==256],downPipeline=dp->second.pipelines[downTile==256];
  const bool tensorDown=downTile==256&&impl_->UsePackedTensorDown(dp->second,a.rows,indices);
  if(tensorDown)downPipeline=dp->second.tensorDown;''')
edit('src/backends/metal/metal_context.mm','  ++impl_->fusedMlpBlocks;impl_->packedDispatches+=3;','  if(tensorDown)++impl_->packedTensorDownDispatches;\n  ++impl_->fusedMlpBlocks;impl_->packedDispatches+=3;')
edit('src/backends/metal/metal_context.mm','    auto pipeline=found->second.pipelines[tile==256?1:0];','''    auto pipeline=found->second.pipelines[tile==256?1:0];
    const bool tensorDown=tile==256&&impl_->UsePackedTensorDown(found->second,a.rows,indices);
    if(tensorDown)pipeline=found->second.tensorDown;''')
edit('src/backends/metal/metal_context.mm','    ++impl_->packedDispatches;++impl_->affineProfileDispatches;if(reportIndexed)++impl_->indexedAffineDispatches;','    if(tensorDown)++impl_->packedTensorDownDispatches;\n    ++impl_->packedDispatches;++impl_->affineProfileDispatches;if(reportIndexed)++impl_->indexedAffineDispatches;')
edit('src/backends/metal/metal_context.mm','    impl_->packedDispatches=0;impl_->diagnosticBulkDispatches=0;','    impl_->packedDispatches=0;impl_->packedTensorDownDispatches=0;impl_->diagnosticBulkDispatches=0;',2)
edit('src/backends/metal/decoder_session.h','    size_t PackedAffinePlanCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->PackedAffinePlanCount():0;}','''    size_t PackedAffinePlanCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->PackedAffinePlanCount():0;}
    size_t LastPackedTensorDownDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastPackedTensorDownDispatches():0;}
    bool PrefillSetTensorDown(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_||poisoned_)throw std::logic_error("Ready session required for tensor down selection");
        gpu_->SetPackedTensorDownEnabled(enabled);return true;
    }''')
edit('src/backends/metal/decoder_session.h','        APISET().AddFunc<1>("prefill_set_packed_affine",&MetalDecoderSession::PrefillSetPackedAffine);','''        APISET().AddFunc<1>("prefill_set_packed_affine",&MetalDecoderSession::PrefillSetPackedAffine);
        APISET().AddFunc<1>("prefill_set_tensor_down",&MetalDecoderSession::PrefillSetTensorDown);
        APISET().AddProp("packed_tensor_down_dispatches",&MetalDecoderSession::LastPackedTensorDownDispatches);''')
edit('src/entry/metal_main.cpp','    info+="; mistral_packed_prefill_optional; packed_prefill_min_rows=128; packed_prefill_scratch_bytes=14680064";','''    info+="; mistral_packed_prefill_optional; packed_prefill_min_rows=128; packed_prefill_scratch_bytes=14680064";
#ifdef GARNET_METAL_MISTRAL_DOWN_TENSOR512
    info+="; native_mistral_down_tensor512_optional_v1";
#endif''')
edit('test2026/metal/packed_affine_test.cpp','using namespace Garnet;using JSON=nlohmann::json;','using namespace Garnet;using JSON=nlohmann::json;\nsize_t tensorCells=0;bool tensorCapable=false;')
edit('test2026/metal/packed_affine_test.cpp','        eligible=gpu.PreparePackedAffineProfile(a,weight,out,m);','        eligible=gpu.PreparePackedAffineProfile(a,weight,out,m);tensorCapable=tensorCapable||gpu.PackedTensorDownEnabled();')
edit('test2026/metal/packed_affine_test.cpp','            Need(gpu.LastAffineProfileMisses()==0','''            const bool expectedTensor=v&&gpu.PackedTensorDownEnabled()&&m==512&&k==14336&&n==4096&&count==512&&dense;
            Need(gpu.LastPackedTensorDownDispatches()==size_t(expectedTensor),"Wrong actual tensor down route");
            if(rep>=0&&expectedTensor)++tensorCells;
            Need(gpu.LastAffineProfileMisses()==0''')
edit('test2026/metal/packed_affine_test.cpp','Reject([&]{gpu.SetPackedAffineEnabled(false);});Reject([&]{gpu.PreparePackedAffineProfile','Reject([&]{gpu.SetPackedAffineEnabled(false);});Reject([&]{gpu.SetPackedTensorDownEnabled(false);});Reject([&]{gpu.PreparePackedAffineProfile',2)
edit('test2026/metal/packed_affine_test.cpp','gpu.Abort();Need(gpu.Download(out)==before,"Invalid call wrote output");','gpu.Abort();Need(gpu.LastPackedTensorDownDispatches()==0,"Tensor counter retained after Abort");Need(gpu.Download(out)==before,"Invalid call wrote output");')
edit('test2026/metal/packed_affine_test.cpp','        auto unready=gpu.Allocate(m,k);','''        if(gpu.PackedTensorDownEnabled()) {
            gpu.SetPackedTensorDownEnabled(false);gpu.Begin();gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,ids);gpu.SubmitAndWait();Need(gpu.LastPackedTensorDownDispatches()==0,"Disabled tensor route executed");gpu.SetPackedTensorDownEnabled(true);
        }else Reject([&]{gpu.SetPackedTensorDownEnabled(true);});
        auto unready=gpu.Allocate(m,k);''')
edit('test2026/metal/packed_affine_test.cpp','report["cases"].push_back(f.Cell(512,false,true,benchmark));f.Guards();','''report["cases"].push_back(f.Cell(512,false,true,benchmark));
        if(!benchmark&&shape[0]==4096&&shape[1]==14336)for(int pattern=0;pattern<2;++pattern){
            auto saved=f.input;
            for(size_t i=0;i<f.input.size();++i){float x=pattern?std::nextafter(0x1.002p0f,(i%3==0)?0.f:2.f):float(int(i%23)-11)*128.f;f.input[i]=(i&1)?x:-x;}
            f.gpu.Update(f.a,f.input);report["cases"].push_back(f.Cell(512,false,true,false));f.input=std::move(saved);f.gpu.Update(f.a,f.input);
        }
        f.Guards();''')
edit('test2026/metal/packed_affine_test.cpp','    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices','    Need(!tensorCapable||tensorCells>=3,"No qualified tensor down cells");report["tensor_down_cells"]=tensorCells;report["tensor_down_capable"]=tensorCapable;\n    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices')
report=dict(passed=True,changed_files=sorted(set(changes)),base_native=str(M/'mistral-causal-batch4-helper1'),prototype_shader_sha256=hashlib.sha256((P/'shared3.metal').read_bytes()).hexdigest(),header_sha256=hashlib.sha256(header.encode()).hexdigest(),default_off=True,no_additional_gpu_buffers=True,operator_interface_changed=False,full_model_qualified=False,goal_complete=False)
(R/'preparation1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
