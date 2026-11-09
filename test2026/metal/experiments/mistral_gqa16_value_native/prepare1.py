"""Add the screened shared-value kernel to the upstream-reconciled native source."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;S=R/'source1';P=R.parent/'mistral-gqa16-value-screen1'
assert json.loads((R/'upstream1.json').read_text())['applied']
def edit(name,old,new,count=1):
 p=S/name;s=p.read_text();assert s.count(old)==count,(name,old,s.count(old));p.write_text(s.replace(old,new))
before={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file()}
(R/'upstream-source-bindings1.json').write_text(json.dumps(dict(files_sha256=before),indent=2)+'\n')
body=(P/'value1.metal').read_text()
(S/'src/backends/metal/verification_gqa_value_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationGqaValueShader=R"METAL(\n'+body+')METAL";\n}\n')
edit('cmake/GarnetMetal.cmake','option(GARNET_METAL_MISTRAL_VERIFY_VECTOR4', '''option(GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES "Opt-in ordered shared-value B1/B4 T4 Mistral verification context at visible>=512" OFF)
if(GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES)
    if(NOT GARNET_METAL_VERIFY_ATTENTION OR NOT GARNET_METAL_MISTRAL_VERIFY_LAYOUT)
        message(FATAL_ERROR "GQA16 verification values require bounded verification and the Mistral layout")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES=1)
endif()
option(GARNET_METAL_MISTRAL_VERIFY_VECTOR4''')
edit('cmake/GarnetMetal.cmake','    add_executable(garnet_mistral_verification_gqa_score_test', '''    add_executable(garnet_mistral_verification_gqa_value_test test2026/metal/mistral_verification_gqa_value_test.cpp)
    target_link_libraries(garnet_mistral_verification_gqa_value_test PRIVATE garnet_metal)
    add_test(NAME metal_mistral_verification_gqa_value_reference COMMAND garnet_mistral_verification_gqa_value_test)
    add_executable(garnet_mistral_verification_gqa_score_test''')
f='src/backends/metal/metal_context.mm'
edit(f,'#include "verification_gqa_score_shader.h"','#include "verification_gqa_score_shader.h"')
edit(f,'#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n#include', '#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES\n#include "verification_gqa_value_shader.h"\n#endif\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n#include')
edit(f,'    id<MTLComputePipelineState> verificationScores,verificationContext;', '''    id<MTLComputePipelineState> verificationScores,verificationContext;
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    std::array<id<MTLComputePipelineState>,2> verificationGqaValues{};
#endif''')
edit(f,'    size_t verificationGqaScoreDispatches=0;', '''    size_t verificationGqaScoreDispatches=0;
    size_t verificationGqaValueDispatches=0;
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    bool verificationGqaValuesEnabled=true;
#endif''')
edit(f,'        impl_->kernels->verificationConvCheckpoint=[', '''#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
        for(size_t index=0;index<2;++index) {
            NSString* source=[[NSString stringWithUTF8String:kVerificationAttentionShader]
                stringByAppendingString:[NSString stringWithUTF8String:kVerificationGqaValueShader]];
            // The screened B1 library uses the existing layout constant policy.
            // B4 retains runtime batch; only the tile is a function constant.
            if(index==0)source=[@"#define GARNET_VERIFY_MISTRAL_LAYOUT 1\\n" stringByAppendingString:source];
            auto valueLibrary=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];
            if(!valueLibrary)throw std::runtime_error("Cannot compile prepared GQA16 verification values: "+std::string(error.localizedDescription.UTF8String));
            MTLFunctionConstantValues* constants=[[MTLFunctionConstantValues alloc] init];
            const unsigned tile=index==0?64:32;
            [constants setConstantValue:&tile type:MTLDataTypeUInt atIndex:2];
            auto function=[valueLibrary newFunctionWithName:@"verification_gqa16_context" constantValues:constants error:&error];
            auto pipeline=function?[impl_->device newComputePipelineStateWithFunction:function error:&error]:nil;
            const size_t dynamicBytes=tile*48*sizeof(float);
            if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||
               pipeline.staticThreadgroupMemoryLength+dynamicBytes>impl_->device.maxThreadgroupMemoryLength)
                throw std::runtime_error("GQA16 verification value pipeline exceeds prepared device limits");
            impl_->kernels->verificationGqaValues[index]=pipeline;
        }
#endif
        impl_->kernels->verificationConvCheckpoint=[''')
edit(f,'impl_->verificationGqaScoreDispatches=0;', 'impl_->verificationGqaScoreDispatches=0;impl_->verificationGqaValueDispatches=0;',count=2)
edit(f,'                ++impl_->verificationValueVectorDispatches;\n','')
anchor='#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    // Only the typed attention entry'
route='''#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    bool useGqaValue=false;size_t gqaValueScratchBytes=0;
    if(operation==23&&impl_->verificationGqaValuesEnabled&&inputs.size()==6&&
       parameters&&parameterBytes==17*sizeof(uint64_t)) {
        const auto* p=static_cast<const uint64_t*>(parameters);
        bool known=(p[0]==1||p[0]==4)&&p[1]==4&&p[2]==32&&p[3]==8&&p[4]==128&&p[5]==4096&&p[16]==0;
        auto bounded=[](const MPSMatrix* matrix){
            const size_t stride=matrix.rowBytes/sizeof(float);
            return matrix.rowBytes%sizeof(float)==0&&matrix.rows>0&&stride>0&&stride<=UINT32_MAX/matrix.rows;
        };
        if(known){known=bounded(c);for(const auto& input:inputs)known=known&&bounded(input.impl_->matrix);}
        const auto slots=inputs[4].impl_->matrix,active=inputs[5].impl_->matrix;
        known=known&&inputs[4].impl_->ready&&!inputs[4].impl_->pending&&inputs[5].impl_->ready&&!inputs[5].impl_->pending&&
            slots.data.storageMode==MTLStorageModeShared&&active.data.storageMode==MTLStorageModeShared&&slots.data.contents&&active.data.contents;
        size_t visible=0;
        if(known)for(size_t row=0;row<p[0];++row) {
            bool ended=false;size_t first=0;
            for(size_t token=0;token<4;++token) {
                const float flag=reinterpret_cast<const float*>(static_cast<const char*>(active.data.contents)+active.offset)[row*(active.rowBytes/4)+token];
                if(flag!=0.0f&&flag!=1.0f){known=false;break;}
                if(flag==0.0f){ended=true;continue;}
                const float slot=reinterpret_cast<const float*>(static_cast<const char*>(slots.data.contents)+slots.offset)[row*(slots.rowBytes/4)+token];
                if(ended||!std::isfinite(slot)||slot<0||slot>=4096||slot!=std::floor(slot)){known=false;break;}
                if(!token)first=size_t(slot);
                if(size_t(slot)!=first+token){known=false;break;}
                visible=std::max(visible,size_t(slot)+1);
            }
            if(!known)break;
        }
        if(known&&visible>=512&&visible<=4096) {
            pipeline=impl_->kernels->verificationGqaValues[p[0]==1?0:1];
            if(!pipeline)throw std::logic_error("Prepared GQA16 verification value pipeline unavailable");
            useGqaValue=true;gqaValueScratchBytes=(p[0]==1?64:32)*48*sizeof(float);
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
            useValueVector4=false;
#endif
        }
    }
#endif
'''
edit(f,anchor,route+anchor)
edit(f,'#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4\n    } else if(useValueVector4)', '''#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    } else if(useGqaValue) {
        const auto* p=static_cast<const uint64_t*>(parameters);
        [encoder setThreadgroupMemoryLength:gqaValueScratchBytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(p[0]*8,4,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        ++impl_->verificationGqaValueDispatches;
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
    } else if(useValueVector4)''')
edit(f,'        [encoder dispatchThreads:MTLSizeMake(threads/4,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];','        [encoder dispatchThreads:MTLSizeMake(threads/4,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];\n        ++impl_->verificationValueVectorDispatches;')
edit(f,'size_t MetalContext::LastVerificationGqaScoreDispatches() const', '''size_t MetalContext::LastVerificationGqaValueDispatches() const {return impl_->verificationGqaValueDispatches;}
void MetalContext::VerificationSetGqaValuesEnabled(bool enabled) {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("GQA16 verification values require a closed, drained command batch");
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    impl_->verificationGqaValuesEnabled=enabled;
#else
    throw std::logic_error("GQA16 verification values are disabled in this build");
#endif
}
size_t MetalContext::LastVerificationGqaScoreDispatches() const''')
edit('src/backends/metal/metal_context.h','    size_t LastVerificationGqaScoreDispatches() const;','    size_t LastVerificationGqaValueDispatches() const;\n    void VerificationSetGqaValuesEnabled(bool enabled);\n    size_t LastVerificationGqaScoreDispatches() const;')
edit('src/backends/metal/verification_session.cpp','{"target_attention_gqa16_score_dispatches",gpu_->LastVerificationGqaScoreDispatches()},','{"target_attention_gqa16_score_dispatches",gpu_->LastVerificationGqaScoreDispatches()},{"target_attention_gqa16_value_dispatches",gpu_->LastVerificationGqaValueDispatches()},')
edit('src/entry/metal_main.cpp','#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    info+=', '#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES\n    info+="; native_mistral_verification_gqa16_values64_32_optional_v1";\n#endif\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    info+=')
after={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file()}
(R/'prepare-source1.json').write_text(json.dumps(dict(passed=True,changed_files=[n for n,h in after.items() if before.get(n)!=h],value_shader_identical_to_screen=True,global_workspace_bytes=0,prepared_tiles_by_batch={'1':64,'4':32},default_option_on=False,xmodel_operator_changes=False),indent=2)+'\n')
print('Prepared shared-value native source.')
