"""Create a new isolated runtime source; never build or edit the dirty root."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1';S=R/'source1';P=M/'mistral-gqa16-score-screen1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists();subprocess.run(['/bin/cp','-cR',str(B/'source1'),str(S)],check=True)
for p in S.rglob('*'):
 if p.is_file():p.chmod(p.stat().st_mode|0o200)
 elif p.is_dir():p.chmod(p.stat().st_mode|0o700)
S.chmod(S.stat().st_mode|0o700)
base=json.loads((B/'source-bindings1.json').read_text())['files_sha256']
assert {str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()}==base
assert json.loads((P/'audit2.json').read_text())['passed']
for name in ['dots1.metal','softmax1.metal']:
 assert sha(P/name)==json.loads((P/'execution-bindings1.json').read_text())[name]
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def edit(name,fn):
 p=S/name;p.write_text(fn(p.read_text()))
dot=(P/'dots1.metal').read_text()
dot=replace(dot,' const uint first=uint(slots[row*uint(p.sStride)]);\n','')
dot=replace(dot,' threadgroup float keys[32*128];',' // All-inactive rows may have NaN slots. Never convert or address them.\n if(maximum==0)return;\n const uint first=uint(slots[row*uint(p.sStride)]);\n threadgroup float keys[32*128];')
(S/'src/backends/metal/verification_gqa_score_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationGqaScoreShader=R"METAL(\n'+dot+(P/'softmax1.metal').read_text()+')METAL";\n}\n')
def cmake(s):
 mark='option(GARNET_METAL_MISTRAL_VERIFY_VECTOR4 '
 pos=s.index(mark)
 s=s[:pos]+'''option(GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES "Opt-in shared-key B1/B4 T4 Mistral verification scores at visible>=512" OFF)
if(GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES)
    if(NOT GARNET_METAL_VERIFY_ATTENTION OR NOT GARNET_METAL_MISTRAL_VERIFY_LAYOUT)
        message(FATAL_ERROR "GQA16 verification scores require bounded verification and the Mistral layout")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES=1)
endif()
'''+s[pos:]
 mark='    add_executable(garnet_mistral_verification_layout_test '
 pos=s.index(mark)
 s=s[:pos]+'''    add_executable(garnet_mistral_verification_gqa_score_test test2026/metal/mistral_verification_gqa_score_test.cpp)
    target_link_libraries(garnet_mistral_verification_gqa_score_test PRIVATE garnet_metal)
    add_test(NAME metal_mistral_verification_gqa_scores COMMAND garnet_mistral_verification_gqa_score_test)
    set_tests_properties(metal_mistral_verification_gqa_scores PROPERTIES TIMEOUT 300)
'''+s[pos:]
 return s
edit('cmake/GarnetMetal.cmake',cmake)
def context(s):
 s=replace(s,'#include "metal_context.h"','#include "metal_context.h"\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n#include "verification_gqa_score_shader.h"\n#endif')
 s=replace(s,'    id<MTLComputePipelineState> verificationScores,verificationContext;','    id<MTLComputePipelineState> verificationScores,verificationContext;\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    std::array<id<MTLComputePipelineState>,2> verificationGqaDots{},verificationGqaSoftmax{};\n#endif')
 s=replace(s,'    size_t verificationSplitScoreDispatches=0;','    size_t verificationSplitScoreDispatches=0;\n    size_t verificationGqaScoreDispatches=0;\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    bool verificationGqaScoresEnabled=true;\n#endif')
 assert s.count('impl_->verificationSplitScoreDispatches=0;')==2
 s=s.replace('impl_->verificationSplitScoreDispatches=0;','impl_->verificationSplitScoreDispatches=0;impl_->verificationGqaScoreDispatches=0;')
 marker='        impl_->kernels->verificationConvCheckpoint='
 pos=s.index(marker)
 s=s[:pos]+'''#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
        for(size_t layout=0;layout<2;++layout) {
            NSString* gqaSource=[[NSString stringWithUTF8String:kVerificationAttentionShader]
                stringByAppendingString:[NSString stringWithUTF8String:kVerificationGqaScoreShader]];
            if(layout)gqaSource=[@"#define GARNET_VERIFY_MISTRAL_LAYOUT 1\\n" stringByAppendingString:gqaSource];
            auto library=[impl_->device newLibraryWithSource:gqaSource options:compileOptions error:&error];
            if(!library)throw std::runtime_error("Cannot compile prepared GQA16 verification scores: "+std::string(error.localizedDescription.UTF8String));
            auto make=[&](NSString* name,size_t dynamicBytes) {
                auto function=[library newFunctionWithName:name];
                auto pipeline=function?[impl_->device newComputePipelineStateWithFunction:function error:&error]:nil;
                if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||
                   pipeline.staticThreadgroupMemoryLength+dynamicBytes>impl_->device.maxThreadgroupMemoryLength)
                    throw std::runtime_error("GQA16 verification pipeline exceeds prepared device limits");
                return pipeline;
            };
            impl_->kernels->verificationGqaDots[layout]=make(@"verification_gqa16_dots",0);
            impl_->kernels->verificationGqaSoftmax[layout]=make(@"verification_gqa16_softmax",4096*sizeof(float));
        }
#endif
'''+s[pos:]
 marker='#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES\n    // EncodeVerificationAttention has validated'
 pos=s.index(marker)
 s=s[:pos]+'''#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    // Only the typed attention entry reaches operation22 after validating the
    // full layout and contiguous immutable controls. Prove uint32 spans again
    // before selecting the fixed-layout shared-key kernel. Other shapes and
    // short prefixes retain their existing dispatch.
    if(operation==22&&impl_->verificationGqaScoresEnabled&&inputs.size()==5&&
       parameters&&parameterBytes==17*sizeof(uint64_t)) {
        const auto* p=static_cast<const uint64_t*>(parameters);
        bool known=(p[0]==1||p[0]==4)&&p[1]==4&&p[2]==32&&p[3]==8&&p[4]==128&&p[5]==4096&&p[16]==0;
        auto bounded=[](const MPSMatrix* matrix){
            const size_t stride=matrix.rowBytes/sizeof(float);
            return matrix.rowBytes%sizeof(float)==0&&matrix.rows>0&&stride>0&&stride<=UINT32_MAX/matrix.rows;
        };
        if(known){known=bounded(c);for(const auto& input:inputs)known=known&&bounded(input.impl_->matrix);}
        const auto slots=inputs[3].impl_->matrix,active=inputs[4].impl_->matrix;
        known=known&&inputs[3].impl_->ready&&!inputs[3].impl_->pending&&inputs[4].impl_->ready&&!inputs[4].impl_->pending&&
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
            const size_t layout=pipeline==impl_->kernels->mistralVerificationScores?1:0;
            auto dotsPipeline=impl_->kernels->verificationGqaDots[layout],softmaxPipeline=impl_->kernels->verificationGqaSoftmax[layout];
            if(!dotsPipeline||!softmaxPipeline)throw std::logic_error("Prepared GQA16 verification pipelines unavailable");
            // Register before either encoder: partial Flush/Abort must invalidate
            // the shared probability output just like the existing split route.
            output.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(output.impl_->PendingKey()));
            output.impl_->ready=false;output.impl_->pending=true;
            if(impl_->pendingOutputs.insert(output.impl_->PendingKey()).second)impl_->outputs.push_back(output.impl_);
            auto bind=[&](id<MTLComputeCommandEncoder> encoder){
                for(size_t i=0;i<inputs.size();++i){const auto m=inputs[i].impl_->matrix;[encoder setBuffer:m.data offset:m.offset atIndex:i];}
                [encoder setBuffer:c.data offset:c.offset atIndex:5];[encoder setBytes:parameters length:parameterBytes atIndex:6];
            };
            auto dots=impl_->ComputeEncoder(__func__);if(!dots)throw std::runtime_error("Cannot encode GQA16 verification dots");
            bind(dots);[dots setComputePipelineState:dotsPipeline];
            [dots dispatchThreadgroups:MTLSizeMake(p[0]*8,(visible+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[dots endEncoding];
            auto softmax=impl_->ComputeEncoder(__func__);if(!softmax)throw std::runtime_error("Cannot encode GQA16 verification softmax");
            bind(softmax);[softmax setComputePipelineState:softmaxPipeline];[softmax setThreadgroupMemoryLength:4096*sizeof(float) atIndex:0];
            [softmax dispatchThreadgroups:MTLSizeMake(p[0]*4*32,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[softmax endEncoding];
            ++impl_->verificationGqaScoreDispatches;return;
        }
    }
#endif
'''+s[pos:]
 marker='size_t MetalContext::LastVerificationSplitScoreDispatches()'
 pos=s.index(marker)
 s=s[:pos]+'''size_t MetalContext::LastVerificationGqaScoreDispatches() const {return impl_->verificationGqaScoreDispatches;}
void MetalContext::VerificationSetGqaScoresEnabled(bool enabled) {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("GQA16 verification scores require a closed, drained command batch");
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    impl_->verificationGqaScoresEnabled=enabled;
#else
    throw std::logic_error("GQA16 verification scores are disabled in this build");
#endif
}
'''+s[pos:]
 return s
edit('src/backends/metal/metal_context.mm',context)
edit('src/backends/metal/metal_context.h',lambda s:replace(s,'    size_t LastVerificationSplitScoreDispatches() const;','    size_t LastVerificationSplitScoreDispatches() const;\n    size_t LastVerificationGqaScoreDispatches() const;\n    void VerificationSetGqaScoresEnabled(bool enabled);'))
edit('src/backends/metal/verification_session.cpp',lambda s:replace(s,'{"target_attention_split_score_dispatches",gpu_->LastVerificationSplitScoreDispatches()},','{"target_attention_split_score_dispatches",gpu_->LastVerificationSplitScoreDispatches()},{"target_attention_gqa16_score_dispatches",gpu_->LastVerificationGqaScoreDispatches()},'))
edit('src/entry/metal_main.cpp',lambda s:replace(s,'    info+="; native_mistral_verification_split_scores128_default_v1";','    info+="; native_mistral_verification_split_scores128_default_v1";\n#endif\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n    info+="; native_mistral_verification_gqa16_scores32_optional_v1";'))
# Existing route-specific tests continue to select the route they qualify.
for name in ['mistral_verification_split_score_test.cpp','mistral_verification_layout_test.cpp']:
 edit('test2026/metal/'+name,lambda s:replace(s,'        MetalContext gpu,other;','        MetalContext gpu,other;\n#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES\n        gpu.VerificationSetGqaScoresEnabled(false);\n#endif'))
print('Prepared isolated GQA runtime source; native test to follow.')
