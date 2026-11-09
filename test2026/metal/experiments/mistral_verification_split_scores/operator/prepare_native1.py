"""Create an isolated native source snapshot; protected Garnet root untouched."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-fused-q4-decode-native1';V=M/'mistral-verification-split-score-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'audit1.json')['passed'] and read(R/'audit1.json')['declared_component_gate_passed']
assert not V.exists();V.mkdir();S=V/'source1'
subprocess.run(['/bin/cp','-cR',str(N/'source3'),str(S)],check=True)
changes={}
def edit(name,a,b):
 p=S/name;s=p.read_text();assert s.count(a)==1,(name,s.count(a),a[:120]);p.write_text(s.replace(a,b));changes[name]=sha(p)
def new(name,s):
 p=S/name;assert not p.exists();p.write_text(s);changes[name]=sha(p)
cm='cmake/GarnetMetal.cmake'
edit(cm,'option(GARNET_METAL_MISTRAL_VERIFY_VECTOR4 ','''option(GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES "Opt-in same-order tiled Mistral B1/T4 verification scores at visible>=512" OFF)
if(GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES)
    if(NOT GARNET_METAL_MISTRAL_VERIFY_LAYOUT)
        message(FATAL_ERROR "Split verification scores require the prepared Mistral verification layout")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES=1)
endif()
option(GARNET_METAL_MISTRAL_VERIFY_VECTOR4 ''')
edit(cm,'    add_executable(garnet_mistral_verification_layout_test ','''    add_executable(garnet_mistral_verification_split_score_test test2026/metal/mistral_verification_split_score_test.cpp)
    target_link_libraries(garnet_mistral_verification_split_score_test PRIVATE garnet_metal)
    add_test(NAME metal_mistral_verification_split_scores COMMAND garnet_mistral_verification_split_score_test)
    set_tests_properties(metal_mistral_verification_split_scores PROPERTIES TIMEOUT 240)
    add_executable(garnet_mistral_verification_layout_test ''')
base='src/backends/metal/';mm=base+'metal_context.mm'
new(base+'verification_split_score_shader.h','// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kVerificationSplitScoreShader=R"METAL('+ (R/'split1.metal').read_text()+')METAL";\n}\n')
edit(mm,'#include "metal_context.h"','#include "metal_context.h"\n#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES\n#include "verification_split_score_shader.h"\n#endif')
edit(mm,'    id<MTLComputePipelineState> mistralVerificationScores,mistralVerificationContext;','''    id<MTLComputePipelineState> mistralVerificationScores,mistralVerificationContext;
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    id<MTLComputePipelineState> mistralVerificationScoreDots,mistralVerificationScoreSoftmax;
#endif''')
edit(mm,'    size_t verificationLayoutDispatches=0;','''    size_t verificationLayoutDispatches=0;
    size_t verificationSplitScoreDispatches=0;
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    bool verificationSplitScoresEnabled=true;
#endif''')
edit(mm,'        auto layoutLibrary=[impl_->device newLibraryWithSource:layoutSource options:compileOptions error:&error];','''#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
        layoutSource=[layoutSource stringByAppendingString:[NSString stringWithUTF8String:kVerificationSplitScoreShader]];
#endif
        auto layoutLibrary=[impl_->device newLibraryWithSource:layoutSource options:compileOptions error:&error];''')
edit(mm,'        impl_->kernels->mistralVerificationContext=layoutPipeline(@"verification_attention_context");','''        impl_->kernels->mistralVerificationContext=layoutPipeline(@"verification_attention_context");
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
        auto scoreConstants=[MTLFunctionConstantValues new];uint32_t scoreTile=128;
        [scoreConstants setConstantValue:&scoreTile type:MTLDataTypeUInt atIndex:1];
        auto scoreDots=[layoutLibrary newFunctionWithName:@"verification_split_score_dots" constantValues:scoreConstants error:&error];
        auto scoreDotsPipeline=scoreDots?[impl_->device newComputePipelineStateWithFunction:scoreDots error:&error]:nil;
        auto scoreSoftmaxPipeline=layoutPipeline(@"verification_split_score_softmax");
        if(!scoreDotsPipeline||scoreDotsPipeline.threadExecutionWidth!=32||scoreDotsPipeline.maxTotalThreadsPerThreadgroup<128||
           scoreDotsPipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength||
           scoreSoftmaxPipeline.staticThreadgroupMemoryLength+4096*sizeof(float)>impl_->device.maxThreadgroupMemoryLength)
            throw std::runtime_error("Split verification score pipelines exceed prepared device limits");
        impl_->kernels->mistralVerificationScoreDots=scoreDotsPipeline;
        impl_->kernels->mistralVerificationScoreSoftmax=scoreSoftmaxPipeline;
#endif''')
# Keep both Begin and Abort reset sites explicit and bound.
p=S/mm;s=p.read_text();a='impl_->verificationAttentionDispatches=0;impl_->verificationLayoutDispatches=0;impl_->verificationValueVectorDispatches=0;';assert s.count(a)==2;p.write_text(s.replace(a,a+'impl_->verificationSplitScoreDispatches=0;'));changes[mm]=sha(p)
a='#ifdef GARNET_METAL_SPLIT_DECODE_SCORES\n    if(operation==8&&impl_->splitScoresEnabled'
b='''#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    // EncodeVerificationAttention has validated dimensions, nonpending shared
    // controls and contiguous active slots. Keep the profile selection tied to
    // the prepared layout pipeline so all other shapes use the original path.
    if(operation==22&&impl_->verificationSplitScoresEnabled&&
       pipeline==impl_->kernels->mistralVerificationScores&&inputs.size()==5&&
       parameters&&parameterBytes==17*sizeof(uint64_t)) {
        const auto* p=static_cast<const uint64_t*>(parameters);
        const auto& slotOwner=inputs[3].impl_;const auto& activeOwner=inputs[4].impl_;
        const auto slots=slotOwner->matrix,active=activeOwner->matrix;
        bool known=slotOwner->ready&&!slotOwner->pending&&activeOwner->ready&&!activeOwner->pending&&
            slots.data.storageMode==MTLStorageModeShared&&active.data.storageMode==MTLStorageModeShared&&slots.data.contents&&active.data.contents;
        size_t visible=0;
        if(known)for(size_t token=0;token<4;++token) {
            const float flag=reinterpret_cast<const float*>(static_cast<const char*>(active.data.contents)+active.offset)[token];
            if(flag!=0.0f&&flag!=1.0f){known=false;break;}
            if(flag==0.0f)continue;
            const float slot=reinterpret_cast<const float*>(static_cast<const char*>(slots.data.contents)+slots.offset)[token];
            if(!std::isfinite(slot)||slot<0.0f||slot>=4096.0f||slot!=std::floor(slot)){known=false;break;}
            visible=std::max(visible,size_t(slot)+1);
        }
        if(known&&visible>=512&&visible<=4096) {
            // Reuse the explicit FP32 probability allocation for raw dots.
            // Register ownership before encoding so Abort invalidates it even
            // if creating the second encoder fails. No workspace allocation.
            output.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(output.impl_->PendingKey()));
            output.impl_->ready=false;output.impl_->pending=true;
            if(impl_->pendingOutputs.insert(output.impl_->PendingKey()).second)impl_->outputs.push_back(output.impl_);
            auto bind=[&](id<MTLComputeCommandEncoder> encoder){
                for(size_t i=0;i<inputs.size();++i){const auto m=inputs[i].impl_->matrix;[encoder setBuffer:m.data offset:m.offset atIndex:i];}
                [encoder setBuffer:c.data offset:c.offset atIndex:5];[encoder setBytes:parameters length:parameterBytes atIndex:6];
            };
            auto dots=impl_->ComputeEncoder(__func__);if(!dots)throw std::runtime_error("Cannot encode verification split score dots");
            bind(dots);[dots setComputePipelineState:impl_->kernels->mistralVerificationScoreDots];
            [dots dispatchThreadgroups:MTLSizeMake(128,(visible+127)/128,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[dots endEncoding];
            // Tracked same-buffer encoder hazard orders dots before softmax.
            auto softmax=impl_->ComputeEncoder(__func__);if(!softmax)throw std::runtime_error("Cannot encode verification split score softmax");
            bind(softmax);[softmax setComputePipelineState:impl_->kernels->mistralVerificationScoreSoftmax];
            [softmax setThreadgroupMemoryLength:4096*sizeof(float) atIndex:0];
            [softmax dispatchThreadgroups:MTLSizeMake(128,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[softmax endEncoding];
            ++impl_->verificationSplitScoreDispatches;return;
        }
    }
#endif
#ifdef GARNET_METAL_SPLIT_DECODE_SCORES
    if(operation==8&&impl_->splitScoresEnabled'''
edit(mm,a,b)
edit(mm,'size_t MetalContext::LastVerificationLayoutDispatches() const {return impl_->verificationLayoutDispatches;}','''size_t MetalContext::LastVerificationLayoutDispatches() const {return impl_->verificationLayoutDispatches;}
size_t MetalContext::LastVerificationSplitScoreDispatches() const {return impl_->verificationSplitScoreDispatches;}
void MetalContext::VerificationSetSplitScoresEnabled(bool enabled) {
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Verification split scores require a closed, drained command batch");
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    impl_->verificationSplitScoresEnabled=enabled;
#else
    throw std::logic_error("Split verification scores are disabled in this build");
#endif
}''')
edit(base+'metal_context.h','    size_t LastVerificationLayoutDispatches() const;','    size_t LastVerificationLayoutDispatches() const;\n    size_t LastVerificationSplitScoreDispatches() const;\n    void VerificationSetSplitScoresEnabled(bool enabled);')
edit(base+'verification_session.cpp','{"target_attention_layout_dispatches",gpu_->LastVerificationLayoutDispatches()},','{"target_attention_layout_dispatches",gpu_->LastVerificationLayoutDispatches()},{"target_attention_split_score_dispatches",gpu_->LastVerificationSplitScoreDispatches()},')
edit('src/entry/metal_main.cpp','    #ifdef GARNET_METAL_FUSED_Q4_DECODE\n    info+=','''#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    info+="; native_mistral_verification_split_scores128_default_v1";
#endif
    #ifdef GARNET_METAL_FUSED_Q4_DECODE
    info+=''')
edit(base+'token_prefix_session.cpp','#if defined(GARNET_METAL_PREFIX_PREFILL_ONLY)||defined(GARNET_METAL_COMMITTED_KV_MPS)||defined(GARNET_METAL_CHUNK_HALF_ATTENTION)','#if defined(GARNET_METAL_PREFIX_PREFILL_ONLY)||defined(GARNET_METAL_COMMITTED_KV_MPS)||defined(GARNET_METAL_CHUNK_HALF_ATTENTION)||defined(GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES)')
edit(base+'token_prefix_session.cpp','#ifdef GARNET_METAL_FUSED_Q4_DECODE\n        "|fused-q4-decode-rows4-default-v1"\n#endif','#ifdef GARNET_METAL_FUSED_Q4_DECODE\n        "|fused-q4-decode-rows4-default-v1"\n#endif\n#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES\n        "|mistral-verify-split-scores128-default-v1"\n#endif')
# Extend the qualified ordinary-reference lifecycle test in a separate target.
name='test2026/metal/mistral_verification_split_score_test.cpp';p=S/'test2026/metal/mistral_verification_layout_test.cpp';s=p.read_text()
s=s.replace('bool allowVector=true) {','bool allowVector=true,bool allowSplit=true) {')
s=s.replace('Need(gpu.LastVerificationLayoutDispatches()==0,"Begin retained layout counter");','Need(gpu.LastVerificationLayoutDispatches()==0,"Begin retained layout counter");Need(gpu.LastVerificationSplitScoreDispatches()==0,"Begin retained split score counter");')
a='    Need(gpu.LastVerificationValueVectorDispatches()==size_t(vector),"Value vector route differs");'
b=a+'''
    size_t visible=0;for(size_t t=0;t<j.chunk;++t)if(j.active[t])visible=std::max(visible,size_t(j.slots[t])+1);
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    const bool split=profiled&&allowSplit&&visible>=512;
#else
    const bool split=false;
#endif
    Need(gpu.LastVerificationSplitScoreDispatches()==size_t(split),"Split verification route differs");
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    {bool rejected=false;try{gpu.VerificationSetSplitScoresEnabled(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationSplitScoreDispatches()==size_t(split),"Open/partial batch accepted split score route change");++rejections;}
#endif'''
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('gpu.LastVerificationAttentionDispatches()==0&&gpu.LastVerificationValueVectorDispatches()==0','gpu.LastVerificationAttentionDispatches()==0&&gpu.LastVerificationValueVectorDispatches()==0&&gpu.LastVerificationSplitScoreDispatches()==0')
a='        Invalid(gpu,other);'
b='''        for(size_t base:{size_t(507),size_t(508),size_t(511),size_t(2048)})for(size_t mask=0;mask<5;++mask)
            Case(gpu,Job(1,4,32,8,128,4096,base,mask),false,checks%3);
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
        gpu.VerificationSetSplitScoresEnabled(false);
        Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,0,true,true,false);
        gpu.VerificationSetSplitScoresEnabled(true);
        Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,2);
#else
        {bool rejected=false;try{gpu.VerificationSetSplitScoresEnabled(true);}catch(const std::logic_error&){rejected=true;}
         Need(rejected,"Disabled build accepted split score selection");++rejections;}
#endif
        Invalid(gpu,other);'''
assert s.count(a)==1;s=s.replace(a,b).replace('Mistral verification layout reference PASS','Mistral verification split scores PASS')
new(name,s)
# Exhaustive source binding: copied source remains authoritative; no dirty root build.
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
(V/'source-bindings1.json').write_text(json.dumps({'files_sha256':files,'files':len(files),'base_source':str(N/'source3'),'changes_sha256':changes,'operator_audit_sha256':sha(R/'audit1.json'),'default_option_on':False,'goal_complete':False},indent=2)+'\n')
(V/'preparation1.json').write_text(json.dumps({'passed':True,'script_sha256':sha(Path(__file__)),'source_bindings_sha256':sha(V/'source-bindings1.json'),'changed_files':sorted(changes),'root_untouched':True,'no_model_or_serving_admission':True,'goal_complete':False},indent=2)+'\n')
print(json.dumps({'prepared':str(S),'files':len(files),'changes':sorted(changes)}))
