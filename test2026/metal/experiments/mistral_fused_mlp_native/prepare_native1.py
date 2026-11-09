"""Prepare an opt-in native fused Mistral MLP overlay from the clean engine."""
from pathlib import Path
import hashlib,json,re,shutil,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
assert ROOT.parent.name=='mistral7b-local-coder' and not ROOT.exists()
ROOT.mkdir()
for name in ['fused_context1.inc', 'fused_prepare1.inc', 'fused_graph_prepare1.inc', 'fused_mlp_use_proof.h', 'native_test1.mm', 'use_proof_test1.cpp', 'worker-model1.py', 'profile-model1.json', 'run_model1.py', 'audit_model1.py', 'run_suite1.py', 'run_default_off1.py', 'build_native1.py', 'prepare_prefix1.py', 'launch_prefix1.py', 'audit_prefix1.py']:shutil.copyfile(PACKAGE/name,ROOT/name)
profile=json.loads((ROOT/'profile-model1.json').read_text());profile['memory_inventory_path']=str(ROOT/'engine-inventory1.json');(ROOT/'profile-model1.json').write_text(json.dumps(profile,indent=2)+'\n')
M=ROOT.parent
BASE=M/'mistral-half-attention-native1'
OLD=M/'mistral-fused-gate-silu-screen1'
TRAINED=M/'mistral-fused-mlp-trained1'
SOURCE=ROOT/'source2'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(BASE/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
assert sha(TRAINED/'provenance1.json')=='191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a'
assert json.loads((TRAINED/'replay-audit1.json').read_text())['passed']
assert not SOURCE.exists();shutil.copytree(BASE/'source7',SOURCE,symlinks=True)
def edit(path,old,new):
    p=SOURCE/path;p.chmod(0o644);s=p.read_text();assert s.count(old)==1,(path,old[:70],s.count(old));p.write_text(s.replace(old,new))
shader=(OLD/'combined1.metal').read_text();shader=shader[shader.index('constant bool fused_shadow'):]
p=SOURCE/'src/backends/metal/fused_mistral_mlp_shader.h';p.write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {inline constexpr const char* kFusedMistralMlpShader=R"METAL(\n'+shader+'\n)METAL";}\n')
shutil.copyfile(ROOT/'fused_mlp_use_proof.h',SOURCE/'src/backends/metal/fused_mlp_use_proof.h')
mh='src/backends/metal/metal_context.h';mm='src/backends/metal/metal_context.mm';gh='src/backends/metal/compiled_graph.h';gc='src/backends/metal/compiled_graph.cpp';dh='src/backends/metal/decoder_session.h';dc='src/backends/metal/decoder_session.cpp';tp='src/backends/metal/token_prefix_session.cpp';entry='src/entry/metal_main.cpp'
anchor='    size_t LastPackedAffineDispatches() const;'
edit(mh,anchor,anchor+'''
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
    size_t LastFusedMistralMlpBlocks() const;
    MetalMatrix FusedMistralMlpScratch() const;
    bool TryEncodeFusedMistralMlp(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalAffineInt4Weight&,
        const MetalAffineInt4Weight&,const MetalMatrix& gateLayout,const MetalMatrix& upLayout,
        const MetalMatrix& activatedLayout,const MetalMatrix& output,const std::vector<size_t>* selectedRows=nullptr);
#endif
''')
edit(mm,'#include "packed_affine_shader.h"','#include "packed_affine_shader.h"\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n#include "fused_mistral_mlp_shader.h"\n#endif')
edit(mm,'    struct PackedPlan {std::array<id<MTLComputePipelineState>,2> pipelines{};','    struct PackedPlan {std::array<id<MTLComputePipelineState>,2> pipelines{};\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n        std::array<id<MTLComputePipelineState>,2> fused{};\n#endif')
edit(mm,'    MetalMatrix packedInput;','    MetalMatrix packedInput;\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    MetalMatrix fusedMlpHalf;id<MTLLibrary> fusedMlpLibrary=nil;size_t fusedMlpBlocks=0;\n#endif')
edit(mm,'        impl_->packedPlans.emplace(key,std::move(plan));return true;',(ROOT/'fused_prepare1.inc').read_text()+'\n        impl_->packedPlans.emplace(key,std::move(plan));return true;')
edit(mm,'bool MetalContext::TryEncodePackedAffine(const MetalMatrix& input,',(ROOT/'fused_context1.inc').read_text()+'\nbool MetalContext::TryEncodePackedAffine(const MetalMatrix& input,')
p=SOURCE/mm;s=p.read_text();needle='    impl_->packedDispatches=0;impl_->diagnosticBulkDispatches=0;';assert s.count(needle)==2;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    impl_->fusedMlpBlocks=0;\n#endif');p.write_text(s)
edit(gh,'    std::vector<uint8_t> residentOps_;','    std::vector<uint8_t> residentOps_;\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    std::vector<uint8_t> fusedMlpSites_;\n#endif')
edit(gh,'    bool Ready() const {return ready_;}','#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    size_t FusedMlpSites() const {std::lock_guard<std::mutex> lock(mutex_);size_t n=0;for(auto v:fusedMlpSites_)n+=v!=0;return n;}\n#endif\n    bool Ready() const {return ready_;}')
edit(gh,'        APISET().AddProp("ready",&MetalCompiledGraph::Ready);','#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n        APISET().AddProp("fused_mlp_sites",&MetalCompiledGraph::FusedMlpSites);\n#endif\n        APISET().AddProp("ready",&MetalCompiledGraph::Ready);')
edit(gc,'#include "compiled_graph.h"','#include "compiled_graph.h"\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n#include "fused_mlp_use_proof.h"\n#endif')
edit(gc,'void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    fusedMlpSites_.clear();\n#endif')
anchor='        resetKVPrefix_[0]=ComputeResetKVPrefix(1);resetKVPrefix_[1]=ComputeResetKVPrefix(2);'
edit(gc,anchor,(ROOT/'fused_graph_prepare1.inc').read_text()+'\n'+anchor)
anchor='#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL\n        if(!profileOps&&slots_[op.a].rows>=128'
replacement='''#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
        if(!profileOps&&i<fusedMlpSites_.size()&&fusedMlpSites_[i]) {
            const auto& up=ops_[i+1];const auto& silu=ops_[i+2];const auto& down=ops_[i+3];
            const auto* selected=(selectedTokenRows&&op.indexedTokenProjection&&selectedTokenRows->size()<slots_[op.a].rows)?selectedTokenRows:nullptr;
            if(gpu.TryEncodeFusedMistralMlp(a,slots_[op.b].affineInt4,slots_[up.b].affineInt4,
                slots_[down.b].affineInt4,out,slots_[up.out].matrix,slots_[silu.out].matrix,slots_[down.out].matrix,selected)){i+=3;continue;}
        }
#endif
'''+anchor
edit(gc,anchor,replacement)
anchor='    append(gpu_->PackedAffineScratch(),SIZE_MAX,"packed_prefill_half_input",{});'
edit(dc,anchor,anchor+'\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    append(gpu_->FusedMistralMlpScratch(),SIZE_MAX,"fused_mlp_half_input",{});\n#endif')
anchor='    BEGIN_PACKAGE(MetalDecoderSession)'
edit(dh,anchor,'#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    size_t FusedMlpBlocks(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastFusedMistralMlpBlocks():0;}\n#endif\n'+anchor+'\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n        APISET().AddProp("fused_mlp_blocks",&MetalDecoderSession::FusedMlpBlocks);\n#endif')
anchor='        "|fp16-prefill-all-p16-gpupack-v1"\n#endif'
edit(tp,anchor,anchor+'\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n        "|fused-mistral-mlp-fast-v1"\n#endif')
anchor='    info+="; fp16_prefill_all_p16_gpupack_v1";'
edit(entry,anchor,anchor+'\n#ifdef GARNET_METAL_FUSED_MISTRAL_MLP\n    info+="; fused_mistral_mlp_prefill_fast_v1";\n#endif')
p=SOURCE/'cmake/GarnetMetal.cmake';p.chmod(0o644);p.write_text(p.read_text()+'''
option(GARNET_METAL_FUSED_MISTRAL_MLP "Opt-in native Mistral gate/up-SiLU-half-down fusion" OFF)
if(GARNET_METAL_FUSED_MISTRAL_MLP)
    if(NOT GARNET_METAL_CHUNK_HALF_ATTENTION OR NOT GARNET_METAL_MISTRAL_PACKED_PREFILL OR NOT GARNET_METAL_AFFINE_FP16_COEFFICIENTS OR NOT GARNET_METAL_AFFINE_PROFILE_INDEX32)
        message(FATAL_ERROR "Fused MLP requires the bounded half-prefill/FP16 packed Mistral profile")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_FUSED_MISTRAL_MLP=1)
endif()
''')
changed=[];files={}
for p in SOURCE.rglob('*'):
    if p.is_file() and not p.is_symlink():
        n=str(p.relative_to(SOURCE));files[n]=sha(p);old=BASE/'source7'/n
        if not old.exists() or sha(old)!=files[n]:changed.append(n)
assert len(changed)==11,(len(changed),changed)
result=dict(passed=True,source=str(SOURCE),changed_paths=sorted(changed),files_sha256=files,foundation_provenance_sha256=sha(BASE/'provenance7.json'),trained_gate_provenance_sha256=sha(TRAINED/'provenance1.json'),old_shader_headers_unchanged=True,xmodel_unchanged=True,default_option='OFF',numerical_policy='Existing half-input affine products, strict FP32 accumulation and Fast SiLU; private half down input.',new_scratch_bytes=14680064,quality_qualified=False,goal_complete=False)
(ROOT/'source-bindings2.json').write_text(json.dumps(result,indent=2)+'\n')
command=json.loads((BASE/'configure-on5-command.json').read_text());command[command.index('-S')+1]=str(SOURCE);command[command.index('-B')+1]=str(ROOT/'build2');command+=['-DGARNET_METAL_FUSED_MISTRAL_MLP=ON'];(ROOT/'configure2-command.json').write_text(json.dumps(command,indent=2)+'\n')
print(json.dumps(dict(prepared=True,changed=sorted(changed),source_files=len(files))))

expected_manifest_sha256='717796e56c43699edb7adce46c4e8c1f31f3af66e3061dee2ba5bbfb5dfc4746'
actual_manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert actual_manifest_sha256==expected_manifest_sha256, 'Prepared source differs from the qualified native source'
(ROOT/'preparation-proof1.json').write_text(json.dumps(dict(passed=True,source_files=len(files),source_manifest_sha256=actual_manifest_sha256,source_byte_exact_to_qualified_native=True),indent=2)+'\n')
