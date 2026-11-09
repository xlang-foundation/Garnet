"""Create an isolated same-library packed descriptor screen from pinned source."""
import sys
from pathlib import Path
import hashlib,json,re
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
base=root.parent/'mistral-half-attention-native1'
source=base/'source7/src/backends/metal'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
raw=lambda name:re.search(r'R"METAL\((.*?)\)METAL"',(source/name).read_text(),re.S).group(1)
packed=raw('packed_affine_shader.h');guard=raw('packed_affine_layout_guard.h')
assert sha(source/'packed_affine_shader.h')=='cc3227e67d52d0d5e441fcaceaca0cabc4dd9d3478d06237e555d5c117b0f8d4'
candidate=packed[packed.index('kernel void half_m128k128'):]
assert candidate.count('false,true,false,')==2
candidate=candidate.replace('false,true,false,','false,true,true,').replace('half_m128k128','relaxed_m128k128').replace('half_m256k128','relaxed_m256k128')
writer=candidate.replace('relaxed_m','writer_m')
pattern=r'\*reinterpret_cast<device float4\*>\(output\+indices\[row\]\*p.oStride\+col\)=float4\([^;]*\);'
assert len(re.findall(pattern,writer))==6
writer=re.sub(pattern,'for(ushort j=0;j<4;++j)atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(output)+indices[row]*p.oStride+col+j,1,memory_order_relaxed);',writer)
relaxed_guard=guard.replace('false,true,false,','false,true,true,').replace('packed_affine_layout_guard','relaxed_layout_guard')
fault=relaxed_guard.replace('relaxed_layout_guard','fault_layout_guard').replace('left.get_capacity()!=8','left.get_capacity()!=9')
probe=r'''
template<class T>METAL_FUNC void descriptor_count(thread const T& value,device atomic_uint* out,device atomic_uint* invalid,uint width,uint height){
 for(ushort i=0;i<value.get_capacity();++i)if(value.is_valid_element(i)){auto xy=value.get_multidimensional_index(i);if(xy[0]<0||xy[1]<0||uint(xy[0])>=width||uint(xy[1])>=height)atomic_fetch_add_explicit(invalid,1,memory_order_relaxed);else atomic_fetch_add_explicit(out+uint(xy[1])*width+uint(xy[0]),1,memory_order_relaxed);}
}
kernel void descriptor_coordinates(device atomic_uint* output [[buffer(0)]]){
 constexpr auto d=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<d,execution_simdgroup> op;
 auto a=op.get_left_input_cooperative_tensor<half,half,float>();auto b=op.get_right_input_cooperative_tensor<half,half,float>();
 auto c=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 descriptor_count(a,output,output+1280,16,16);descriptor_count(b,output+256,output+1280,16,32);descriptor_count(c,output+768,output+1280,32,16);
}
'''
combined='#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n'+raw('affine_int4_profile_shader.h')+packed+guard+candidate+writer+relaxed_guard+fault+probe
path=root/'combined1.metal';assert not path.exists();path.write_text(combined)
captures=root.parent/'mistral-gpu-shader-profile1'
frozen=json.loads((captures/'provenance1.json').read_text())['files_sha256'];assets={}
for folder in ['capture-up2','capture-down1','capture-qo1']:
    for name in ['input-parent.f32','reference-parent.f32','weight.u32','scales.f16','biases.f16']:
        p=captures/folder/name;rel=str(p.relative_to(captures));assert sha(p)==frozen[rel],rel;assets[str(p)]=sha(p)
prior=root.parent/'mistral-prefill-relaxed-operator3/audit1.json'
old=json.loads(prior.read_text());assert old['passed'] and old['all_words_bit_identical']
binding=dict(scope='Distinct packed M128/M256 stageK128 half-input screen. Earlier reuse64 relaxed result is negative and is not rerun. No native runtime, model, API or serving change.',
    foundation_provenance_sha256=sha(base/'provenance7.json'),shader_sha256=sha(path),source_headers_sha256={n:sha(source/n) for n in ['packed_affine_shader.h','packed_affine_layout_guard.h','affine_int4_profile_shader.h']},
    captures=str(captures),capture_provenance_sha256=sha(captures/'provenance1.json'),assets_sha256=assets,
    earlier_reuse64_relaxed_audit_sha256=sha(prior),earlier_reuse64_words_exact=old['compared_words'],
    variants=['preserved_strict','relaxed_descriptor_only'],writer_probe=True,cooperative_coordinate_probe=True,numerical_gate='Exact full native control output bits plus 2e-6+4e-6*sum_abs FP64 samples; no tolerance adjustment',
    correctness_before_timing=True,native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
(root/'bindings1.json').write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps(dict(prepared=True,shader_sha256=binding['shader_sha256'],frozen_fixture_files=len(assets))))
