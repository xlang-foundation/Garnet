"""Prepare an isolated gate/up MMA + SiLU + half-output correctness screen."""
from pathlib import Path
import hashlib,json,re,shutil,sys
import numpy as np
from safetensors import safe_open

PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True,exist_ok=False)
for name in ['prepare1.py','host1.mm','host3.mm','run_correctness1.py','run_timing2.py','compile1.py','timing1-assessment.json']:
    shutil.copyfile(PACKAGE/name,ROOT/name)
M=ROOT.parent
BASE=M/'mistral-half-attention-native1'
SOURCE=BASE/'source7/src/backends/metal'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(BASE/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
headers=['affine_int4_profile_shader.h','packed_affine_shader.h','paired_prefill_shader.h','packed_affine_layout_guard.h']
def metal(name):return re.search(r'R"METAL\((.*?)\)METAL"',(SOURCE/name).read_text(),re.S).group(1)
paired=metal('paired_prefill_shader.h')
fused=paired.replace('kernel void pair_m','kernel void fused_m')
needle='device float* outputUp [[buffer(11)]],'
assert fused.count(needle)==2
fused=fused.replace(needle,needle+'device half* outputHalf [[buffer(13)]],device atomic_uint* writers [[buffer(14)]],')
pattern=r'\*reinterpret_cast<device float4\*>\(output\+indices\[row\]\*p.oStride\+col\)=float4\((gate[01])\[ci\],[^;]+\);'
def gate_store(match):
    name=match.group(1);up='up'+name[-1]
    return 'if(fused_shadow){'+match.group(0)+'}\n    for(uint j=0;j<4;++j){float g='+name+'[ci+j];float value=(g/(1.0f+exp(-g)))*'+up+'[ci+j];outputHalf[row*p.columns+col+j]=half(value);if(fused_writers)atomic_fetch_add_explicit(writers+row*p.columns+col+j,1u,memory_order_relaxed);}'
assert len(re.findall(pattern,fused))==3
fused=re.sub(pattern,gate_store,fused)
pattern=r'\*reinterpret_cast<device float4\*>\(outputUp\+indices\[row\]\*p.oStride\+col\)=float4\([^;]+\);'
assert len(re.findall(pattern,fused))==3
fused=re.sub(pattern,lambda m:'if(fused_shadow){'+m.group(0)+'}',fused)
fused='constant bool fused_shadow [[function_constant(20)]];\nconstant bool fused_writers [[function_constant(21)]];\n'+fused
combined='#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n'+metal(headers[0])+metal(headers[1])+metal(headers[2])+metal(headers[3])+fused
dest=ROOT/'combined1.metal';assert not dest.exists();dest.write_text(combined)
native=(SOURCE/'metal_context.mm').read_text()
params=re.search(r'struct Params \{ ulong rows,columns,aStride,bStride,cStride;float epsilon,offset; \};',native).group(0)
start=native.index('kernel void silu_mul(');end=native.index('struct JoinedGlobalSiluParams',start)
ordinary='#include <metal_stdlib>\nusing namespace metal;\n'+params+'\n'+native[start:end]
(ROOT/'ordinary1.metal').write_text(ordinary)
model=Path(json.loads((BASE/'model-bindings7.json').read_text())['model'])
model_hash=sha(model/'model.safetensors');assert model_hash=='d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
assets=ROOT/'assets1';assets.mkdir();n,k=14336,4096
capture=M/'mistral-gpu-shader-profile1/capture-up2/input-parent.f32'
provenance=json.loads((capture.parent.parent/'provenance1.json').read_text())['files_sha256']
assert sha(capture)==provenance['capture-up2/input-parent.f32']
input_data=np.fromfile(capture,dtype='<f4').reshape(514,k)
input_data.tofile(assets/'input-parent.f32')
fixtures=[]
with safe_open(model/'model.safetensors',framework='np') as handle:
    for layer in [0,15,31]:
        folder=assets/('layer-'+str(layer));folder.mkdir()
        for projection in ['gate','up']:
            for name,suffix,dtype in [('weight','u32','<u4'),('scales','f16','<f2'),('biases','f16','<f2')]:
                key=f'model.layers.{layer}.mlp.{projection}_proj.{name}'
                a=handle.get_tensor(key)
                assert a.dtype==np.dtype(dtype) and a.shape==(n,k//(8 if name=='weight' else 64)),(key,a.dtype,a.shape)
                a.tofile(folder/(projection+'-'+name+'.'+suffix))
        fixtures.append(dict(label=folder.name,trained_weights=True,input_scope='Frozen synthetic finite affine input; not a captured trained MLP activation.'))
folder=assets/'synthetic';folder.mkdir()
scales=[[0x2c00,0x8001,0x0001,0xac00,0,0x8000,0x3000,0x3001],[0xa800,0x2801,0,1,0x8001,0x2c03,0xac04,0x2805]]
biases=[[0xb000,1,0x8001,0x2c00,0,0x8000,0x2c01],[0x2c02,0xac03,1,0x8001,0,0xb001,0x3001]]
for i,projection in enumerate(['gate','up']):
    codes=np.arange(n*k//8,dtype='<u4')*np.uint32(2246822519 if i else 2654435761)+np.uint32(927 if i else 131)
    codes.tofile(folder/(projection+'-weight.u32'))
    for name,values in [('scales',scales[i]),('biases',biases[i])]:
        np.resize(np.array(values,dtype='<u2'),n*k//64).tofile(folder/(projection+'-'+name+'.f16'))
fixtures.insert(0,dict(label='synthetic',trained_weights=False,input_scope='Independent gate/up Q4 codes and signed/subnormal/zero coefficients.'))
binding=dict(scope='Standalone paired affine gate/up MMA with fused SiLU and compact half down-input output. No runtime integration or serving speed claim.',
 foundation_provenance_sha256=sha(BASE/'provenance7.json'),source_headers_sha256={name:sha(SOURCE/name) for name in headers},native_context_sha256=sha(SOURCE/'metal_context.mm'),
 combined_sha256=sha(dest),ordinary_sha256=sha(ROOT/'ordinary1.metal'),model=str(model),model_sha256=model_hash,
 assets_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(assets.rglob('*')) if p.is_file()},fixtures=fixtures,
 rows=[127,128,129,255,256,257,383,511,512],physical_rows=514,input_stride=4104,output_stride=14344,
 separate_gathers=3,fused_gathers=1,main_control_math='Original silu_mul source, explicit MSL3.1 surrogate. Selected native build uses nil/default options; see real trained MLP follow-up.',
 projection_control_math='Original packed shader, MSL4 Safe, strict descriptors.',candidate_math_modes=['Safe','Fast'],
 numerical_gate='Complete gate/up FP32 bits and complete post-SiLU half bits must match original control; no relaxed tolerance. Independent projection FP64 samples use fixed2e-6+4e-6*sum_abs.',
 shadows_and_atomic_writers_before_timing=True,correctness_before_timing=True,quality_qualified=False,native_integration=False,goal_complete=False)
(ROOT/'bindings1.json').write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps(dict(prepared=True,fixtures=len(fixtures),assets=len(binding['assets_sha256']),shader_sha256=binding['combined_sha256'])))

profile_header=SOURCE/'silu_profile_shader.h'
(ROOT/'ordinary2.metal').write_text((ROOT/'ordinary1.metal').read_text()+metal('silu_profile_shader.h'))
timing_source=dict(scope='Corrected compact-extent fused block screen; no redundant dense output clear. Frozen original separate/paired projections and actual profile32 SiLU controls.',host_source_sha256=sha(ROOT/'host3.mm'),host='host3',ordinary_shader_sha256=sha(ROOT/'ordinary2.metal'),profile32_header_sha256=sha(profile_header),profile32_exact_source_appended=True,timing_physical_rows=[128,256,512],input_parent_guard_rows=2,input_base_offset=64+4104*4,selected_rows='reverse0..m-1; input base shifted byone guard row; exact same compact values as correctness controls.',dense_all_rows_selected=True,no_redundant_output_clear=True,prior_timing_excluded=True,no_serving_claim=True)
(ROOT/'timing-source3.json').write_text(json.dumps(timing_source,indent=2)+'\n')
