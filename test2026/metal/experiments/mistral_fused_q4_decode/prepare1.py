"""Pin native controls/settings and prepare the distinct FP32 fused decode screen."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent;M=R.parent;J=M/'mistral-prefill-completion-wide1';T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not(R/'compile1-status.json').exists()
assert sha(J/'provenance1.json')=='3dd125f8f9ed9bef0a944cdb14eca7ff0f7af3b0bc19f818fc884763d99a9411'
assert sha(T/'provenance1.json')=='191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a'
source=J/'source2/src/backends/metal';bindings=read(J/'source-bindings2.json')['files_sha256'];inputs={}
def shader(name):
    p=source/name;assert sha(p)==bindings['src/backends/metal/'+name];inputs[str(p)]=sha(p)
    return re.search(r'R"METAL\((.*?)\)METAL"',p.read_text(),re.S).group(1)
prefix='#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n'+shader('affine_int4_profile_shader.h')
scalar=shader('affine_int4_shader.h');scalar=scalar[:scalar.index('kernel void affine_q4_gemm')]
indexed=shader('affine_int4_indexed_decode_shader.h');blocked=shader('affine_int4_blocked_decode_shader.h')
controls=prefix+indexed+'\n#define GARNET_BLOCKED_COLUMNS 4\n#define GARNET_BLOCKED_ROWS selected_decode_rows\n#define GARNET_BLOCKED_INDEXED 1\n'+blocked
(R/'ordinary1.metal').write_text(prefix+scalar)
fused=(R/'fused1.inc').read_text();(R/'decode1.metal').write_text(controls+'\n'+fused+'\n')
p=source/'metal_context.mm';assert sha(p)==bindings['src/backends/metal/metal_context.mm'];inputs[str(p)]=sha(p)
s=p.read_text();start=s.index('kernel void silu_mul(');end=s.index('struct JoinedGlobalSiluParams',start)
activation='#include <metal_stdlib>\nusing namespace metal;\nstruct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};\n'+s[start:end]
activation+='kernel void decode_pressure(device uint* data [[buffer(0)]],uint i [[thread_position_in_grid]]){data[i]=data[i]*1664525u+1013904223u;}\n'
(R/'activation1.metal').write_text(activation)
plan=dict(scope='FP32 Q4 gate/up/SiLU decode fusion only; down projection remains outside the measured span.',base_engine_provenance_sha256=sha(J/'provenance1.json'),trained_provenance_sha256=sha(T/'provenance1.json'),native_source_sha256=inputs,control='Native strict MSL3.1 fastMathEnabled=NO scalar for occupancy1/2/3 and blocked4 only for occupancy4 K4096/N14336; original silu_mul uses nil/default compiler options.',candidates=['strict-dot/exp','strict-dot/fast::exp','nil-default/exp'],precision='Original Q4 words/FP16 coefficients, FP32 input/dequantization/FMA/accumulation/product; no new quantization or half activation.',layers=[0,15,31],source_rows=[0,127,256,511],columns=[14336,1,7,17,33],layouts=['contiguous','padded'],masks='ordinary1/2/3/4 plus all16 physical-row masks including zero',gates=['Complete FP32 product parents exact to original gate/up plus original SiLU; complete gate/up shadow parents exact.','No input/weight/guard writes. Independent FP64 gate/up checks at original2e-6+4e-6*sum_abs bound.','All pipeline profiles prepared before first sample; shadow and normal outputs agree.','Only candidates exact across all cells can enter four fresh serial balanced warm/pressure timing processes.','At least3% lower median GPU time in every process/occupancy and no individual regression over2%, before considering native integration.'],no_comparator_execution=True,no_native_runtime_integration=True,serving_speed_claim=False,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n')
prepared=dict(passed=True,inputs_sha256=inputs,generated_sha256={n:sha(R/n) for n in ['fused1.inc','decode1.metal','ordinary1.metal','activation1.metal','prepare1.py']},frozen_asset_binding_sha256=sha(T/'replay-bindings1.json'))
(R/'preparation1.json').write_text(json.dumps(prepared,indent=2)+'\n');print(json.dumps(prepared,indent=2))
