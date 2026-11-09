"""Prepare exact cached half weights with the existing strict MPP consumer."""
from pathlib import Path
import hashlib,json,re,shutil,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
assert ROOT.parent.name=='mistral7b-local-coder' and not ROOT.exists()
ROOT.mkdir()
for name in ['screen1.mm', 'derive_host1.py', 'host-derivation1.json', 'normal_prepare1.mm', 'normal-prepare-derivation1.json', 'run_correctness1.py', 'run_normal_prepare1.py', 'run_timing1.py', 'prepare_pressure1.py', 'run_pressure1.py', 'audit_timing1.py', 'audit_correctness1.py', 'audit_normal_prepare1.py', 'prepare1.py']:shutil.copyfile(PACKAGE/name,ROOT/name)
M=ROOT.parent
N=M/'mistral-fused-mlp-native1';T=M/'mistral-fused-mlp-trained1';F=M/'mistral-packed-fragment-schedule1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(N/'provenance1.json')=='b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'
assert sha(F/'provenance1.json')=='35165e34967e1a558a8e424f1ec17e3f5722b9352355b3f2d76c3c0bc542808d'
source=(F/'combined1.metal').read_text();prefix=source[:source.index('constant bool fragment_writers')]
functions=[]
for mode in range(4):
    for tile in [128,256]:
        name=f'fragment_mode{3 if mode==3 else 0}_m{tile}'
        start=source.index('kernel void '+name);end=source.find('kernel void ',start+1)
        body=source[start:end if end!=-1 else len(source)].strip().replace(name,f'cached_mode{mode}_m{tile}',1)
        if mode:
            body=body.replace('device atomic_uint* writers [[buffer(7)]],','device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],',1)
            begin=body.index('  const uint offset=n*');finish=body.index('  threadgroup_barrier(mem_flags::mem_threadgroup);',begin)
            addr='n*p.inner+base+localK' if mode==1 else '((group.x*(p.inner/128)+base/128)*64+localN)*136+localK'
            load=f'''  const uint cacheAt={addr};
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){{
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }}
'''
            body=body[:begin]+load+body[finish:]
        functions.append(body)
prep='''
constant bool prepared_writers [[function_constant(23)]];
kernel void prepare_half_weights(device const uint* packed [[buffer(0)]],device const ushort* scales [[buffer(1)]],
 device const ushort* biases [[buffer(2)]],device half* output [[buffer(3)]],constant uint4& dims [[buffer(4)]],
 device atomic_uint* writers [[buffer(5)]],uint gid [[thread_position_in_grid]]) {
 uint count=dims.x*dims.y;if(dims.z)count=count/128*136;
 const uint first=gid*4;if(first>=count)return;
 for(uint j=0;j<4;++j){uint at=first+j,n,k;bool padding=false;
  if(dims.z){uint block=at/(64*136),local=at%(64*136);n=(block/(dims.x/128))*64+local/136;uint kk=local%136; k=(block%(dims.x/128))*128+kk;padding=kk>=128;}
  else{n=at/dims.x;k=at%dims.x;}
  half value=half(0);
  if(!padding){uint word=packed[n*(dims.x/8)+k/8],code=(word>>((k%8)*4))&15u,coefficient=n*(dims.x/64)+k/64;
   float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);value=half(fma(float(code),scale,bias));}
  output[at]=value;if(prepared_writers)atomic_fetch_add_explicit(writers+at,1u,memory_order_relaxed);
 }
}
'''
out=ROOT/'combined1.metal';assert not out.exists();out.write_text(prefix+'\nconstant bool fragment_writers [[function_constant(22)]];\n'+prep+'\n'+'\n'.join(functions)+'\n')
plan=dict(scope='Kernel-only prepared half-weight screen. Same strict native MPP arithmetic and trained fixtures; predecode once per weight, outside hot projection loop.',previous_goal_turn='progress: bf640c64f closed exact K32/vector screen without runtime replacement.',
 modes={'-1':'unaltered native inline Q4','0':'copied inline Q4 with disabled writer observer','1':'prepared row-major half weights, original K16 consumer','2':'prepared N64/K128/padded136 half tiles, original K16 consumer','3':'prepared padded half tiles with previously qualified K32/vector consumer'},
 numerical_contract='Same Q4/group64 values, FP16 coefficients, F32 fused dequantization then F16 RNE, half matrix operands, strict M16/N32/K16 MPP accumulation and FP32 output. All operator outputs must equal the original native captures exactly.',
 memory_contract='Row cache=N*K*2 bytes; tiled cache=N*K*2*17/16 bytes, plus bounded guards. Prototype keeps both only to compare; runtime admission must choose a layout and explicit total byte budget, including existing engine allocations and256MiB margin.',
 prior_distinct='Earlier prefill-fp16-cache1 used cached/streamed MPS products on synthetic inputs; this uses trained inputs and the existing strict manual MPP consumer. Public residency-set experiments only pin existing buffers and do not predecode weights.',
 combined_sha256=sha(out),base_fragment_provenance_sha256=sha(F/'provenance1.json'),native_provenance_sha256=sha(N/'provenance1.json'),trained_provenance_sha256=sha(T/'provenance1.json'),
 no_runtime_integration=True,comparator_inference_rerun=False,goal_complete=False)
(ROOT/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(prepared=True,kernels=9,combined_sha256=sha(out))))

assert sha(out)=='49ef7a4d49965ac4a980440cc35797d49ca7b3da62b3c7a00fceea350c6b712a', 'Prepared shader differs from qualified source'
assert sha(ROOT/'screen1.mm')=='1c5734757b3e233b13bf48a15e3a5b68fcbf809bffc919f6dd6762dc520ed53b', 'Prepared host differs from qualified source'
normal=read(ROOT/'normal-prepare-derivation1.json');derived=(ROOT/'screen1.mm').read_text()
for change in normal['changes']:
    assert derived.count(change['old'])==1;derived=derived.replace(change['old'],change['new'])
assert derived==(ROOT/'normal_prepare1.mm').read_text()
