"""Generate isolated strict K16/vector-load/K32 scheduling factors."""
from pathlib import Path
import hashlib,json,re,shutil,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
assert ROOT.parent.name=='mistral7b-local-coder' and not ROOT.exists()
ROOT.mkdir()
for name in ['screen1.mm', 'run_correctness1.py', 'audit_correctness1.py', 'run_timing1.py', 'audit_timing1.py', 'prepare1.py']:shutil.copyfile(PACKAGE/name,ROOT/name)
M=ROOT.parent
N=M/'mistral-fused-mlp-native1';T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(N/'provenance1.json')=='b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe'
assert sha(T/'provenance1.json')=='191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a'
shader=N/'source2/src/backends/metal/packed_affine_shader.h'
assert sha(shader)==read(N/'source-bindings2.json')['files_sha256'][str(shader.relative_to(N/'source2'))]
original=shader.read_text().split('R"METAL(\n',1)[1].split(')METAL";',1)[0]
combined=(T/'combined1.metal').read_text();assert original in combined
combined=combined[:combined.index('constant bool fused_shadow')]
generated=[];variants={}
def right_load(target,local,vector):
    if not vector:return f'''   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){{
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i){target}[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+{local}+fragmentColumn+i%4];
   }}'''
    return f'''   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){{
    const uint at=((simd%2)*32+part*16+fragmentRow)*weightStride+{local}+fragmentColumn;
    const half4 lo=*reinterpret_cast<threadgroup const half4*>(weights+at);
    const half4 hi=*reinterpret_cast<threadgroup const half4*>(weights+at+8*weightStride);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){{{target}[part*8+j]=lo[j];{target}[part*8+4+j]=hi[j];}}
   }}'''
def left_load(target,local,offset):
    return f'''   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){{
    const uint row=firstRow+{offset}+fragmentRow+fragmentPart*8,k=base+{local}+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){target}[fragmentPart*4+j]=value[j];
   }}'''
for mode in range(4):
    for tile in [128,256]:
        name=f'half_m{tile}k128'
        start=original.index('kernel void '+name)
        end=original.find('kernel void ',start+1)
        body=original[start:end if end!=-1 else len(original)].rstrip()
        newname=f'fragment_mode{mode}_m{tile}'
        body=body.replace(name,newname,1)
        needle='uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]])'
        assert body.count(needle)==1
        body=body.replace(needle,'device atomic_uint* writers [[buffer(7)]],'+needle)
        begin=body.index('  for(uint local=0;local<stageK;local+=16) {')
        finish=body.index('  threadgroup_barrier(mem_flags::mem_threadgroup);',begin)
        if mode==1:
            inner=body[begin:finish]
            pattern=r'   #pragma clang loop unroll\(full\)\n   for\(uint part=0;part<2;\+\+part\) \{\n    #pragma clang loop unroll\(full\)\n    for\(uint i=0;i<8;\+\+i\)right\[part\*8\+i\]=weights\[.*?;\n   \}'
            inner,n=re.subn(pattern,right_load('right','local',True),inner);assert n==1,n
            body=body[:begin]+inner+body[finish:]
        elif mode>=2:
            declarations=''' auto rightNext=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto leftNext=multiply.get_left_input_cooperative_tensor<half,half,float>();
'''
            pos=body.index(' for(ushort i=0;');body=body[:pos]+declarations+body[pos:]
            begin=body.index('  for(uint local=0;local<stageK;local+=16) {')
            finish=body.index('  threadgroup_barrier(mem_flags::mem_threadgroup);',begin)
            inner='  for(uint local=0;local<stageK;local+=32) {\n'+right_load('right','local',mode==3)+'\n'+right_load('rightNext','(local+16)',mode==3)+'\n'
            for part in range(tile//64):
                inner+=left_load('left','local',part*16)+'\n'+left_load('leftNext','(local+16)',part*16)+'\n'
                inner+=f'   multiply.run(left,right,c{part});\n   multiply.run(leftNext,rightNext,c{part});\n'
            inner+='  }\n';body=body[:begin]+inner+body[finish:]
        # Observe complete write multiplicity without changing the timed PSO.
        pattern=r'(    \*reinterpret_cast<device float4\*>\(output\+indices\[row\]\*p.oStride\+col\)=float4\([^\n]+;\n)'
        replacement=r'\1    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);\n'
        body,n=re.subn(pattern,replacement,body);assert n==tile//64,(tile,n)
        generated.append(body);variants[newname]=dict(mode=mode,tile=tile,vector_right_load=bool(mode&1),k32_schedule=mode>=2)
text=combined+'\nconstant bool fragment_writers [[function_constant(22)]];\n'+'\n'.join(generated)+'\n'
dest=ROOT/'combined1.metal';assert not dest.exists();dest.write_text(text)
assert sha(dest)=='52d0021f1c098bcc3ce6a8210da0efa6e7d64599fb4adb88c256cc56508de38c', 'Prepared shader differs from qualified source'
mlx= M.parent/'qwen-local-coder/comparators/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels'
refs={str(p):sha(p) for p in [mlx/'quantized_nax.h',mlx/'steel/gemm/nax.h']}
plan=dict(scope='Kernel-only trained-data strict fragment-loading/scheduling screen. No runtime, model, precision, production policy or serving change.',
    prior_goal_turn='progress: cd5bb5f12 and501f0c905 validated native fusion/cache/pressure/coding and new serial serving results; all2x/half-TTFT targets remain unmet.',
    variants=variants,source_header_sha256=sha(shader),combined_sha256=sha(dest),primary_reference_files_sha256=refs,
    reference_observation='Pinned MLX quantized_nax qmm_t_nax_tgp_impl loads SK32 A/B tiles, then steel tile_matmad_nax performs K16 updates per accumulator. Native original schedules K16 across all M fragments before the next K16.',
    prior_distinct='Earlier descriptor-only, tile-size, branch-free fulltile and compiler loop-hint screens did not implement this explicit K32 operand preload/per-accumulator schedule or explicit vector threadgroup RHS loads.',
    numerical_contract='Same half operands, half coefficient decode, strict M16/N32/K16 MPP operations, and increasing K16 updates for each FP32 accumulator. Exact trained capture equality, guards and one-writer maps required; fixed FP64 bound unchanged.',
    baseline_kernel_selection_proven=False,comparator_inference_rerun=False,goal_complete=False)
(ROOT/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(prepared=True,kernels=len(variants),combined_sha256=sha(dest))))
