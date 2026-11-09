"""Prepare direct tensor row/threadgroup alternatives over the unchanged learned Q4 fixtures."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;P=R.parent/'mistral-tensor-view-prefill-screen1';Q=R.parent/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(P/'source-audit2.json')['passed'] and read(P/'replay-audit2.json')['passed']
base=(P/'shared3.metal').read_text();variants=[]
# Descriptor M/N, threadgroup row tile, threads, shared columns.
specs=[('row512n64',64,64,512,256),('row512n32',64,32,512,256),('row256n64t128',64,64,256,128),('row256n32t128',64,32,256,128)]
for name,dm,dn,rm,threads in specs:
 s=base.replace('shared_tensor_m256',name).replace('decoded[64*STRIDE]',f'decoded[{dn}*STRIDE]')
 s=s.replace('block.y*256+simd*32,column=block.x*64',f'block.y*{rm}+simd*{dm},column=block.x*{dn}')
 s=s.replace(' const uint localN=tid/4,localK=(tid%4)*PACKET,n=column+localN;',' const uint localK=(tid%4)*PACKET;')
 s=s.replace('extents<int,BK,64>',f'extents<int,BK,{dn}>').replace('matmul2d_descriptor(32,64,128',f'matmul2d_descriptor({dm},{dn},128')
 s=s.replace('  const uint offset=n*','  for(uint localN=tid/4;localN<'+str(dn)+';localN+='+str(threads//4)+') {\n  const uint n=column+localN;\n  const uint offset=n*')
 s=s.replace('  threadgroup_barrier(mem_flags::mem_threadgroup);\n  auto left=', '  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  auto left=',1)
 s=s.replace('a.slice<128,32>',f'a.slice<128,{dm}>').replace('d.slice<64,32>',f'd.slice<{dn},{dm}>')
 variants.append(s)
(R/'row_tiles1.metal').write_text('\n'.join(variants))
native=(Q/'source1/src/backends/metal/packed_affine_shader.h').read_text().split('R"METAL(\n',1)[1].split(')METAL"',1)[0]
prior=(P/'combined3.metal').read_text();assert native in prior
# Preserve the already qualified control library byte-for-byte; only add candidates.
(R/'combined1.metal').write_text(prior+'\n'+'\n'.join(variants))
plan=dict(scope='Six bounded learned-input operator variants,512 actual dense rows. No runtime/model/HTTP admission.',base_operator_root=str(P),base_provenance_sha256=sha(P/'provenance1.json'),base_native_root=str(Q),trained_assets=str(R.parent/'mistral-fused-mlp-trained1/assets1'),trained_captures=str(R.parent/'mistral-fused-mlp-trained1/captures1'),trained_provenance_sha256=sha(R.parent/'mistral-fused-mlp-trained1/provenance1.json'),control_source_sha256=hashlib.sha256(native.encode()).hexdigest(),shader_sha256=sha(R/'combined1.metal'),variant_sha256=sha(R/'row_tiles1.metal'),shapes=['gate4096x14336','down14336x4096'],rows=512,methods=[dict(name='inline',function='half_m256k128',rows_per_group=256,columns_per_group=64,threads=256,shared_bytes=17408),dict(name='tensor256',function='shared_tensor_m256',rows_per_group=256,columns_per_group=64,threads=256,shared_bytes=17408)]+[dict(name=n,function=n,rows_per_group=rm,columns_per_group=dn,threads=t,shared_bytes=dn*136*2) for n,dm,dn,rm,t in specs],no_weight_expansion_for_any_executed_method=True,no_extra_scatter=True,requires_exact_full_parent_and_unchanged_fp64_bound=True,no_comparator_execution=True,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n')
# Derive the host from the previously qualified pilot, removing unexecuted expansion methods/resources.
s=(P/'pilot3.mm').read_text()
s=s.replace('auto gather=pipe(@"packed_affine_gather"),scatter=pipe(@"tensor_scatter");auto pc=[MTLFunctionConstantValues new];bool no=false;[pc setConstantValue:&no type:MTLDataTypeBool atIndex:23];auto prepare=pipe(@"prepare_half_weights",pc);','auto gather=pipe(@"packed_affine_gather");bool no=false;')
a=s.index(' const std::array<std::string,6> names');b=s.index('\n for(bool down:',a)
s=s[:a]+' const std::array<std::string,6> names{{"inline","tensor256","row512n64","row512n32","row256n64t128","row256n32t128"}};\n const std::array<size_t,6> tileM{{256,256,512,512,256,256}},tileN{{64,64,64,32,64,32}},threads{{256,256,256,256,128,128}};'+s[b:]
s=s.replace('prepared=buffer((n*k+64)*2),out=buffer((M*n+32)*4),temporary=buffer((M*n+32)*4),writers=buffer(128)','out=buffer((M*n+32)*4)')
a=s.index('  std::fill_n((uint16_t*)prepared.contents');b=s.index('  auto constants=',a)
s=s[:a]+'''  const auto wSha=Sha(w.contents,w.length),sSha=Sha(sc.contents,sc.length),bSha=Sha(bi.contents,bi.length);
  std::vector<uint16_t> expectedCompact(M*k+64,H);for(size_t at=0;at<M*k;++at)expectedCompact[32+at]=Garnet::Float32ToFloat16RNE(input[at]);
'''+s[b:]
a=s.index('  std::array<id<MTLComputePipelineState>,6> pipelines');b=s.index('  std::vector<uint32_t> indices',a)
s=s[:a]+'''  std::array<id<MTLComputePipelineState>,6> pipelines{{pipe(@"half_m256k128",constants),pipe(@"shared_tensor_m256",constants),pipe(@"row512n64",constants),pipe(@"row512n32",constants),pipe(@"row256n64t128",constants),pipe(@"row256n32t128",constants)}};
  for(size_t mode=0;mode<6;++mode){auto p=pipelines[mode];Need(p.maxTotalThreadsPerThreadgroup>=threads[mode],"Insufficient prepared threads");Need(p.staticThreadgroupMemoryLength==tileN[mode]*136*2,"Unexpected static shared bytes");report["pipeline_resources"].push_back({{"shape",shape},{"mode",names[mode]},{"threads",threads[mode]},{"maximum_threads",p.maxTotalThreadsPerThreadgroup},{"shared_bytes",p.staticThreadgroupMemoryLength},{"simd_width",p.threadExecutionWidth}});}
'''+s[b:]
s=s.replace('std::fill_n((uint32_t*)temporary.contents,M*n+32,G);','')
s=s.replace('   cb=[queue commandBuffer];e=[cb computeCommandEncoder];','   auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];')
a=s.index('   if(mode<2||mode==5){');b=s.index('   finish(cb);',a)
s=s[:a]+'''   [e setBuffer:w offset:0 atIndex:1];[e setBuffer:sc offset:0 atIndex:2];[e setBuffer:bi offset:0 atIndex:3];[e setBuffer:out offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];[e setBytes:indices.data() length:indices.size()*4 atIndex:6];
   [e dispatchThreadgroups:MTLSizeMake(n/tileN[mode],M/tileM[mode],1) threadsPerThreadgroup:MTLSizeMake(threads[mode],1,1)];[e endEncoding];
'''+s[b:]
s=s.replace('Need(((uint32_t*)temporary.contents)[i]==G&&((uint32_t*)temporary.contents)[16+M*n+i]==G,"Temporary parent guard changed");','')
s=s.replace('Need(Sha(prepared.contents,prepared.length)==preparedSha,"Read-only prepared weights changed");','Need(Sha(w.contents,w.length)==wSha&&Sha(sc.contents,sc.length)==sSha&&Sha(bi.contents,bi.length)==bSha,"Read-only quantized weights changed");')
s=s.replace('Need(mode!=0||exact==M*n,"Unchanged native control differs from trained capture");','Need(exact==M*n,"Candidate differs from unchanged native trained capture");')
s=s.replace('all_input_and_prepared_weight_bits_unchanged','all_input_and_quantized_weight_bits_unchanged')
s=s.replace('  }\n }}','  }\n  Save(dest/(shape+"-compact.u16"),compact.contents,compact.length);\n }}')
assert 'prepared.contents' not in s and 'temporary.contents' not in s and 'tensor_scatter' not in s
(R/'pilot1.mm').write_text(s)
print('Prepared six bounded methods without any expanded-weight allocation.')
