"""Prepare exact-current fused controls and direct-tensor whole-MLP learned-input replay."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=R.parent;P=M/'mistral-tensor-view-prefill-screen1';Q=M/'mistral-causal-batch4-helper1';T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=Q/'source1/src/backends/metal';native=(q/'fused_mistral_mlp_shader.h').read_text().split('R"METAL(\n',1)[1].split(')METAL"',1)[0];base=(P/'combined3.metal').read_text()
assert 'fused_m256k128' not in base
(R/'combined1.metal').write_text(base+'\n'+native+'\n'+(R/'fused_tensor1.metal').read_text())
base_host=(T/'replay1.mm').read_text();s=base_host
s=s.replace('std::array<id<MTLComputePipelineState>,3> fused;', 'auto downTensor=pipe(safe,@"shared_tensor_m256",constants(N,K));std::array<std::array<id<MTLComputePipelineState>,3>,2> fused;')
s=s.replace('for(size_t v=0;v<3;v++){fused[v]=pipe(fast,@"fused_m256k128",constants(K,N,v==0,v==2));Need(fused[v].staticThreadgroupMemoryLength==17408&&fused[v].maxTotalThreadsPerThreadgroup>=256,', 'for(size_t method=0;method<2;method++)for(size_t v=0;v<3;v++){fused[method][v]=pipe(fast,method?@"fused_tensor_m256":@"fused_m256k128",constants(K,N,v==0,v==2));Need(fused[method][v].staticThreadgroupMemoryLength==17408&&fused[method][v].maxTotalThreadsPerThreadgroup>=256,')
s=s.replace('   for(size_t v=0;v<3;v++){\n    reset();', '   for(size_t method=0;method<4;method++)for(size_t v=0;v<3;v++){\n    reset();')
s=s.replace('projection(cb,0,fused[v],true);projection(cb,2,downPipe,false);','projection(cb,0,fused[method>=2][v],true);projection(cb,2,(method==1||method==2)?downTensor:downPipe,false);')
s=s.replace('J item={{"variant",v}', 'J item={{"method",method},{"variant",v}')
s=s.replace('prefix+"-fused-shadow-"','prefix+"-method-"+std::to_string(method)+"-fused-shadow-"')
s=s.replace('std::string name=prefix+"-fused-"','std::string name=prefix+"-method-"+std::to_string(method)+"-fused-"')
s=s.replace('   auto guards=[&](){Need(', '   auto guards=[&](){for(size_t stream=0;stream<3;stream++)Need(!std::memcmp(weights[stream].contents,hostWeights[stream].data(),hostWeights[stream].size()*4)&&!std::memcmp(scales[stream].contents,hostScales[stream].data(),hostScales[stream].size()*2)&&!std::memcmp(biases[stream].contents,hostBiases[stream].data(),hostBiases[stream].size()*2),"Read-only Q4 weight parents changed");Need(')
s=s.replace('"Wrong full trained writer map"','"Wrong full trained writer map"')
assert 'fused[method>=2][v]' in s and 'method<4' in s and 'fused[v]' not in s
(R/'pilot1.mm').write_text(s)
plan=dict(scope='Whole MLP, actual learned layers0/15/31 and512 dense identity rows; operator qualification only.',base_native=str(Q),base_native_provenance_sha256=sha(Q/'provenance1.json'),trained_root=str(T),trained_provenance_sha256=sha(T/'provenance1.json'),current_fused_control_sha256=hashlib.sha256(native.encode()).hexdigest(),shader_sha256=sha(R/'combined1.metal'),new_shader_sha256=sha(R/'fused_tensor1.metal'),host_sha256=sha(R/'pilot1.mm'),ordinary_shader=str(T/'ordinary1.metal'),methods={'0':'Existing fused gate/up and existing down','1':'Existing fused gate/up and qualified tensor down','2':'New tensor fused gate/up and qualified tensor down','3':'New tensor fused gate/up and existing down'},variants=['full FP32 shadows','no intermediates','full GPU writer map'],fast_gate_up_safe_down=True,half_output_and_down_must_match_every_bit=True,no_xmodel_operator_changes=True,no_comparator_execution=True,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared four whole-MLP methods and three correctness variants across three learned layers.')
