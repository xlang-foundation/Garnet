"""Recursive learned MTP tree versus original serial MTP at matched GEMM geometry.
Correctness-only; not target generation, model quality, latency or process VRAM.
"""
import hashlib,json,os,struct,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
from qwen35_mtp_stage_probe import MTPStageProbe
from xModel.qwen35.vl_0_8b import cuda_speculative as original
from xModel.qwen35.vl_0_8b.cuda_tree_mtp import CUDATreeMTPProgram
from qwen35_shared_mtp_experiment import SharedMTPProgram,SharedMTPSession

request_path,output_path=map(Path,sys.argv[1:3]);assert not output_path.exists(),'Preserve first recursive MTP result'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
evidence={'request':request,'events':[],'paths':[],'all_passed':False,'full_goal_met':False,
    'performance_or_process_memory_measurement':False,'target_generation_or_independent_quality_proven':False}
def save():output_path.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
def event(name):
    evidence['events'].append({'name':name,'unix_seconds':time.time(),'pid':os.getpid()});save();print('TREE_MTP_EVENT',name,flush=True)
def reject(call):
    try:call()
    except Exception as e:return str(e)
    raise AssertionError('Expected tree MTP rejection')
def words(data):return list(struct.unpack('<'+'f'*(len(data)//4),data))
def sha_tensor(io,value,b,width):
    flat=io.storage_view(value,[b,width]);out=hashlib.sha256()
    for start in range(0,width,131072):out.update(io.finite_f32_bytes(io.slice(flat,start,[b,min(131072,width-start)])))
    return out.hexdigest()
serial_probe=tree_probe=None;model=tree=reference=None;owner=object();claimed=False;saved_factory=original.CUDASpeculativeSession
try:
    case=request['case'];b=request['profile']['batch'];n=request['nodes']
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
    original.CUDASpeculativeSession=lambda *args:SharedMTPSession(*args,mode='pool')
    event('load_begin')
    try:model=Model(request['checkpoint'],request['profile'],request['model_id'])
    finally:original.CUDASpeculativeSession=saved_factory
    d=model.decoder;s=model.speculative;io=d.io;c=d.config;h=c['hidden_size'];kv=c['num_key_value_heads'];dim=c['head_dim']
    event('load_complete');s.reset()
    rows=[model._prepare(case['prompts'][r],case['images'][r]) for r in range(b)]
    hidden=list(io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
    length=max(len(r[2]) for r in rows);pos=[r[1]+[[0,0,0]]*(length-len(r[1])) for r in rows]
    current=s.prefill(hidden,pos,[len(r[2]) for r in rows],[r[3] for r in rows]);assert s.slots==d.slots
    evidence['real_prefix']={'target_slots':list(d.slots),'mtp_slots':list(s.slots),'current':list(current),'prompt_token_ids':[r[2] for r in rows]}
    event('shifted_real_image_prefill_complete')
    tree=CUDATreeMTPProgram(d,n)
    reference=SharedMTPProgram(d.checkpoint,c,b,d.capacity,n,d.projection_workspace_bytes,io)
    reference_workspace=io.zeros([max(1,(json.loads(reference.graph.stats_json())['context_workspace_required_bytes']+3)//4)])
    reference_output=io.zeros([b,n*h+2*s.cache_width])
    reference.graph.set_workspace_storage(reference_workspace);reference.graph.set_output_storage(reference_output)
    reference.graph.enable_cuda_graph(True)
    def canonical():return {'target':sha_tensor(io,d.state,b,d.programs[d.decode_chunk_size].state_width),
        'target_hidden':sha_tensor(io,d.last_hidden,b,h),'mtp_key':sha_tensor(io,s.key,b,s.cache_width),
        'mtp_value':sha_tensor(io,s.value,b,s.cache_width),'mtp_hidden':sha_tensor(io,s.last_hidden,b,h),
        'target_slots':list(d.slots),'mtp_slots':list(s.slots)}
    before=canonical();evidence['canonical_before']=before
    root_bytes=io.finite_f32_bytes(s.last_hidden);root=words(root_bytes)
    # Probe graph construction happens before claiming the canonical parent.
    serial_probe=MTPStageProbe(d,False);tree_probe=MTPStageProbe(d,True)
    d._claim_state_transaction(owner);claimed=True
    proposal=tree.draft(s,current,[1]*b,[16]*b,s.current_positions,model.stops,owner,request['beam'])
    full_features_bytes=io.finite_f32_bytes(tree.features);full_features=words(full_features_bytes)
    assert b==1 and n==32
    path=[1,3,7];assert [proposal['parents'][0][node] for node in path]==[0,1,3]
    evidence['proposal']=proposal;evidence['stage_path']=path
    target=io.zeros([1,n,h]);embedding=io.zeros([1,n,h]);position=io.zeros([1,n,3]);slot=io.zeros([1,n]);flags=io.zeros([1,n]);packed_hidden=io.zeros([1,n,h])
    # Independent serial oracle first derives its OWN level1/2 features.
    oracle={};serial_inputs=None;truth_serial=None
    for level in (1,2,3):
        tf=[0.]*(n*h);rf=[0.]*n;rs=[float(s.slots[0])]*n;rp=[0.]*(n*3);ids=[[0] for _ in range(n)]
        source_checks=[]
        for t,node in enumerate(path[:level]):
            previous=root[:h] if t==0 else oracle[path[t-1]]
            tf[node*h:(node+1)*h]=previous;rf[node]=1.;rs[node]=float(s.slots[0]+t)
            rp[node*3:node*3+3]=[float(s.current_positions[0]+t+1)]*3;ids[node][0]=proposal['tokens'][0][node]
            expected=full_features_bytes[proposal['parents'][0][node]*h*4:(proposal['parents'][0][node]+1)*h*4]
            source_checks.append(struct.pack('<'+'f'*h,*previous)==expected)
        io.update(target,tf);io.embedding_chunk(ids,embedding);io.update(flags,rf);io.update(slot,rs);io.update(position,rp)
        truth_result=reference.run([target,embedding,s.key,s.value,position,slot,flags]);io.slice_into(truth_result,0,packed_hidden)
        truth_serial=io.finite_f32_bytes(packed_hidden);v=words(truth_serial)
        for node in path[:level]:oracle[node]=v[node*h:(node+1)*h]
        if level==3:assert all(source_checks),'First difference parent inputs no longer exact'
    serial_inputs=[target,embedding,s.key,s.value,position,slot,flags]
    # Reconstruct the tree cache just before level3: dummyroot and finalized
    # depth1/2 features retained; later slots zero as in the actual draft.
    pre_features=[0.]*(n*h);pre_features[:h]=root[:h]
    for node in range(1,proposal['used_nodes'][0]):
        if proposal['depth'][0][node]<=2:pre_features[node*h:(node+1)*h]=full_features[node*h:(node+1)*h]
    io.update(tree.features,pre_features)
    io.embedding_chunk([[proposal['tokens'][0][j]] for j in range(n)],tree.embedding)
    stage_flags=[float(bool(proposal['mtp_active'][0][j]) and proposal['depth'][0][j]<=3) for j in range(n)]
    io.update(tree.active,stage_flags);io.update(tree.parents,proposal['parents'][0]);io.update(tree.prefix,[float(s.slots[0])])
    io.update(tree.position,[float(s.current_positions[0]+proposal['depth'][0][j]) for j in range(n) for axis in range(3)])
    pre_tree=io.clone(tree.features)
    tree_inputs=[pre_tree,tree.embedding,s.key,s.value,tree.position,tree.parents,tree.active,tree.prefix]
    tree.run(s.key,s.value,owner);truth_tree=io.finite_f32_bytes(tree.features)
    assert truth_tree[7*h*4:8*h*4]==full_features_bytes[7*h*4:8*h*4],'Original tree replay drifted before instrumentation'
    def compare_bytes(a,e):
        aa=words(a);ee=words(e);bad=[i for i in range(len(aa)) if a[4*i:4*i+4]!=e[4*i:4*i+4]]
        return {'words':len(aa),'different_words':len(bad),'max_abs':max((abs(aa[i]-ee[i]) for i in bad),default=0.),
            'actual_sha256':hashlib.sha256(a).hexdigest(),'reference_sha256':hashlib.sha256(e).hexdigest()}
    original_check=compare_bytes(truth_tree[7*h*4:8*h*4],truth_serial[7*h*4:8*h*4])
    assert original_check['different_words']==2021 and original_check['max_abs']==7.62939453125e-6,'Original strict failure must reproduce on new diagnostic runtime'
    evidence['original_uninstrumented_failure']=original_check
    probe_packets={};probe_arrays={}
    for name,probe,inputs in (('serial',serial_probe,serial_inputs),('tree',tree_probe,tree_inputs)):
        packed=probe.run(inputs);data=io.finite_f32_bytes(packed)
        assert len(data)==probe.output_width*4
        file=output_path.with_name(output_path.stem+'-'+name+'-all-stages.f32');file.write_bytes(data)
        probe_packets[name]=data;probe_arrays[name]=words(data)
    assert serial_probe.layout==tree_probe.layout
    layout=tree_probe.layout;evidence['stage_layout']=layout
    def part_bytes(name,node=None,mode='tree'):
        item=next(v for v in layout if v['name']==name);start=item['offset'];width=item['width']
        if node is not None:start+=node*item['width_per_node'];width=item['width_per_node']
        return probe_packets[mode][start*4:(start+width)*4]
    evidence['probe_perturbation_checks']=[]
    for node in path:
        evidence['probe_perturbation_checks'].append({'node':node,
            'tree_probe_vs_tree_truth':compare_bytes(part_bytes('retained',node),truth_tree[node*h*4:(node+1)*h*4]),
            'serial_probe_vs_serial_truth':compare_bytes(part_bytes('normalized',node,mode='serial'),truth_serial[node*h*4:(node+1)*h*4])})
    evidence['stage_comparisons']=[]
    for node in path:
        for item in layout:
            name=item['name']
            if 'width_per_node' not in item:continue
            actual=part_bytes(name,node);expected=part_bytes(name,node,mode='serial')
            comparison={'node':node,'stage':name,**compare_bytes(actual,expected)}
            evidence['stage_comparisons'].append(comparison)
            if comparison['different_words']:
                output_path.with_name(output_path.stem+'-node'+str(node)+'-'+name+'-tree.f32').write_bytes(actual)
                output_path.with_name(output_path.stem+'-node'+str(node)+'-'+name+'-serial.f32').write_bytes(expected)
    import math
    evidence['independent_scalar_fp64_normalization']=[]
    for mode in ('serial','tree'):
        scale=words(part_bytes('norm_scale',mode=mode));assert len(scale)==h
        assert words(part_bytes('status',mode=mode))==[0.]
        for node in path:
            raw=words(part_bytes('raw_hidden',node,mode));actual=words(part_bytes('normalized',node,mode))
            mean=math.fsum(float(v)*float(v) for v in raw)/h;denom=math.sqrt(mean+c['rms_norm_eps'])
            independent=[raw[k]/denom*scale[k] for k in range(h)]
            errors=[abs(a-e) for a,e in zip(actual,independent)]
            evidence['independent_scalar_fp64_normalization'].append({'mode':mode,'node':node,'scalar_fp64_mean_square':mean,
                'scalar_fp64_denominator':denom,'probe_mean_square':words(part_bytes('mean_square',node,mode))[0],
                'probe_denominator':words(part_bytes('denominator',node,mode))[0],'max_abs_normalized_error':max(errors),
                'max_relative_normalized_error':max(err/max(abs(ref),1e-30) for err,ref in zip(errors,independent)),
                'scope':'Independent scalar FP64 diagnostic from exported raw hidden and scale, no replacement of failed original exact gate.'})
    assert canonical()==before;evidence['canonical_after']=canonical()
    evidence['original_exact_gate_still_failed']=True;evidence['diagnostic_complete']=True
    d._release_state_transaction(owner);claimed=False
    serial_probe.close();tree_probe.close();serial_probe=tree_probe=None
    tree.close();reference.close();tree=reference=None;model.close();model=None
    event('closed');save();print('MTP_STAGE_DIAGNOSTIC_COMPLETE_ORIGINAL_EXACT_GATE_FAILED',flush=True)

except Exception as e:
    if tree is not None:
        evidence['last_tree_status']=list(tree.last_status)
        evidence['last_frontier_controls']=getattr(tree,'last_frontier_controls',None)
    evidence['error']=repr(e);save();raise
finally:
    original.CUDASpeculativeSession=saved_factory
    if claimed and model is not None:model.decoder._release_state_transaction(owner)
    if serial_probe is not None:serial_probe.close()
    if tree_probe is not None:tree_probe.close()
    if reference is not None:reference.close()
    if tree is not None:tree.close()
    if model is not None:model.close()
