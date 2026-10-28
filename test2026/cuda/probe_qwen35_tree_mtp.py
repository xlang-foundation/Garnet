"""Recursive learned MTP tree versus original serial MTP at matched GEMM geometry.
Correctness-only; not target generation, model quality, latency or process VRAM.
"""
import hashlib,json,os,struct,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
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
model=tree=reference=None;owner=object();claimed=False;saved_factory=original.CUDASpeculativeSession
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
    d._claim_state_transaction(owner);claimed=True
    remaining=[16]*b
    proposal=tree.draft(s,current,[1]*b,remaining,s.current_positions,model.stops,owner,request['beam'])
    assert canonical()==before;features_bytes=io.finite_f32_bytes(tree.features)
    evidence['proposal']=proposal;evidence['feature_sha256']=hashlib.sha256(features_bytes).hexdigest();save()
    assert all(proposal['mtp_active'][r][0]==0 and proposal['parents'][r][0]==-1 for r in range(b))
    assert all(features_bytes[(r*n)*h*4:(r*n+1)*h*4]==root_bytes[r*h*4:(r+1)*h*4] for r in range(b))
    assert any(len(proposal['rounds'][0]['expanded_nodes'][r])==2 for r in range(b)),'Actual learned siblings were not expanded'
    # Select up to beam distinct active leaves per row, deepest first. Each
    # serial oracle recursively computes its OWN parent features, never tree
    # features, and starts from unchanged original canonical MTP caches.
    leaves=[]
    for r in range(b):
        active=[j for j in range(1,proposal['used_nodes'][r]) if proposal['mtp_active'][r][j]]
        children={proposal['parents'][r][j] for j in active}
        candidates=sorted((j for j in active if j not in children),key=lambda j:(-proposal['depth'][r][j],j))
        assert candidates,'No learned active MTP leaf'
        leaves.append(candidates[:request['beam']])
    target=io.zeros([b,n,h]);embedding=io.zeros([b,n,h]);position=io.zeros([b,n,3]);slot=io.zeros([b,n]);flags=io.zeros([b,n])
    packed_hidden=io.zeros([b,n,h]);out_key=io.zeros([b,s.cache_width]);out_value=io.zeros([b,s.cache_width])
    exact_feature_words=0;exact_kv_words=0
    # The final MTP call may precede inactive terminal proposals; all active
    # nodes were run. Keep the actual packet before subsequent error/replays.
    packet=words(io.finite_f32_bytes(tree.last_output));pw=tree.output_width
    for path_index in range(max(len(v) for v in leaves)):
        paths=[]
        for r in range(b):
            node=leaves[r][path_index%len(leaves[r])];path=[]
            while node>0:path.append(node);node=proposal['parents'][r][node]
            path.reverse();paths.append(path)
        oracle=[{} for _ in range(b)];last_result=None
        for level in range(1,max(len(v) for v in paths)+1):
            tf=[0.]*(b*n*h);rf=[];rs=[];rp=[];ids=[[0]*b for _ in range(n)]
            for r in range(b):
                base=s.slots[r]
                for t in range(n):
                    valid=t<min(level,len(paths[r]));rf.append(float(valid));rs.append(float(base+t if valid else base+min(level,len(paths[r]))))
                    rp.extend([s.current_positions[r]+t+1]*3 if valid else [0,0,0])
                    if valid:
                        node=paths[r][t];ids[t][r]=proposal['tokens'][r][node]
                        source=root[r*h:(r+1)*h] if t==0 else oracle[r][paths[r][t-1]]
                        tf[(r*n+t)*h:(r*n+t+1)*h]=source
            io.update(target,tf);io.embedding_chunk(ids,embedding);io.update(flags,rf);io.update(slot,rs);io.update(position,rp)
            last_result=reference.run([target,embedding,s.key,s.value,position,slot,flags])
            io.slice_into(last_result,0,packed_hidden);raw=io.finite_f32_bytes(packed_hidden);values=words(raw)
            for r in range(b):
                for t,node in enumerate(paths[r][:level]):
                    oracle[r][node]=values[(r*n+t)*h:(r*n+t+1)*h]
                    actual=features_bytes[(r*n+node)*h*4:(r*n+node+1)*h*4];expected=raw[(r*n+t)*h*4:(r*n+t+1)*h*4]
                    if actual!=expected:
                        a=output_path.with_name(output_path.stem+'-tree-feature.f32');e=output_path.with_name(output_path.stem+'-serial-feature.f32')
                        a.write_bytes(actual);e.write_bytes(expected);evidence['first_difference']={'path':path_index,'level':level,'row':r,'node':node,'actual':str(a),'reference':str(e)};save()
                        raise AssertionError('Recursive MTP tree feature differs from original serial MTP at matched geometry')
                    exact_feature_words+=h
        io.slice_into(last_result,n*h,out_key);io.slice_into(last_result,n*h+s.cache_width,out_value)
        key_words=words(io.finite_f32_bytes(out_key));value_words=words(io.finite_f32_bytes(out_value))
        for r in range(b):
            for t,node in enumerate(paths[r]):
                for head in range(kv):
                    for k in range(dim):
                        original_index=r*s.cache_width+(head*d.capacity+s.slots[r]+t)*dim+k
                        tree_index=r*pw+n*h+(node*kv+head)*dim+k
                        assert packet[tree_index]==key_words[original_index],'MTP tree key differs from serial original'
                        assert packet[tree_index+n*kv*dim]==value_words[original_index],'MTP tree value differs from serial original'
                        exact_kv_words+=2
        evidence['paths'].append({'case':path_index,'active_paths':paths,'exact_feature_words_running':exact_feature_words,'exact_kv_words_running':exact_kv_words});save()
        print('TREE_MTP_RECURSIVE_PATH_PASS',path_index,flush=True)
    evidence['exact_recursive_feature_words']=exact_feature_words;evidence['exact_node_kv_words']=exact_kv_words
    repeated=tree.draft(s,current,[1]*b,remaining,s.current_positions,model.stops,owner,request['beam'])
    assert repeated==proposal and io.finite_f32_bytes(tree.features)==features_bytes and canonical()==before
    evidence['exact_deterministic_replay']=True
    evidence['wrong_owner_rejected']=reject(lambda:tree.run(s.key,s.value,object()))
    parents=[v for row in proposal['parents'] for v in row]
    # Invalidate a live node in the LAST row. Safe gather must reject every
    # form before publishing features, including nonfinite and huge values.
    bad_node=leaves[-1][0];bad_index=(b-1)*n+bad_node;negatives=[]
    baseline_features=io.finite_f32_bytes(tree.features)
    for bad in (float('nan'),float('inf'),-float('inf'),.5,1e30,float(n),float(bad_node)):
        control=list(parents);control[bad_index]=bad;io.update(tree.parents,control)
        reason=reject(lambda:tree.run(s.key,s.value,owner))
        assert any(v>0 for v in tree.last_status) and io.finite_f32_bytes(tree.features)==baseline_features and canonical()==before
        negatives.append({'bad':str(bad),'reason':reason,'status':list(tree.last_status)})
    evidence['safe_gather_rejections']=negatives
    recovered=tree.draft(s,current,[1]*b,remaining,s.current_positions,model.stops,owner,request['beam'])
    assert recovered==proposal and io.finite_f32_bytes(tree.features)==features_bytes and canonical()==before
    evidence['invalid_to_valid_recovery']=True
    # Ragged lengths/inactive rows/real EOS proposals use the actual recursive
    # implementation; no target inputs or shifted MTP caches are committed.
    short=[2+(r%3) for r in range(b)];working=[int(r%3!=2) for r in range(b)]
    ragged=tree.draft(s,current,working,short,s.current_positions,model.stops,owner,request['beam'])
    for r in range(b):
        assert all(ragged['depth'][r][j]<short[r] for j in range(ragged['used_nodes'][r]))
        if not working[r]:assert not any(ragged['active'][r]) and ragged['used_nodes'][r]==1
    eos=tree.draft(s,current,[1]*b,remaining,s.current_positions,set(range(c['vocab_size'])),owner,request['beam'])
    assert all(v==1 for v in eos['used_nodes']) and not any(v for row in eos['active'] for v in row)
    assert canonical()==before
    evidence['ragged_controls']=ragged;evidence['all_current_stopped_no_expansion']=True;evidence['canonical_after']=canonical()
    evidence['memory_scoped_diagnostic']=tree.memory_info()
    d._release_state_transaction(owner);claimed=False
    tree.close();reference.close();tree=reference=None;model.close();model=None
    event('closed');evidence['all_passed']=True;save();print('TREE_MTP_GROUP_PASS',request['model_id'],b,flush=True)
except Exception as e:
    if tree is not None:
        evidence['last_tree_status']=list(tree.last_status)
        evidence['last_frontier_controls']=getattr(tree,'last_frontier_controls',None)
    evidence['error']=repr(e);save();raise
finally:
    original.CUDASpeculativeSession=saved_factory
    if claimed and model is not None:model.decoder._release_state_transaction(owner)
    if reference is not None:reference.close()
    if tree is not None:tree.close()
    if model is not None:model.close()
