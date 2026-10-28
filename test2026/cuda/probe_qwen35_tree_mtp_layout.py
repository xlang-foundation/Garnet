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
    # Observational diagnostic retains the original compact exact failure.
    # It does not redefine or satisfy that failed gate.
    parents=proposal['parents'][0];active=proposal['mtp_active'][0]
    live=[j for j in range(1,proposal['used_nodes'][0]) if active[j]]
    child={parents[j] for j in live};leaf=sorted((j for j in live if j not in child),key=lambda j:(-proposal['depth'][0][j],j))[0]
    path=[];node=leaf
    while node>0:path.append(node);node=parents[node]
    path.reverse();assert b==1 and n==32 and path[:3]==[1,3,7]
    evidence['diagnostic_active_path']=path
    packet_bytes=io.finite_f32_bytes(tree.last_output);packet=words(packet_bytes)
    target=io.zeros([b,n,h]);embedding=io.zeros([b,n,h]);position=io.zeros([b,n,3]);slot=io.zeros([b,n]);flags=io.zeros([b,n])
    packed_hidden=io.zeros([b,n,h]);out_key=io.zeros([b,s.cache_width]);out_value=io.zeros([b,s.cache_width])
    observations=[]
    for mode in ('original_compact','same_physical_nodes'):
        oracle={};rows=[]
        for level in range(1,len(path)+1):
            tf=[0.]*(n*h);rf=[0.]*n;rs=[float(s.slots[0])]*n;rp=[0.]*(n*3);ids=[[0] for _ in range(n)]
            source_checks=[]
            for t,node in enumerate(path[:level]):
                index=t if mode=='original_compact' else node
                source=root[:h] if t==0 else oracle[path[t-1]]
                tf[index*h:(index+1)*h]=source;rf[index]=1.;rs[index]=float(s.slots[0]+t)
                rp[index*3:index*3+3]=[float(s.current_positions[0]+t+1)]*3;ids[index][0]=proposal['tokens'][0][node]
                original_parent=parents[node]
                source_bytes=struct.pack('<'+'f'*h,*source)
                original_source=features_bytes[original_parent*h*4:(original_parent+1)*h*4]
                source_checks.append({'node':node,'source_equals_original_tree_parent':source_bytes==original_source})
            io.update(target,tf);io.embedding_chunk(ids,embedding);io.update(flags,rf);io.update(slot,rs);io.update(position,rp)
            result=reference.run([target,embedding,s.key,s.value,position,slot,flags])
            io.slice_into(result,0,packed_hidden);raw=io.finite_f32_bytes(packed_hidden);values=words(raw)
            io.slice_into(result,n*h,out_key);io.slice_into(result,n*h+s.cache_width,out_value)
            key_data=io.finite_f32_bytes(out_key);value_data=io.finite_f32_bytes(out_value)
            compared=[]
            for t,node in enumerate(path[:level]):
                index=t if mode=='original_compact' else node
                oracle[node]=values[index*h:(index+1)*h]
                a=features_bytes[node*h*4:(node+1)*h*4];e=raw[index*h*4:(index+1)*h*4]
                aa=words(a);ee=words(e);bad=[k for k in range(h) if a[k*4:k*4+4]!=e[k*4:k*4+4]]
                kb=vb=0
                for head in range(kv):
                    for k in range(dim):
                        oi=(head*d.capacity+s.slots[0]+t)*dim+k;ti=n*h+(node*kv+head)*dim+k
                        kb+=packet_bytes[ti*4:ti*4+4]!=key_data[oi*4:oi*4+4]
                        vi=ti+n*kv*dim;vb+=packet_bytes[vi*4:vi*4+4]!=value_data[oi*4:oi*4+4]
                compared.append({'node':node,'serial_physical_row':index,'different_feature_words':len(bad),
                    'max_abs':max((abs(aa[k]-ee[k]) for k in bad),default=0.),'different_node_key_words':kb,'different_node_value_words':vb,
                    'tree_feature_sha256':hashlib.sha256(a).hexdigest(),'serial_feature_sha256':hashlib.sha256(e).hexdigest()})
                if bad:
                    stem=output_path.stem+'-'+mode+'-l'+str(level)+'-node'+str(node)
                    (output_path.parent/(stem+'-tree.f32')).write_bytes(a);(output_path.parent/(stem+'-serial.f32')).write_bytes(e)
            rows.append({'level':level,'sources':source_checks,'comparisons':compared})
            evidence['diagnostic_partial']={'mode':mode,'rows':rows};save()
        observations.append({'mode':mode,'rows':rows})
    evidence['layout_observations']=observations
    original=observations[0]['rows'][2]['comparisons'][2]
    assert original['node']==7 and original['different_feature_words']==2021,'Original frozen failure did not reproduce; preserve diagnostic outcome'
    assert canonical()==before
    evidence['canonical_after']=canonical();evidence['memory_scoped_diagnostic']=tree.memory_info()
    evidence['original_exact_gate_still_failed']=True;evidence['diagnostic_complete']=True
    d._release_state_transaction(owner);claimed=False
    tree.close();reference.close();tree=reference=None;model.close();model=None
    event('closed');save();print('TREE_MTP_LAYOUT_DIAGNOSTIC_COMPLETE_ORIGINAL_GATE_FAILED',flush=True)
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
