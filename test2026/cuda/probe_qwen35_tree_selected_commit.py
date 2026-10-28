"""Learned selected recurrent/conv/KV canonical commit against the original serial target.
Correctness-only reference copies; no generator, quality, speed or process-memory claim.
"""
import hashlib,json,os,struct,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b.cuda_tree_program import CUDATreeDecoderProgram

request_path,output_path=map(Path,sys.argv[1:3])
assert not output_path.exists(),'Preserve original learned tree outcome'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
evidence={'request':request,'events':[],'cases':[],'all_passed':False,
    'performance_or_process_memory_measurement':False,'independent_model_quality_proven':False,
    'branching_or_speculative_generation_accepted':False,'full_goal_met':False}
def save():output_path.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
def event(name):
    evidence['events'].append({'name':name,'unix_seconds':time.time(),'pid':os.getpid()});save()
    print('TREE_SELECTED_COMMIT_EVENT',name,flush=True)
def reject(call):
    try:call()
    except Exception as e:return str(e)
    raise AssertionError('Expected ownership/status rejection')
def canonical(d):
    flat=d.io.storage_view(d.state,[d.batch,d.programs[d.decode_chunk_size].state_width])
    width=d.programs[d.decode_chunk_size].state_width;h=hashlib.sha256()
    for start in range(0,width,131072):
        count=min(131072,width-start)
        value=d.io.slice(flat,start,[d.batch,count]);h.update(d.io.finite_f32_bytes(value))
    return h.hexdigest()
def hidden_words(io,value):
    data=io.finite_f32_bytes(value)
    return data,struct.unpack('<'+str(len(data)//4)+'f',data)

model=tree=reference=committer=None;owner=object();claimed=False
try:
    assert request['profile']['batch'] in (1,2,4,8),'Selected commit batch profile invalid'
    case=request['case'];assert len(case['images'])==len(case['prompts'])==request['profile']['batch']
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
    event('load_begin');model=Model(request['checkpoint'],request['profile'],request['model_id']);d=model.decoder;io=d.io
    event('load_complete');d.reset()
    rows=[model._prepare(case['prompts'][r],case['images'][r]) for r in range(d.batch)]
    assert model.pack_prompt
    hidden=list(io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
    count=max(len(r[2]) for r in rows)
    positions=[r[1]+[[0,0,0]]*(count-len(r[1])) for r in rows]
    prediction=d.prefill_sequence(hidden,positions,[len(r[2]) for r in rows])
    evidence['real_image_prefix']={'slots':list(d.slots),'first_prediction':list(prediction),'prompt_token_ids':[r[2] for r in rows]}
    before=canonical(d);evidence['canonical_before_sha256']=before;event('real_image_prefill_complete')
    n=request['nodes'];b=d.batch;h=d.config['hidden_size']
    tree=CUDATreeDecoderProgram(d,n);reference=d.programs[n]
    assert n==d.prefill_chunk_size and reference.compact_kv_outputs and reference.state_commit is not None
    committer=garnet.cuda_tree_commit()
    specification={'config':d.config,'layout':tree.layout,'batch':b,'nodes':n,'capacity':d.capacity,
        'state_width':tree.state_width,'packed_width':tree.output_width,'hidden_width':tree.hidden_width,
        'status_offset':tree.status_offset,'metadata_offset':tree.metadata_offset}
    committer.build(d.checkpoint,json.dumps(specification))
    base=io.clone(d.state);reference_state=io.clone(d.state)
    input_hidden=io.zeros([b,n,h]);active=io.zeros([b,n]);pos=io.zeros([b,n,3]);parents=io.zeros([b,n]);prefix=io.zeros([b,1])
    ref_active=io.zeros([b,n]);ref_slots=io.zeros([b,n])
    io.embedding_chunk([[int(prediction[r]) if t==0 else 100+(t*37+r*19)%10000 for r in range(b)] for t in range(n)],input_hidden)
    flags=[float(t<8) for r in range(b) for t in range(n)]
    parent_values=[float(-1 if t==0 else t-1 if t<8 else 0) for r in range(b) for t in range(n)]
    position_values=[float(rows[r][3]+min(t,8)) for r in range(b) for t in range(n) for axis in range(3)]
    prefix_values=[float(v) for v in d.slots]
    io.update(active,flags);io.update(parents,parent_values);io.update(pos,position_values);io.update(prefix,prefix_values)
    def compare_states(actual,expected):
        ha=hashlib.sha256();he=hashlib.sha256();compared=0
        for start in range(0,tree.state_width,131072):
            width=min(131072,tree.state_width-start)
            a=io.finite_f32_bytes(io.slice(actual,start,[b,width]));e=io.finite_f32_bytes(io.slice(expected,start,[b,width]))
            if a!=e:
                evidence['canonical_difference']={'chunk_row_start':start,'words_per_row':width,'actual_sha256':hashlib.sha256(a).hexdigest(),'reference_sha256':hashlib.sha256(e).hexdigest()};save()
                raise AssertionError('Selected tree commit changed full canonical words against original serial native target')
            ha.update(a);he.update(e);compared+=b*width
        return {'exact_canonical_words':compared,'actual_sha256':ha.hexdigest(),'reference_sha256':he.hexdigest()}
    event('selected_commit_helper_and_reference_copies_ready')
    d._claim_state_transaction(owner);claimed=True
    tree.run(input_hidden,active,pos,parents,prefix,owner)
    assert canonical(d)==before
    for limit in (0,1,4,8):
        counts=[max(0,limit-r%(limit+1)) if limit else 0 for r in range(b)]
        selections=[]
        for r in range(b):selections.extend([d.slots[r],counts[r]-1])
        io.copy_into(base,d.state);io.copy_into(base,reference_state)
        io.update(ref_active,[float(t<counts[r]) for r in range(b) for t in range(n)])
        io.update(ref_slots,[float(d.slots[r]+min(t,counts[r])) for r in range(b) for t in range(n)])
        original=reference.run([input_hidden,d.state,ref_active,pos,ref_slots])
        reference.commit_all(original,reference_state,ref_slots,ref_active)
        d.poisoned=True
        code=committer.commit(tree.last_output,d.state,parents,active,prefix,selections)
        assert code==0,'Selected native commit unexpectedly rejected a valid packet'
        result=compare_states(d.state,reference_state)
        result.update({'selection_limit':limit,'accepted_input_counts':counts,'native_status':code})
        evidence['cases'].append(result);save();d.poisoned=False
        event('selected_count_'+str(limit)+'_passed')
    # A malformed last row must suppress selected writes in the FIRST row too.
    # Other rows may have empty selections; validation still covers every row.
    io.copy_into(base,d.state);selections=[]
    for r in range(b):selections.extend([d.slots[r],7 if r==0 else -1])
    evidence['gpu_rejections_without_canonical_advance']=[]
    for kind in ('parent','active','prefix'):
        pp=list(parent_values);aa=list(flags);ss=list(prefix_values)
        if kind=='parent':pp[(b-1)*n]=0.
        elif kind=='active':aa[(b-1)*n]=.5
        else:ss[b-1]=-1.
        io.update(parents,pp);io.update(active,aa);io.update(prefix,ss)
        d.poisoned=True
        code=committer.commit(tree.last_output,d.state,parents,active,prefix,selections)
        assert code==1,'Malformed control failed whole-batch commit rejection'
        assert canonical(d)==before,'Rejected packet advanced canonical state'
        d.poisoned=False;evidence['gpu_rejections_without_canonical_advance'].append(kind);save()
    io.update(parents,parent_values);io.update(active,flags);io.update(prefix,prefix_values)
    # Recover using the same captured forward and the same commit buffers.
    tree.run(input_hidden,active,pos,parents,prefix,owner)
    io.copy_into(base,reference_state)
    counts=[8 if r==0 else 0 for r in range(b)]
    io.update(ref_active,[float(t<counts[r]) for r in range(b) for t in range(n)])
    io.update(ref_slots,[float(d.slots[r]+min(t,counts[r])) for r in range(b) for t in range(n)])
    original=reference.run([input_hidden,d.state,ref_active,pos,ref_slots]);reference.commit_all(original,reference_state,ref_slots,ref_active)
    d.poisoned=True;assert committer.commit(tree.last_output,d.state,parents,active,prefix,selections)==0
    evidence['invalid_to_valid_recovery']=compare_states(d.state,reference_state);d.poisoned=False
    io.copy_into(base,d.state);assert canonical(d)==before
    d._release_state_transaction(owner);claimed=False
    evidence['helper_memory_scoped_diagnostic']=json.loads(committer.stats_json())
    evidence['correctness_only_reference_copy_bytes']=2*io.tensor_bytes(base)
    evidence['scope']='Learned real-image canonical prefix, original P64 native serial reference at matching B*N geometry, empty/partial/ragged selected chain inputs, ALL recurrent/conv/KV canonical words exact. Whole-batch bad parent/active/prefix no-write rejection and recovery. Correctness-only full-state reference copies are excluded from any future serving-memory/speed claim. Not adaptive branching/MTP generation, independent model quality or performance/process-memory qualification.'
    evidence['all_passed']=True;save();event('learned_selected_commit_passed')
except Exception as error:
    evidence['failure']=str(error);save();raise
finally:
    if claimed:model.decoder._release_state_transaction(owner)
    if committer is not None:committer.close()
    if tree is not None:tree.close()
    if model is not None:model.close()
    event('closed')
