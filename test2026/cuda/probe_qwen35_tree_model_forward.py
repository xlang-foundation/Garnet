"""Initial learned whole-target tree smoke, not a generator or benchmark.

Compare active chain1/4/8 outputs against the unchanged native serial target at
the SAME B*N GEMM geometry, starting from the same real-image canonical prefix.
This deliberately does not qualify adaptive branches, model quality or speed.
"""
import hashlib,json,os,struct,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
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
    print('TREE_MODEL_FORWARD_EVENT',name,flush=True)
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

model=tree=reference=None;owner=object();claimed=False
try:
    assert request['profile']['batch']==1,'Initial learned smoke is B1 only'
    case=request['case'];assert len(case['images'])==len(case['prompts'])==1
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
    tree=CUDATreeDecoderProgram(d,n)
    assert n==d.prefill_chunk_size,'Reuse the existing serial prefill graph at matching GEMM geometry'
    reference=d.programs[n]
    shape=[b,n,h];input_hidden=io.zeros(shape);active=io.zeros([b,n]);pos=io.zeros([b,n,3]);parents=io.zeros([b,n]);prefix=io.zeros([b,1]);slots=io.zeros([b,n])
    serial_hidden=io.zeros(shape)
    ids=[[int(prediction[r]) if t==0 else 100+(t*37)%10000 for r in range(b)] for t in range(n)]
    io.embedding_chunk(ids,input_hidden)
    event('tree_and_matched_geometry_reference_built')
    evidence['parent_close_while_tree_borrowed_rejection']=reject(d.close)
    for length in (1,4,8):
        flags=[float(t<length) for r in range(b) for t in range(n)]
        parent_values=[float(-1 if t==0 else t-1 if t<length else 0) for r in range(b) for t in range(n)]
        position_values=[float(rows[r][3]+min(t,length)) for r in range(b) for t in range(n) for axis in range(3)]
        slot_values=[float(d.slots[r]+min(t,length)) for r in range(b) for t in range(n)]
        io.update(active,flags);io.update(pos,position_values);io.update(parents,parent_values);io.update(prefix,[float(v) for v in d.slots]);io.update(slots,slot_values)
        d._claim_state_transaction(owner);claimed=True
        tree.run(input_hidden,active,pos,parents,prefix,owner)
        raw,actual=hidden_words(io,tree.hidden)
        serial=reference.run([input_hidden,d.state,active,pos,slots]);io.slice_into(serial,0,serial_hidden)
        raw_reference,expected=hidden_words(io,serial_hidden)
        bad=[];max_abs=0.;max_fraction=0.;exact=0
        if raw==raw_reference:exact=len(actual)
        else:
            for i,(x,y) in enumerate(zip(actual,expected)):
                error=abs(x-y);bound=2e-5+2e-4*abs(y)
                max_abs=max(max_abs,error);max_fraction=max(max_fraction,error/bound)
                if raw[4*i:4*i+4]==raw_reference[4*i:4*i+4]:exact+=1
                elif len(bad)<32:bad.append({'index':i,'actual':x,'reference':y,'abs_error':error,'original_bound':bound})
        result={'chain_active_length':length,'compared_words':len(actual),'exact_words':exact,'max_abs':max_abs,'max_original_bound_fraction':max_fraction,
            'first_word_differences':bad,'tree_sha256':hashlib.sha256(raw).hexdigest(),'reference_sha256':hashlib.sha256(raw_reference).hexdigest(),
            'canonical_unchanged':canonical(d)==before,'status':[tree.last_status[r*len(tree.layout):(r+1)*len(tree.layout)] for r in range(b)]}
        evidence['cases'].append(result);save()
        assert raw==raw_reference,'Learned same-geometry serial/tree hidden words changed; retain original failure'
        assert result['canonical_unchanged'],'Read-only whole-target forward changed canonical state'
        if length==1:
            result['wrong_owner_rejected']=reject(lambda:tree.predictions(object()))
            result['tree_close_pending_rejected']=reject(tree.close)
            result['parent_reset_pending_rejected']=reject(d.reset)
            result['ordinary_chunk_pending_rejected']=reject(lambda:d.chunk_hidden([],[],[],False))
        d._release_state_transaction(owner);claimed=False
        event('chain_'+str(length)+'_passed')
    # Reusing the same Python owner across a NEW transaction cannot authorize
    # predictions from the preceding successful verification.
    d._claim_state_transaction(owner);claimed=True
    evidence['prior_transaction_prediction_rejected']=reject(lambda:tree.predictions(owner))
    io.update(prefix,[-1.]*b)
    evidence['invalid_metadata_rejection']=reject(lambda:tree.run(input_hidden,active,pos,parents,prefix,owner))
    evidence['failed_run_prediction_rejected']=reject(lambda:tree.predictions(owner))
    assert canonical(d)==before
    io.update(prefix,[float(v) for v in d.slots]);tree.run(input_hidden,active,pos,parents,prefix,owner)
    assert hidden_words(io,tree.hidden)[0]==raw and canonical(d)==before
    evidence['same_graph_invalid_to_valid_recovery_exact']=True
    d._release_state_transaction(owner);claimed=False
    evidence['tree_memory_scoped_diagnostic']=tree.memory_info()
    evidence['all_passed']=True
    evidence['scope']='B1 learned real-image prefix; same-geometry chain1/4/8 target hidden outputs, readonly canonical state, status recovery and parent/transaction ownership only. No adaptive-branch, recurrent commit, MTP generation, independent quality, speed, batch/capacity or process-memory qualification.'
    save();event('initial_learned_tree_forward_passed')
except Exception as error:
    evidence['failure']=str(error);save();raise
finally:
    if claimed:model.decoder._release_state_transaction(owner)
    if tree is not None:tree.close()
    if model is not None:model.close()
    event('closed')
