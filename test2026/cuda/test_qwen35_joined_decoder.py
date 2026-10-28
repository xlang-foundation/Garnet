"""Separate old/joined fixture decoders, full finite states and explicit owners."""
import hashlib,json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];config=fixture['config'];checkpoint=fixture['checkpoint']
activation_mode=len(sys.argv)>3 and sys.argv[3]=='1'
internal_mode=len(sys.argv)>3 and sys.argv[3]=='internal-stores'
records=[]
resume_path=sys.argv[4] if internal_mode and len(sys.argv)>4 else None
if resume_path:
    prior=json.load(open(resume_path))
    assert not prior['all_passed'] and prior['current_batch']==4
    assert [r['batch'] for r in prior['records']]==[1,2]
    assert all(r['same_all_full_finite_states_and_ids'] for r in prior['records'])
    assert len(prior['baseline_snapshots'])==9 and prior['baseline_snapshots']==prior['candidate_snapshots']
    records=list(prior['records'])
    records.append({'batch':4,'snapshots':prior['candidate_snapshots'],'same_all_full_finite_states_and_ids':True,
        'recovered_from_saved_full_comparison':resume_path,'native_memory_metadata_not_persisted':True})

def digest_width(model,value,width):
    assert model.io.tensor_bytes(value)>=model.batch*width*4
    step=min(width,1048576//model.batch);result=hashlib.sha256()
    for start in range(0,width,step):
        part=model.io.slice(value,start,[model.batch,min(step,width-start)])
        result.update(model.io.finite_f32_bytes(part))
    return result.hexdigest()
def digest(model,value):
    # Preserve every finite FP32 word without per-element Python conversion.
    width=model.programs[model.decode_chunk_size].state_width
    assert model.io.tensor_bytes(value)==model.batch*width*4
    return digest_width(model,value,width)
def run(batch,joined,shared_activation=False,internal_stores=False):
    print('JOINED_DECODER_CASE_BEGIN',batch,joined,shared_activation,flush=True)
    model=CUDAPackedDecoder(checkpoint,config,batch,32,32,4,register_delta_128=True,flat_concat=True,
        shared_output_storage=True,join_decode_input_projections=joined,shared_activation_storage=shared_activation,
        internal_packed_stores=internal_stores,retain_linear_inputs=internal_mode)
    model.resident_controls=True;model.serial_target_graph_reuse=True;model.enable_cuda_graph(True)
    states=[]
    def snapshot(ids):
        states.append({'predictions':list(ids),'slots':list(model.slots),'state_sha256':digest(model,model.state),
            'hidden_sha256':hashlib.sha256(model.io.finite_f32_bytes(model.last_hidden)).hexdigest()})
        if internal_mode:states[-1]['packed_sha256']=digest_width(model,model.last_packed,model.programs[model.decode_chunk_size].output_width) if model.last_packed is not None else None
    try:
        prompt=[]
        for t in range(5):
            ids=fixture['steps'][t%4]['tokens'];prompt.append(model.io.embedding([ids[row%len(ids)] for row in range(batch)]))
        positions=[[[t]*3 for t in range(5)] for _ in range(batch)];lengths=[5-row%3 for row in range(batch)]
        snapshot(model.prefill_sequence(prompt,positions,lengths))
        for step in range(8):
            ids=fixture['steps'][step%4]['tokens'];tokens=[ids[row%len(ids)] for row in range(batch)]
            flags=[int((step+row)%4!=2) for row in range(batch)]
            snapshot(model.step(tokens,flags,[[slot]*3 for slot in model.slots]))
        info=model.memory_info();groups=info['joined_projection_group_count']
        expected=config['layer_types'].count('linear_attention') if joined else 0
        assert groups==expected
        assert sum(g['joined_projection_inputs'] for g in info['graphs'])==expected
        assert all(g['graph_capture_failures']==0 for g in info['graphs'])
        bodies=info['graphs'][:-1]
        assert sum(g['internal_packed_stores'] for g in bodies)==(2*len(config['layer_types']) if internal_stores else 0)
        assert sum(g['internal_packed_root_fills'] for g in bodies)==(2 if internal_stores else 0)
        assert all(not g['network_output_aliases_input'] for g in info['graphs'])
        assert all(g['user_managed_workspace']==shared_activation for g in bodies)
        if shared_activation:
            assert info['shared_activation_pool_bytes']>=max(g['context_workspace_required_bytes'] for g in bodies)
            assert all(g['external_workspace_bound'] and g['external_workspace_capacity_bytes']==info['shared_activation_pool_bytes'] for g in bodies)
        return states,info
    finally:model.close()
for batch in (1,2,4,8):
    if batch in [r['batch'] for r in records]:continue
    baseline,old=run(batch,activation_mode or internal_mode,internal_mode)
    candidate,new=run(batch,True,activation_mode or internal_mode,internal_mode)
    with open(output,'w') as stream:json.dump({'records':records,'current_batch':batch,'baseline_snapshots':baseline,
        'candidate_snapshots':candidate,'all_passed':False},stream,indent=2)
    assert baseline==candidate,('Joined full-state/output mismatch',batch,
        [i for i,(a,b) in enumerate(zip(baseline,candidate)) if a!=b])
    assert old['projection_weight_pool_bytes']==new['projection_weight_pool_bytes']
    records.append({'batch':batch,'snapshots':candidate,'same_all_full_finite_states_and_ids':True,
        'old_weight_bytes':old['projection_weight_pool_bytes'],'joined_weight_bytes':new['projection_weight_pool_bytes'],
        'old_owned_decoder_bytes':old['owned_decoder_bytes'],'candidate_owned_decoder_bytes':new['owned_decoder_bytes'],
        'shared_activation_pool_bytes':new['shared_activation_pool_bytes'],
        'old_body_context_bytes':[g['context_workspace_required_bytes'] for g in old['graphs'][:-1]],
        'candidate_body_context_bytes':[g['context_workspace_required_bytes'] for g in new['graphs'][:-1]],
        'internal_packed_stores':sum(g['internal_packed_stores'] for g in new['graphs']),
        'native_joined_inputs':sum(g['joined_projection_inputs'] for g in new['graphs'])})
    with open(output,'w') as stream:json.dump({'records':records,'all_passed':False,'completed_batches':[r['batch'] for r in records],'resume_source':resume_path},stream,indent=2)
with open(output,'w') as stream:json.dump({'records':records,'all_passed':True,
    'shared_activation_transition':activation_mode,
    'comparison_kind':'internal packed stores' if internal_mode else 'shared activation storage' if activation_mode else 'joined input projections',
    'resume_source':resume_path,
    'scope':'Sequential independent baseline/candidate fixture models; prefill and eight masked steps; full finite state preservation. Not pretrained quality or speed acceptance.'},stream,indent=2)
print('QWEN35_JOINED_DECODER_PASS',len(records),flush=True)
