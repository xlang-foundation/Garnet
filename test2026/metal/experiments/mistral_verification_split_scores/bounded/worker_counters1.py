"""Diagnostic saved-target replay with calibrated direct GPU encoder intervals.
No draft inference, HTTP throughput, TTFT, or coding-quality admission.
Arguments: LIB SOURCE SAVED-STAGE NEW-OUTPUT.
"""
import os,sys,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:saved=json.load(f)
inputs=saved['inputs'];out=sys.argv[4];assert not os.path.exists(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert 'native_fused_q4_decode_rows4_default_v1' not in garnet.backend_info()
assert os.getenv('GARNET_METAL_PROFILE_OPS') in (None,'0')
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
decoder=None;records=[];prepared=None;cleanup=None
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def mark(event,phase,case,tick,extra=None):
    row=dict(event=event,phase=phase,case=case,tick=tick)
    if extra is not None:row['stats']=extra
    print('verification_gpu_frame '+json.dumps(row),flush=True)
def save(done):
    with open(out,'w') as f:json.dump(dict(completed=done,scope=__doc__,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,records=records,cleanup=cleanup),f,indent=2)
try:
    decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,1,inputs['capacity'])
    decoder.prepare_prefill(128);decoder.prepare_verification(4)
    assert decoder.session.prepare_gpu_counters()
    prepared=dict(target_graphs=len(decoder.prefill_graphs),memory=json.loads(garnet.memory_stats_json()))
    assert prepared['target_graphs']==224
    assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-268435456
    for case,old in zip(inputs['cases'],saved['records']):
        label=case['id'];assert label==old['case_id']
        mark('begin','reset',label,-1);decoder.reset()
        assert decoder.session.verification_set_mistral_layout(True)
        assert decoder.session.verification_set_value_vector4(True)
        mark('end','reset',label,-1)
        prompt=case['prompt_ids'];mark('begin','prefill',label,-1)
        known=decoder.prefill_sequence([decoder.io.embedding([token]) for token in prompt],[[[i]*3 for i in range(len(prompt))]])[0]
        mark('end','prefill',label,-1)
        assert known==old['generated_ids'][0];generated=[known];slot=len(prompt);rounds=[]
        for tick,q in enumerate(old['rounds']):
            assert q['slot']==slot and q['committed_before']==len(generated) and q['submitted_ids'][0]==known
            proposal=q['proposal'];remaining=inputs['output_limit']-len(generated);phase='verify' if proposal else 'fallback'
            mark('begin',phase,label,tick)
            if proposal:
                length=len(proposal)+1;submitted=[known]+proposal+[0]*(4-length)
                assert submitted==q['submitted_ids'];positions=[float(slot+t) for t in range(4)]
                result=decoder.session.verify_greedy(decoder.io.stack_tokens([decoder.io.embedding([token]) for token in submitted]),val([p for p in positions for axis in range(3)],[1,4,3]),val(positions,[1,4]),val([float(t<length) for t in range(4)],[1,4]),submitted,inputs['target_stops'],[remaining],None)[0]
                emitted=list(result[0]);consumed=result[1];stats=json.loads(decoder.session.verification_stats_json())
                assert stats['target_affine_dispatches']==225 and stats['target_affine_misses']==0
                assert stats['target_attention_layout_dispatches']==stats['target_attention_vector4_dispatches']==32
                assert stats['target_affine_decode_block_dispatches']==(161 if length==4 else 1)
            else:
                emitted=[decoder.step([known])[0]];consumed=1;stats=dict(gpu_ms=decoder.session.gpu_ms)
            mark('end',phase,label,tick,stats)
            assert emitted==q['emitted_ids'] and consumed==q['consumed']
            generated.extend(emitted);known=emitted[-1];slot+=consumed;decoder.slots=[slot]
            rounds.append(dict(tick=tick,phase=phase,active_tokens=len(proposal)+1,consumed=consumed,stats=stats))
        mark('begin','witness',label,-1)
        kv=[[decoder.io.verification_f32_sha256(x) for x in decoder.session.snapshot(i)] for i in range(32)]
        hidden=decoder.io.verification_f32_sha256(decoder.session.hidden())
        mark('end','witness',label,-1)
        assert generated==old['generated_ids'] and slot==old['slots'] and kv==old['state_sha256'] and hidden==old['hidden_sha256']
        records.append(dict(case_id=label,prompt_ids=prompt,generated_ids=generated,slots=slot,state_sha256=kv,hidden_sha256=hidden,rounds=rounds));save(False)
        print('verification_gpu_case_complete '+label,flush=True)
finally:
    mark('begin','close','all',-1)
    if decoder is not None:decoder.close()
    garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
    mark('end','close','all',-1)
    for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:assert cleanup[key]==0
    assert cleanup['metal_allocated_bytes']<=4194304
    save(len(records)==len(inputs['cases']))
assert len(records)==len(inputs['cases'])==1
