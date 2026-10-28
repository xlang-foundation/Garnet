"""Full4K candidate target-state replay; no timing/quality promotion."""
import hashlib,math,struct
from .bf16_packed_state_gate import same_packed_bf16_row
ORDER=(0,1,2,3,0,2)


def replay_report(report,read,cases):
    assert report['status']=='mtp_prefill16_full4k_collected_needs_replay'
    assert type(report['physical_batch']) is int and report['physical_batch']==1 and type(report['capacity']) is int and report['capacity']==4096 and len(cases)==4
    assert report['quality_qualified'] is report['performance_admitted'] is False
    states=0
    def state(value,slots):
        nonlocal states
        logical=value['logical'];assert logical['slots']==[slots] and same_packed_bf16_row(logical,0,logical,0)
        assert set(value['vectors'])=={'hidden','normalized'}
        for item in value['vectors'].values():
            assert '/' not in item['file'] and '\\' not in item['file'];raw=read(item['file'])
            assert len(raw)==item['bytes']==20480 and hashlib.sha256(raw).hexdigest()==item['sha256']
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
        states+=1;return value
    def same(a,b):
        assert a['logical']==b['logical'] and all(a['vectors'][n]['sha256']==b['vectors'][n]['sha256'] for n in ('hidden','normalized'))
    empty=state(report['empty_state'],0);same(empty,state(report['final_reset'],0));assert len(report['cases'])==len(ORDER)
    earlier={};summaries=[]
    for sequence,(record,index) in enumerate(zip(report['cases'],ORDER)):
        control=cases[index];assert type(record['case_index']) is int and record['case_index']==index and type(record['sequence']) is int and record['sequence']==sequence and record['key']==control['key'] and record['request_sha256']==control['request_sha256'] and record['image_sha256']==control['image_sha256']
        assert record['prompt_ids']==control['prompt_ids'] and len(record['prompt_ids'])==3968
        assert record['ordinary_ids']==record['candidate_ids']==control['generated_ids_by_mode']['ordinary']==control['generated_ids_by_mode']['mtp']
        ids=record['candidate_ids'];assert len(ids)==128 and all(type(t) is int and 0<=t<248320 for t in ids)
        same(empty,state(record['reset_state'],0))
        a=state(record['ordinary_prefill'],3968);b=state(record['candidate_prefill'],3968);same(a,b)
        c=state(record['ordinary_final'],4095);d=state(record['candidate_final'],4095);same(c,d)
        assert record['prefill_first_tokens']==[[ids[0]]] and record['callbacks']==[[0,t,i+1] for i,t in enumerate(ids)]
        assert record['sampled_tokens_including_stop']==[128] and record['stop_token_ids']==[None] and record['speculative_metrics']['mtp_prefill_calls']==249
        if index in earlier:
            old=earlier[index]
            for name in ('ordinary_prefill','candidate_prefill','ordinary_final','candidate_final'):same(record[name],old[name])
        else:earlier[index]=record
        summaries.append({'sequence':sequence,'key':record['key'],'case':control['case'],'actual_prompt_tokens':3968,'actual_output_tokens':128,'target_complete_state_and_raw_vectors_exact':True,'actual_candidate_draft_prefill_calls':249,'reset_repeat':index in ORDER[:sequence]})
    return {'status':'mtp_prefill16_full4k_target_state_replayed','cases':summaries,'captured_states':states,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':'Actual six3968+128 ordinary/candidate comparisons, complete target canonical state and finite raw vectors at prefill/final, raw IDs/callbacks/reset-repeat and actual249draft-prefill calls. No805quality, HTTP, isolated timing, throughput, repetition benchmark or2x admission.'}
