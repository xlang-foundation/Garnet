"""Bounded input source must preserve the actual retained B8 model byte for byte."""
from .verify_bf16_batch8_admission import replay_cases

def same_state(actual,reference,read,read_reference):
    assert actual['logical']==reference['logical']
    assert set(actual['vectors'])==set(reference['vectors'])=={'hidden','normalized'}
    for kind in ('hidden','normalized'):
        a=actual['vectors'][kind];b=reference['vectors'][kind]
        assert a['bytes']==b['bytes']==163840 and a['sha256']==b['sha256']
        assert read(a['file'])==read_reference(b['file'])

def same_case(actual,reference,read,read_reference):
    for key in ('group','repeat','order','actual_limit','inputs','generated_ids','text'):
        assert actual[key]==reference[key],key
    for name in ('reset_state','prefill_state','final_state'):same_state(actual[name],reference[name],read,read_reference)
    stats=actual['prompt_cache'];count=max(len(r['prompt_ids']) for r in actual['inputs'])
    assert stats['physical_batch']==8 and stats['tokens']==count and stats['page_size']==128 and stats['lengths']==[len(r['prompt_ids']) for r in actual['inputs']]
    assert stats['pack_calls']==(count+127)//128 and len(stats['cached_pages'])<=2 and 0<stats['peak_cached_packed_logical_bytes']<=41943040

def replay_preserved(report,read,reference,read_reference):
    assert len(reference['cases'])==len(report['cases'])==9
    groups=[]
    for i,(label,limit) in enumerate((('short-mixed',8),('full4k-text',128),('full4k-image',128))):
        original=reference['cases'][i*3]
        assert original['group']==label and original['repeat']==0 and original['order']==list(range(8)) and original['actual_limit']==limit
        groups.append((label,original['inputs'],limit))
    baseline=replay_cases(reference,read_reference,groups);actual=replay_cases(report,read,groups);assert baseline==actual
    same_state(report['empty_state'],reference['empty_state'],read,read_reference)
    same_state(report['final_reset'],reference['final_reset'],read,read_reference)
    for a,b in zip(report['cases'],reference['cases']):same_case(a,b,read,read_reference)
    return {'status':'paged_batch8_retained_model_preserved','cases':actual,'complete_states':29,'raw_vectors':58,'all_retained_states_vectors_tokens_exact':True,
            'external_b1_b4_math_qualified':False,'mtp_qualified':False,'broad_quality_qualified':False,'http_qualified':False,'performance_admitted':False,'two_times_target_achieved':False}
