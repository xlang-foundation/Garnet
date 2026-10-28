"""Full-context B4 admission; selection must come from the pinned 4K corpus loader."""
from .b4_b1_exact_gate import compare_capture,require_ids
from .b4_padding_lifecycle_inputs import select_lifecycle
from .b4_public_event_gate import replay_public
import base64,hashlib
from .b4_b1_short_m16_gate import PHASES,COUNTS

def observe(selection,reference,candidate,read_reference,read_candidate):
    assert selection==select_lifecycle(selection['inputs'])
    expected=selection['inputs'];groups=selection['groups'];assert len(expected)==8 and len(groups)==5
    assert reference['status']=='eight_B1_full4k_oracles_collected' and candidate['status']=='B4_padding_lifecycle_raw_capture_needs_CPython_admission'
    assert reference['inputs']==candidate['inputs']==expected
    for mode,report in (('reference',reference),('candidate',candidate)):
        assert (report['physical_batch'],report['capacity'],report['prefill_chunk'],report['decode_chunk'])==((1 if mode=='reference' else 4),4096,32,4)
        assert report['performance_admitted'] is report['MTP_quality_qualified'] is False and set(report['graph_stats'])=={'4','32'}
        for t,stats in report['graph_stats'].items():
            for key,count in COUNTS.items():assert type(stats[key]) is int and stats[key]==(count if mode=='candidate' and t=='32' else 0)
        assert all(report['vision_stats'][key]==0 for key in COUNTS)
        for row in report['rows']:
            assert len(row['generated_ids'])==report['physical_batch']
            assert row['reset_state']['logical']==row['final_reset']['logical']
            for key in ('hidden','normalized'):assert row['reset_state']['vectors'][key]['sha256']==row['final_reset']['vectors'][key]['sha256']
            for phase,slot in (('reset_state',0),('prefill_state',3968),('final_state',4095),('final_reset',0)):
                assert row[phase]['logical']['slots']==[slot]*report['physical_batch']
    assert [row['keys'] for row in reference['rows']]==[[key] for key in expected]
    oracle={row['keys'][0]:row for row in reference['rows']};assert len(oracle)==8
    assert [(row['label'],row['keys'],row['real_keys'],row['real_rows'],row['public_ids']) for row in candidate['rows']]==[(g['label'],g['keys'],g['real_keys'],g['real_rows'],g['public_ids']) for g in groups]
    results=[]
    for group,row in zip(groups,candidate['rows']):
        assert type(group['real_rows']) is type(row['real_rows']) is int
        assert group['generated_IDs']==128 and group['total_context_capacity']==4096 and len(group['keys'])==4 and len(group['real_keys'])==len(set(group['real_keys']))==group['real_rows']
        keys=group['keys'];refs=[oracle[key] for key in keys]
        for key in keys:assert len(expected[key]['prompt_ids'])==3968
        comparisons={phase:compare_capture(row[phase],[r[phase] for r in refs],lambda name:read_candidate(group['label'],name),lambda index,name:read_reference(keys[index],name)) for phase in PHASES}
        actual=row['generated_ids'];gold=[r['generated_ids'][0] for r in refs]
        assert len(actual)==len(gold)==4
        for a,b in zip(actual,gold):assert len(a)==len(b)==128 and all(type(n) is int and 0<=n<248320 for n in a+b)
        results.append(dict(label=group['label'],keys=keys,comparisons=comparisons,IDs_exact=[a==b for a,b in zip(actual,gold)],all_states_and_vectors_exact=all(c['all_exact'] for c in comparisons.values())))
    return dict(status='B4_full4k_complete_state_vectors_and_IDs_observed',groups=results,independent_B1_oracles=8,full_phase_row_comparisons=80,components_per_row=128,all_states_and_vectors_exact=all(r['all_states_and_vectors_exact'] for r in results),all_128_IDs_exact=all(all(r['IDs_exact']) for r in results),numerical_tolerance_applied=False,MTP_quality_HTTP_or_performance_qualified=False)

def admit(selection,reference,candidate,read_reference,read_candidate):
    result=observe(selection,reference,candidate,read_reference,read_candidate)
    assert result['all_states_and_vectors_exact'],'Full4K B4 complete state/raw vectors differ from independent B1 oracles'
    oracle={row['keys'][0]:row for row in reference['rows']}
    public=[]
    stops=candidate['default_stop_ids'];assert len(stops)==len(set(stops))==2 and all(type(t) is int for t in stops)
    for index,row in enumerate(candidate['rows']):
        require_ids(row['generated_ids'],[oracle[key]['generated_ids'][0] for key in row['keys']],128)
        assert row['public_event_file']=='case'+str(index)+'-public.jsonl'
        raw=read_candidate(row['label'],row['public_event_file']);assert hashlib.sha256(raw).hexdigest()==row['public_event_sha256']
        keys=row['real_keys'];inputs=selection['inputs'];jobs=[dict(id=key,body=inputs[source]['body']) for key,source in zip(row['public_ids'],keys)]
        images=[]
        for key in keys:
            content=inputs[key]['body']['messages'][0]['content']
            images.append(None if isinstance(content,str) else hashlib.sha256(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)).hexdigest())
        public.append(replay_public(raw,jobs,[inputs[k]['prompt_ids'] for k in keys],[oracle[k]['generated_ids'][0] for k in keys],stops,images,row['speculative_metrics']))
    assert sum(p['public_ID_values'] for p in public)==1664
    return {**result,'status':'B4_padding_lifecycle_all_physical_state_vectors_IDs_and_public_wire_exact','public_streams':public,'physical_ID_values':2560,'public_ID_values':1664,'HTTP_quality_or_performance_qualified':False}
