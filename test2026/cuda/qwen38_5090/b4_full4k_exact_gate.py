"""Full-context B4 admission; selection must come from the pinned 4K corpus loader."""
from .b4_b1_exact_gate import compare_capture,require_ids
from .b4_b1_short_m16_gate import PHASES,COUNTS

def observe(selection,reference,candidate,read_reference,read_candidate):
    assert selection['status']=='retained_fixed4k_B4_correctness_inputs_selected_not_executed'
    expected=selection['inputs'];groups=selection['groups'];assert len(expected)==8 and len(groups)==3
    assert reference['status']=='eight_B1_full4k_oracles_collected' and candidate['status']=='B4_full4k_raw_capture_needs_CPython_admission'
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
    assert [(row['label'],row['keys']) for row in candidate['rows']]==[(g['label'],g['keys']) for g in groups]
    results=[]
    for group,row in zip(groups,candidate['rows']):
        assert group['generated_IDs']==128 and group['total_context_capacity']==4096 and len(group['keys'])==len(set(group['keys']))==4
        keys=group['keys'];refs=[oracle[key] for key in keys]
        for key in keys:assert len(expected[key]['prompt_ids'])==3968
        comparisons={phase:compare_capture(row[phase],[r[phase] for r in refs],lambda name:read_candidate(group['label'],name),lambda index,name:read_reference(keys[index],name)) for phase in PHASES}
        actual=row['generated_ids'];gold=[r['generated_ids'][0] for r in refs]
        assert len(actual)==len(gold)==4
        for a,b in zip(actual,gold):assert len(a)==len(b)==128 and all(type(n) is int and 0<=n<248320 for n in a+b)
        results.append(dict(label=group['label'],keys=keys,comparisons=comparisons,IDs_exact=[a==b for a,b in zip(actual,gold)],all_states_and_vectors_exact=all(c['all_exact'] for c in comparisons.values())))
    return dict(status='B4_full4k_complete_state_vectors_and_IDs_observed',groups=results,independent_B1_oracles=8,full_phase_row_comparisons=48,components_per_row=128,all_states_and_vectors_exact=all(r['all_states_and_vectors_exact'] for r in results),all_128_IDs_exact=all(all(r['IDs_exact']) for r in results),numerical_tolerance_applied=False,MTP_quality_HTTP_or_performance_qualified=False)

def admit(selection,reference,candidate,read_reference,read_candidate):
    result=observe(selection,reference,candidate,read_reference,read_candidate)
    assert result['all_states_and_vectors_exact'],'Full4K B4 complete state/raw vectors differ from independent B1 oracles'
    oracle={row['keys'][0]:row for row in reference['rows']}
    for row in candidate['rows']:require_ids(row['generated_ids'],[oracle[key]['generated_ids'][0] for key in row['keys']],128)
    return {**result,'status':'B4_full4k_all_state_vectors_and_128_IDs_exact'}
