"""Same-DLL flagoff parity and flagon selected-row numerical intervention."""
from .b4_b1_exact_gate import _logical,_vectors,compare_capture
from .b4_full4k_selection52 import ids,observe,compare_selected,same_profile_step7
from .b4_k35_cache52 import checked_cache,compare_cache
CAPTURES=('reset_state','prefill_state','final_state','final_reset')

def control(report,parent,read,read_parent,new_library,old_library):
    assert report['library_sha256']==new_library and parent['library_sha256']==old_library
    assert report['intervention_policy']=='off'
    for key in ('status','mode','physical_batch','capacity','prefill_chunk','decode_chunk','inputs'):assert report[key]==parent[key]
    assert report['physical_batch']==4 and [r['keys'] for r in report['rows']]==[r['keys'] for r in parent['rows']]
    assert len(report['rows'])==1;row,old=report['rows'][0],parent['rows'][0];ids(row['generated_ids'],old['generated_ids'],4,53)
    assert [v['step'] for v in row['post_step_states']]==[v['step'] for v in old['post_step_states']]==[7,52]
    assert row['final_state']==row['post_step_states'][-1]['capture'] and old['final_state']==old['post_step_states'][-1]['capture']
    pairs=[(row[k],old[k]) for k in CAPTURES]+[(row['post_step_states'][0]['capture'],old['post_step_states'][0]['capture'])]
    for a,b in pairs:
        _logical(a['logical'],4);_logical(b['logical'],4);assert a['logical']==b['logical'] and _vectors(a['vectors'],read,4)==_vectors(b['vectors'],read_parent,4)
    a=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],4,2,read)
    b=checked_cache(old['key35_cache52'],old['final_state']['logical']['components'][70],4,2,read_parent)
    assert a==b
    return dict(status='same_DLL_flagoff_all_four_rows_53IDs_five_complete_captures_and_raw_cache_exact',new_library_sha256=new_library,original_library_sha256=old_library,complete_captures=5,components_per_row=128,raw_cache_bytes=len(a),numeric_tolerance_applied=False,model_or_performance_qualified=False)

def intervention(selection,key,reference,report,full_reference,prefix_candidate,read_reference,read,read_full,read_prefix):
    assert report['intervention_policy']=='on'
    observation=observe(selection,key,reference,report,lambda k,n:read_reference(n),read)
    early=same_profile_step7(report,prefix_candidate,read,read_prefix)
    row=report['rows'][0];keys=selection['groups'][2]['keys'];assert row['keys']==keys and keys[2]==key
    refs={r['keys'][0]:r for r in full_reference['rows']};assert set(keys)<=set(refs)
    endpoints={phase:compare_capture(row[phase],[refs[k][phase] for k in keys],read,lambda i,n:read_full(n)) for phase in ('reset_state','prefill_state','final_reset')}
    assert all(c['all_exact'] for c in endpoints.values())
    gold=[refs[k]['generated_ids'][0][:53] for k in keys]
    for a,b in zip(row['generated_ids'],gold):assert len(a)==len(b)==53 and a[:52]==b[:52],'All four consumed trajectories must match before interpretation'
    ref=reference['rows'][0]
    a=checked_cache(ref['key35_cache52'],ref['final_state']['logical']['components'][70],1,0,read_reference)
    b=checked_cache(row['key35_cache52'],row['final_state']['logical']['components'][70],4,2,read)
    cache=compare_cache(a,b)
    ids_exact=[a==b for a,b in zip(row['generated_ids'],gold)]
    passed=observation['step52_state_hidden_normalized_exact'] and not observation['prediction_52_differs'] and all(ids_exact) and cache['all_bits_exact']
    return dict(status='same_DLL_flagon_selected_step52_intervention_observed',early_all_four_rows=early,all_four_endpoint_comparisons=endpoints,selected_row_observation=observation,raw_key_cache=cache,all_four_53_IDs_exact=ids_exact,selected_step52_intervention_passed=passed,full128_output_4K_MTP_quality_HTTP_or_performance_qualified=False,original_full4k_still_failed=True,two_times_target_achieved=False,scope='Immutable source60 B1/full4K/prefix oracles, same-DLL source85 off/on intervention. All4rows endpoints/step7/53IDs checked; only selected row has an independent step52 complete-state/raw-key oracle. Not full128-output/4K or MTP admission.')
