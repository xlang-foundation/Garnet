"""Selected-row step52: distinguish decoder-state drift from token selection."""
from .b4_b1_exact_gate import _logical,_vectors
from .b4_b1_short_m16_gate import COUNTS
STATUS='selected_full4k_step7_step52_and53_IDs_captured_unqualified'
STEPS=(7,52)
PHASES=('reset_state','prefill_state','final_reset')

def first_mixed_difference(selection,reference,candidate):
    keys=selection['groups'][2]['keys'];assert selection['groups'][2]['label']=='full4k-mixed'
    refs={r['keys'][0]:r['generated_ids'][0] for r in reference['rows']}
    matches=[r for r in candidate['rows'] if r['keys']==keys];assert len(matches)==1
    row=matches[0];indices=[]
    for position,key in enumerate(keys):
        a,b=row['generated_ids'][position],refs[key]
        assert len(a)==len(b)==128 and all(type(v) is int and 0<=v<248320 for v in a+b)
        indices.append(next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None))
    first=min(i for i in indices if i is not None);assert first==52
    position=indices.index(first)
    return dict(group='full4k-mixed',first_differing_ID_indices=indices,selected_key=keys[position],physical_row=position,zero_based_unconsumed_ID_index=first)

def ids(actual,expected,batch,limit):
    assert type(batch) is int and batch in (1,4) and len(actual)==len(expected)==batch
    for a,b in zip(actual,expected):
        assert len(a)==len(b)==limit and all(type(n) is int and 0<=n<248320 for n in a+b) and a==b

def same_profile_full_parent(report,parent,read,read_parent):
    batch=report['physical_batch'];assert batch==parent['physical_batch'] and report['inputs']==parent['inputs']
    original={tuple(r['keys']):r for r in parent['rows']};results=[]
    for row in report['rows']:
        old=original[tuple(row['keys'])];ids(row['generated_ids'],[v[:53] for v in old['generated_ids']],batch,53)
        for phase in PHASES:
            _logical(row[phase]['logical'],batch);_logical(old[phase]['logical'],batch)
            assert row[phase]['logical']==old[phase]['logical']
            assert _vectors(row[phase]['vectors'],read,batch)==_vectors(old[phase]['vectors'],read_parent,batch)
        results.append(dict(keys=row['keys'],first53_IDs_exact=True,reset_prefill_final_reset_exact=True))
    return dict(status='same_profile_full_parent_endpoints_and53_IDs_exact',rows=results,performance_qualified=False)

def same_profile_step7(report,parent,read,read_parent):
    batch=report['physical_batch'];assert batch==parent['physical_batch'] and report['inputs']==parent['inputs']
    original={tuple(r['keys']):r for r in parent['rows']}
    for row in report['rows']:
        old=original[tuple(row['keys'])];a,b=row['post_step_states'][0],old['post_step_states'][6]
        assert a['step']==b['step']==7
        x,y=a['capture'],b['capture'];_logical(x['logical'],batch);_logical(y['logical'],batch)
        assert x['logical']==y['logical'] and _vectors(x['vectors'],read,batch)==_vectors(y['vectors'],read_parent,batch)
        ids([v[:8] for v in row['generated_ids']],old['generated_ids'],batch,8)
    return dict(status='all_current_step7_primary_rows_exact_against_qualified_prefix_parent',physical_batch=batch,rows=len(report['rows']),components_per_row=128,performance_qualified=False)

def compare_selected(actual,reference,position,read_actual,read_reference):
    assert type(position) is int and 0<=position<4
    x,y=actual['logical'],reference['logical'];_logical(x,4);_logical(y,1)
    a,b=_vectors(actual['vectors'],read_actual,4),_vectors(reference['vectors'],read_reference,1)
    changed=[i for i,(u,v) in enumerate(zip(x['components'],y['components'])) if u['row_sha256'][position]!=v['row_sha256'][0]]
    vectors={kind:a[kind][position*20480:(position+1)*20480]==b[kind] for kind in a}
    result=dict(physical_row=position,slot_exact=x['slots'][position]==y['slots'][0],physical_hash_exact=x['physical_row_sha256'][position]==y['physical_row_sha256'][0],logical_hash_exact=x['logical_row_sha256'][position]==y['logical_row_sha256'][0],different_component_indices=changed,finite_raw_vector_bytes_exact=vectors,components_per_row=128,numerical_tolerance_applied=False)
    result['all_exact']=result['slot_exact'] and result['physical_hash_exact'] and result['logical_hash_exact'] and not changed and all(vectors.values());return result

def observe(selection,selected_key,reference,candidate,read_reference,read_candidate):
    keys=selection['groups'][2]['keys'];assert selection['groups'][2]['label']=='full4k-mixed' and len(set(keys))==4 and selected_key in keys
    position=keys.index(selected_key);assert [r['keys'] for r in reference['rows']]==[[selected_key]] and [r['keys'] for r in candidate['rows']]==[keys]
    for mode,report in (('reference',reference),('candidate',candidate)):
        batch=1 if mode=='reference' else 4
        assert report['status']==STATUS and report['inputs']==selection['inputs']
        assert (report['physical_batch'],report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(batch,4096,32,4)
        assert report['performance_admitted'] is report['MTP_quality_qualified'] is False
        assert set(report['graph_stats'])=={'4','32'}
        for size,stats in report['graph_stats'].items():
            for name,count in COUNTS.items():assert type(stats[name]) is int and stats[name]==(count if mode=='candidate' and size=='32' else 0)
        assert all(report['vision_stats'][k]==0 for k in COUNTS)
        for row in report['rows']:
            assert all(len(selection['inputs'][k]['prompt_ids'])==3968 for k in row['keys'])
            ids(row['generated_ids'],row['generated_ids'],batch,53)
            assert [v['step'] for v in row['post_step_states']]==list(STEPS)
            assert row['final_state']==row['post_step_states'][-1]['capture']
            assert row['reset_state']['logical']==row['final_reset']['logical']
            for phase,slot in (('reset_state',0),('prefill_state',3968),('final_state',4020),('final_reset',0)):
                assert row[phase]['logical']['slots']==[slot]*batch
            for entry in row['post_step_states']:assert entry['capture']['logical']['slots']==[3968+entry['step']]*batch
    r,c=reference['rows'][0],candidate['rows'][0];a,b=c['generated_ids'][position],r['generated_ids'][0]
    assert a[:52]==b[:52],'All52 consumed tokens must match before interpreting step52'
    comp=lambda x,y:compare_selected(x,y,position,read_candidate,lambda n:read_reference(selected_key,n))
    endpoints={phase:comp(c[phase],r[phase]) for phase in PHASES};assert all(v['all_exact'] for v in endpoints.values())
    checkpoints=[dict(step=step,comparison=comp(c['post_step_states'][i]['capture'],r['post_step_states'][i]['capture'])) for i,step in enumerate(STEPS)]
    assert checkpoints[0]['comparison']['all_exact'],'Step7 must remain independently exact'
    exact52=checkpoints[1]['comparison']['all_exact'];different=a[52]!=b[52]
    return dict(status='selected_step52_state_and_unconsumed_prediction_observed',selected_key=selected_key,physical_row=position,endpoint_comparisons=endpoints,checkpoints=checkpoints,all52_consumed_IDs_exact=True,reference_prediction_52=b[52],candidate_prediction_52=a[52],prediction_52_differs=different,step52_state_hidden_normalized_exact=exact52,token_selection_path_differs_with_exact_decoder_state=exact52 and different,original_full4k_admission_still_failed=True,MTP_quality_HTTP_or_performance_qualified=False,numerical_tolerance_applied=False,
                scope='One selectedrow: complete128statecomponents/physical-logicalhash/finitehidden-normalizedvectors after7and52steps. IDs0..51 consumed identically; differingID52 is unconsumed. Exactdecoderstate withdifferentID isolates token-selectionpath, notuniqueinternalheadoperator. OtherB1step52states unobserved; no full4K/quality/performanceadmission.')
