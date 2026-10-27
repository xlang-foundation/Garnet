"""Pure CPU layer comparison, conditional on every ordinary primary snapshot matching."""
from .b4_b1_exact_gate import _logical,_vectors
from .b4_full4k_decode_prefix import observe,same_profile_parent,STATUS
from .decode_boundary_analysis import blob,difference

LAYERS=tuple(range(64))
ROW_BYTES=4*5120*4

def primary_parity(report,parent,read,read_parent):
    for key in ('status','mode','physical_batch','capacity','prefill_chunk','decode_chunk','library_sha256','inputs'):
        assert report[key]==parent[key],('primary metadata drift',key)
    assert report['status']==STATUS
    result=same_profile_parent(report,parent,read,read_parent)
    assert [r['keys'] for r in report['rows']]==[r['keys'] for r in parent['rows']]
    batch=report['physical_batch'];snapshots=0
    for row,old in zip(report['rows'],parent['rows']):
        assert row['generated_ids']==old['generated_ids']
        assert len(row['post_step_states'])==len(old['post_step_states'])==7
        for step,(a,b) in enumerate(zip(row['post_step_states'],old['post_step_states']),1):
            assert a['step']==b['step']==step
            x,y=a['capture'],b['capture'];_logical(x['logical'],batch);_logical(y['logical'],batch)
            assert x['logical']==y['logical'],('observer changed primary state',row['keys'],step)
            assert _vectors(x['vectors'],read,batch)==_vectors(y['vectors'],read_parent,batch),('observer changed primary vector',row['keys'],step)
            snapshots+=1
    return dict(status='all_seven_primary_states_vectors_and_8_IDs_exact',mode=report['mode'],snapshots=snapshots,endpoint_parity=result,numerical_tolerance_applied=False,performance_qualified=False)

def layers(report,read):
    assert report['status']==STATUS and report['physical_batch'] in (1,4)
    batch=report['physical_batch'];profile=report['observer']
    assert profile['layers']==list(LAYERS) and profile['tokens']==4 and profile['hidden']==5120 and profile['hidden_width']==20480
    assert type(profile['primary_width']) is int and profile['primary_width']>0
    assert profile['wide_width']==profile['primary_width']+64*20480
    selected=profile['selected_steps'];assert isinstance(selected,list) and selected and selected==sorted(set(selected)) and all(type(s) is int and 1<=s<=7 for s in selected)
    assert report['boundary_runs']==len(report['rows'])*7
    result={}
    for group,row in enumerate(report['rows']):
        entries=row['layer_observations'];assert [e['step'] for e in entries]==selected
        for entry in entries:
            step=entry['step'];assert entry['program_runs']==group*7+step and entry['consumed_tokens']==[ids[step-1] for ids in row['generated_ids']]
            assert set(entry['boundaries'])=={str(i) for i in LAYERS}
            assert len({e['file'] for e in entry['boundaries'].values()})==64
            raw={layer:blob(entry['boundaries'][str(layer)],read,batch*ROW_BYTES) for layer in LAYERS}
            for position,key in enumerate(row['keys']):
                assert (step,key) not in result
                result[step,key]=dict(consumed_token=entry['consumed_tokens'][position],boundaries={layer:v[position*ROW_BYTES:(position+1)*ROW_BYTES] for layer,v in raw.items()})
    return result

def analyze(selection,reference,candidate,ordinary_reference,ordinary_candidate,read_reference,read_candidate,read_ordinary_reference,read_ordinary_candidate):
    parity={mode:primary_parity(report,parent,read,old_read) for mode,report,parent,read,old_read in (
        ('reference',reference,ordinary_reference,lambda n:read_reference(None,n),lambda n:read_ordinary_reference(None,n)),
        ('candidate',candidate,ordinary_candidate,read_candidate,read_ordinary_candidate))}
    original=observe(selection,reference,candidate,read_reference,read_candidate)
    assert reference['observer']['selected_steps']==candidate['observer']['selected_steps']
    r=layers(reference,lambda n:read_reference(None,n));c=layers(candidate,read_candidate)
    assert set(r)==set(c);rows=[]
    for step in candidate['observer']['selected_steps']:
        for position,key in enumerate(selection['groups'][2]['keys']):
            a,b=r[step,key],c[step,key];assert a['consumed_token']==b['consumed_token']
            diff={str(layer):difference(a['boundaries'][layer],b['boundaries'][layer],active_words=5120) for layer in LAYERS}
            rows.append(dict(step=step,physical_row=position,key=key,consumed_token=a['consumed_token'],all_consumed_IDs_through_step_exact=True,boundaries=diff,
                             first_differing_full_T4_layer=next((i for i in LAYERS if not diff[str(i)]['all_bytes_exact']),None),
                             first_differing_active_layer=next((i for i in LAYERS if diff[str(i)]['active_different_words']),None)))
    return dict(status='all64_layers_compared_after_seven_step_complete_primary_parity',primary_parity=parity,rows=rows,ordinary_cross_profile_state_observation=original,numerical_tolerance_applied=False,original_full4k_admission_still_failed=True,MTP_quality_HTTP_or_performance_qualified=False,
                scope='Selected ordinary decode steps, all64 completeT4hiddenarrays withactive/inactive words distinguished. Firstdifferent output is an observed boundary, notuniqueoperatorcausality or internalactivation capture. Original128IDfull4K remainsFAILED.')
