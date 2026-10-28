"""Observe early complete-state drift without promoting the failed full4K gate."""
from .b4_b1_exact_gate import compare_capture,require_ids
from .b4_b1_short_m16_gate import COUNTS

PHASES=('reset_state','prefill_state','final_reset')
STATUS='full4k_first_seven_decode_steps_captured_unqualified'

def checked_ids(actual,expected,batch):
    assert type(batch) is int and batch in (1,4) and len(actual)==len(expected)==batch
    for a,b in zip(actual,expected):
        assert len(a)==len(b)==8 and all(type(n) is int and 0<=n<248320 for n in a+b) and a==b
    if batch==4:require_ids(actual,expected,8)

def same_profile_parent(report,parent,read,read_parent):
    assert report['physical_batch']==parent['physical_batch']
    assert report['inputs']==parent['inputs']
    originals={tuple(row['keys']):row for row in parent['rows']}
    results=[]
    for row in report['rows']:
        old=originals[tuple(row['keys'])]
        checked_ids(row['generated_ids'],[ids[:8] for ids in old['generated_ids']],report['physical_batch'])
        for phase in PHASES:
            assert row[phase]['logical']==old[phase]['logical']
            for kind in ('hidden','normalized'):
                a,b=row[phase]['vectors'][kind],old[phase]['vectors'][kind]
                x,y=read(a['file']),read_parent(b['file'])
                import hashlib
                assert len(x)==len(y)==a['bytes']==b['bytes']
                assert hashlib.sha256(x).hexdigest()==a['sha256']
                assert hashlib.sha256(y).hexdigest()==b['sha256'] and x==y
        results.append(dict(keys=row['keys'],first_8_IDs_exact=True,reset_prefill_final_reset_exact=True))
    return dict(status='same_profile_parent_endpoints_and_first8_IDs_exact',rows=results,
                intermediate_state_not_in_parent=True,performance_qualified=False)

def observe(selection,reference,candidate,read_reference,read_candidate):
    group=selection['groups'][2];keys=group['keys'];assert group['label']=='full4k-mixed' and len(set(keys))==4
    assert len(selection['inputs'])==8 and all(len(selection['inputs'][k]['prompt_ids'])==3968 for k in keys)
    for mode,report in (('reference',reference),('candidate',candidate)):
        batch=1 if mode=='reference' else 4
        assert report['status']==STATUS and report['inputs']==selection['inputs']
        assert (report['physical_batch'],report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(batch,4096,32,4)
        assert report['performance_admitted'] is report['MTP_quality_qualified'] is False
        assert set(report['graph_stats'])=={'4','32'}
        for size,stats in report['graph_stats'].items():
            for key,count in COUNTS.items():assert type(stats[key]) is int and stats[key]==(count if mode=='candidate' and size=='32' else 0)
        assert all(report['vision_stats'][k]==0 for k in COUNTS)
        for row in report['rows']:
            checked_ids(row['generated_ids'],row['generated_ids'],batch)
            assert len(row['generated_ids'])==batch and len(row['post_step_states'])==7
            for phase,slot in (('reset_state',0),('prefill_state',3968),('final_state',3975),('final_reset',0)):
                assert row[phase]['logical']['slots']==[slot]*batch
            assert row['reset_state']['logical']==row['final_reset']['logical']
            assert row['final_state']==row['post_step_states'][-1]['capture']
            for step,entry in enumerate(row['post_step_states'],1):
                assert entry['step']==step and entry['capture']['logical']['slots']==[3968+step]*batch
    assert [r['keys'] for r in reference['rows']]==[[k] for k in keys]
    assert len(candidate['rows'])==1 and candidate['rows'][0]['keys']==keys
    row=candidate['rows'][0];refs=reference['rows']
    def compare(capture,captures):return compare_capture(capture,captures,read_candidate,lambda i,n:read_reference(keys[i],n))
    endpoints={phase:compare(row[phase],[r[phase] for r in refs]) for phase in PHASES}
    assert all(v['all_exact'] for v in endpoints.values()),'Initial/reset endpoints must remain exact'
    require_ids(row['generated_ids'],[r['generated_ids'][0] for r in refs],8)
    steps=[dict(step=i+1,comparison=compare(entry['capture'],[r['post_step_states'][i]['capture'] for r in refs])) for i,entry in enumerate(row['post_step_states'])]
    return dict(status='early_full4k_decode_drift_observed_not_numerically_admitted',keys=keys,
                endpoint_comparisons=endpoints,steps=steps,
                first_nonexact_step=next((s['step'] for s in steps if not s['comparison']['all_exact']),None),
                components_per_row=128,first_8_IDs_exact=True,numerical_tolerance_applied=False,
                original_full4k_gate_still_failed=True,MTP_quality_HTTP_or_performance_qualified=False)
