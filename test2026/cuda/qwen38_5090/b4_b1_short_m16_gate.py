"""Original complete-state/word/ID gates run under CPython, plus actual route proof."""
import json,re
from .b4_b1_exact_gate import compare_capture,require_ids
PHASES=('reset_state','prefill_state','final_state','final_reset')
COUNTS={'native_fp8_m128_canonical_projections':48,'native_fp8_z128_canonical_projections':0,'native_fp8_z128_reconstructed_projections':48,'native_fp8_kv128_canonical_projections':32,'native_fp8_q128_canonical_projections':16,'native_fp8_o128_canonical_projections':16,'native_full_chunk_stage_witnesses':0}
FACTORY_APIS=['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion','cublasLtCreate','cublasLtMatmulDescCreate']+['cublasLtMatmulDescSetAttribute']*2+['cublasLtMatrixLayoutCreate']*5+['cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceSetAttribute','cublasLtMatmulAlgoGetHeuristic']+['cublasLtMatmulAlgoConfigGetAttribute']*7+['cublasLtMatmulAlgoCheck']*2

def observe(reference,candidate,read_reference,read_candidate):
    assert reference['status']=='four_B1_short_oracles_collected' and candidate['status']=='B4_short_mixed_tail_raw_capture_needs_CPython_admission'
    assert reference['inputs']==candidate['inputs'] and [len(v['prompt_ids']) for v in reference['inputs']]==[41,104,93,93]
    assert [v['physical_rows'] for v in reference['rows']]==[[0],[1],[2],[3]] and [v['physical_rows'] for v in candidate['rows']]==[[0,1,2,3]]
    for mode,report in (('reference',reference),('candidate',candidate)):
        assert report['physical_batch']==(1 if mode=='reference' else 4) and (report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(4096,32,4)
        assert report['performance_admitted'] is report['full4k_MTP_quality_qualified'] is False and set(report['graph_stats'])=={'4','32'}
        for t,stats in report['graph_stats'].items():
            for key,count in COUNTS.items():assert type(stats[key]) is int and stats[key]==(count if mode=='candidate' and t=='32' else 0)
        assert all(report['vision_stats'][key]==0 for key in COUNTS)
        for row in report['rows']:
            assert row['reset_state']['logical']==row['final_reset']['logical'] and all(row['reset_state']['vectors'][key]['sha256']==row['final_reset']['vectors'][key]['sha256'] for key in ('hidden','normalized'))
    actual=candidate['rows'][0];comparisons={phase:compare_capture(actual[phase],[r[phase] for r in reference['rows']],read_candidate,lambda row,name:read_reference(name)) for phase in PHASES}
    ids=actual['generated_ids'];gold=[r['generated_ids'][0] for r in reference['rows']]
    assert len(ids)==len(gold)==4
    for a,b in zip(ids,gold):assert len(a)==len(b)==8 and all(type(n) is int and 0<=n<248320 for n in a+b)
    return dict(status='B4_M16_short_complete_state_and_word_observation',comparisons=comparisons,IDs_exact=[a==b for a,b in zip(ids,gold)],all_states_and_vectors_exact=all(v['all_exact'] for v in comparisons.values()),full4k_MTP_805_HTTP_or_performance_qualified=False)

def admit(reference,candidate,read_reference,read_candidate):
    result=observe(reference,candidate,read_reference,read_candidate)
    assert result['all_states_and_vectors_exact'],'B4 complete state/raw vector mismatch against four independent B1 oracles'
    require_ids(candidate['rows'][0]['generated_ids'],[v['generated_ids'][0] for v in reference['rows']],8)
    return {**result,'status':'B4_M16_short_complete_state_words_and_IDs_exact'}

def route_trace(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8',errors='replace')
    factory=[json.loads(line.split('GARNET_GDNM16_FACTORY ',1)[1]) for line in raw.splitlines() if line.startswith('GARNET_GDNM16_FACTORY ')]
    initialization=[json.loads(line.split('GARNET_MODELOPT_INIT ',1)[1]) for line in raw.splitlines() if line.startswith('GARNET_MODELOPT_INIT ')]
    groups={}
    for row in factory:assert set(row)=={'id','seq','event','name','a','b','c','opaque'};groups.setdefault(row['id'],[]).append(row)
    assert len(groups)>=2;created=[];cached=0
    for rows in groups.values():
        assert [r['seq'] for r in rows]==list(range(1,len(rows)+1)) and rows[0]['event']=='begin' and rows[-1]['event']=='end'
        assert rows[-1]['name'] in ('created','cached')
        apis=[r for r in rows if r['event']=='api']
        if rows[-1]['name']=='cached':assert len(apis)==3 and [r['name'] for r in apis]==['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion'] and [r['a'] for r in apis]==[0,0,130400];cached+=1;continue
        assert [r['name'] for r in apis]==FACTORY_APIS and all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in apis)
        configs=[r for r in rows if r['event']=='configuration'];assert [(r['name'],r['a'],r['b'],r['c']) for r in configs]==[(n,v,4,v) for n,v in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),(35,13,17,3,2,0,1))]
        checks=[r for r in rows if r['event']=='check'];assert [r['name'] for r in checks]==['M4','M16'] and all(r['a']==0 and 0<=r['b']<=4<<20 for r in checks)
        selected=[r for r in rows if r['event']=='selection'];assert len(selected)==1 and selected[0]['name']=='observed_M4_control' and re.fullmatch('[0-9a-f]{128}',selected[0]['opaque'])
        eligible=[r for r in rows if r['event']=='candidate' and r['a']==selected[0]['a']];assert len(eligible)==1 and eligible[0]['b']==0 and eligible[0]['opaque']==selected[0]['opaque'];created.append(dict(configuration=[r['a'] for r in configs],checks=checks,opaque_live_SDK_selection_preserved=True))
    routes=[r for r in initialization if r['route']=='fp8_gdn_m16'];assert routes and len(routes)%2==0;pairs={}
    for row in routes:pairs.setdefault(row['id'],[]).append(row)
    counts={10240:0,6144:0}
    for rows in pairs.values():
        assert len(rows)==2 and [r['phase'] for r in rows]==['begin','end'];a,b=rows;assert a['instance']==b['instance'] and a['rows']==b['rows'] and a['columns']==b['columns']==5120 and a['policy']==b['policy']==16 and b['rows'] in counts and b['context_present'] and not b['exception'] and b['result']==0;counts[b['rows']]+=1
    assert all(n>=48 for n in counts.values())
    return dict(status='actual_B4T4_GDN_M16_route_and_live_SDK_selection_replayed',created_contexts=len(created),cached_factory_calls=cached,initializations_by_output_rows={str(n):v for n,v in counts.items()},created=created,all_seven_attributes_checked=True,original_4MiB_workspace_preserved=True)
