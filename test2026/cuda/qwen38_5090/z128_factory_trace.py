"""Replay actual constructor observations; CPU synthetic faults remain explicitly separate."""
import json,re
PREFIX='GARNET_Z128_FACTORY '
APIS=('cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion','cublasLtCreate','cublasLtMatmulDescCreate','cublasLtMatmulDescSetAttribute','cublasLtMatmulDescSetAttribute','cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutCreate','cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceSetAttribute','cublasLtMatmulAlgoGetHeuristic','cublasLtMatmulAlgoConfigGetAttribute','cublasLtMatmulAlgoCheck')
GUARDS=('device_or_version',)*3+('handle_or_operation',)*2+('operation_or_actual_layout',)*5+('canonical_layout_or_preference',)*3+('workspace_preference','heuristic_api','algorithm_ID','actual_algorithm_check')
MODES=('success','cached','major','minor','version','found0','all_failed','skip_failed','opaque','mismatch_then_expected','id','id_bytes','check_state','workspace')+tuple('fail_'+str(i) for i in range(1,18))+tuple('throw_'+str(i) for i in range(1,18))
POLICIES=(None,'','0','true','01',' 1','1 ','-1','2','1')

def events(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    return [json.loads(line[len(PREFIX):]) for line in raw.splitlines() if line.startswith(PREFIX)]

def replay(raw):
    rows=events(raw);groups={}
    for row in rows:
        assert set(row)=={'id','seq','event','name','a','b','c','opaque'}
        assert all(type(row[key]) is int for key in ('id','seq','a','b','c')) and row['id']>0 and row['seq']>0
        assert isinstance(row['name'],str) and isinstance(row['opaque'],str) and re.fullmatch(r'(?:[0-9a-f]{128})?',row['opaque'])
        assert row['event'] in ('begin','api','heuristic','candidate','selection','reject','accept','end')
        group=groups.setdefault(row['id'],[]);assert row['seq']==len(group)+1;group.append(row)
    assert set(groups)==set(range(1,len(groups)+1));result=[]
    for i,group in groups.items():
        assert group[0]['event']=='begin' and group[0]['name']=='factory' and group[-1]['event']=='end'
        assert sum(r['event']=='begin' for r in group)==sum(r['event']=='end' for r in group)==1
        api=[r for r in group if r['event']=='api'];names=[r['name'] for r in api];outcome=group[-1]['name'];assert outcome in ('empty','created','cached','exception')
        assert names==list(APIS[:len(names)]) and len(names)<=17
        rejection=[r for r in group if r['event']=='reject'];accepted=[r for r in group if r['event']=='accept']
        if outcome=='empty':assert len(rejection)==1 and not accepted
        elif outcome in ('created','cached'):assert not rejection and len(accepted)==1 and accepted[0]['name']==outcome and len(api)==(17 if outcome=='created' else 3)
        else:assert not rejection and not accepted
        candidates=[r for r in group if r['event']=='candidate'];found=[r for r in group if r['event']=='heuristic']
        if found:
            assert len(found)==1 and found[0]['name']=='found' and 0<=found[0]['a']<=8 and len(candidates)==found[0]['a']
            assert [r['a'] for r in candidates]==list(range(len(candidates))) and all(len(r['opaque'])==128 for r in candidates)
        else:assert not candidates
        result.append({'id':i,'outcome':outcome,'API_calls':len(api),'API_results':api,'reject':rejection[0] if rejection else None,'candidates':candidates,'events':len(group)})
    return {'status':'actual_Z128_factory_events_replayed','factory_calls':len(groups),'events':len(rows),'calls':result,'algorithm_substituted':False,'model_or_quality_or_performance_qualified':False}

def cpu_case(report,raw,mode,enabled):
    assert report['mode']==mode and report['unexpected_matmuls']==0
    success=mode in ('success','cached','skip_failed');thrown=mode.startswith('throw_')
    assert report['created'] is success and report['info_okay'] is success and report['cached'] is (mode=='cached') and report['thrown'] is thrown
    assert report['algorithm']==(67 if success else -1) and report['workspace']==0
    if not enabled:assert not events(raw);return {'status':'disabled_trace_silent_original_SDK_transcript_exact'}
    q=replay(raw);assert q['factory_calls']==(2 if mode=='cached' else 1);first=q['calls'][0]
    assert first['outcome']==('created' if success else 'exception' if thrown else 'empty')
    if mode.startswith(('fail_','throw_')):
        index=int(mode.split('_')[1]);assert first['API_calls']==index-(1 if thrown else 0)
        if not thrown:assert first['reject']['name']==GUARDS[index-1]
    elif mode in ('major','minor','version'):assert first['reject']['name']=='device_or_version'
    elif mode in ('opaque','mismatch_then_expected'):assert first['reject']['name']=='opaque_mismatch' and first['API_calls']==15
    elif mode in ('found0','all_failed'):assert first['reject']['name']=='no_supported_candidate_or_CUDART'
    elif mode in ('id','id_bytes'):assert first['reject']['name']=='algorithm_ID'
    elif mode in ('check_state','workspace'):assert first['reject']['name']=='actual_algorithm_check'
    return q
