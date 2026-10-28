"""Replay separate constructor API configuration and explicit CPU/SDK scope."""
import json,re
PREFIX='GARNET_Z128_RECONSTRUCT '
FIELDS=(('ID',67,4),('TILE_ID',13,4),('STAGES_ID',17,4),('SPLITK_NUM',1,4),('REDUCTION_SCHEME',0,4),('CTA_SWIZZLING',1,4),('CUSTOM_OPTION',18,4),('INNER_SHAPE_ID',0,2),('CLUSTER_SHAPE_ID',0,2))
APIS=('cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion','cublasLtCreate','cublasLtMatmulDescCreate','cublasLtMatmulDescSetAttribute','cublasLtMatmulDescSetAttribute')+('cublasLtMatrixLayoutCreate',)*5+('cublasLtMatmulAlgoInit',)+('cublasLtMatmulAlgoConfigSetAttribute',)*8+('cublasLtMatmulAlgoConfigGetAttribute',)*9+('cublasLtMatmulAlgoCheck',)*2
GUARDS=('device_or_version',)*3+('handle_or_operation',)*2+('operation_or_actual_layout',)*5+('canonical_layout',)*2+('algorithm_init',)+('configuration_set',)*8+('configuration_get',)*9+('algorithm_check_M32','algorithm_check_M128')
MODES=('success','cached','major','minor','version')+tuple('fail_'+str(i) for i in range(1,33))+tuple('throw_'+str(i) for i in range(1,33))+tuple('get_value_'+str(i) for i in range(9))+tuple('get_bytes_'+str(i) for i in range(9))+('check_state_32','check_state_128','workspace_32','workspace_128')
POLICIES=(None,'','0','true','01',' 1','1 ','-1','2','1')
def events(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    return [json.loads(line[len(PREFIX):]) for line in raw.splitlines() if line.startswith(PREFIX)]
def replay(raw):
    rows=events(raw);groups={}
    for row in rows:
        assert set(row)=={'id','seq','event','name','a','b','c','opaque'} and all(type(row[k]) is int for k in ('id','seq','a','b','c')) and row['id']>0
        assert re.fullmatch(r'(?:[0-9a-f]{128})?',row['opaque']) and isinstance(row['name'],str)
        assert row['event'] in ('begin','api','configuration','algorithm','check','reject','accept','end')
        group=groups.setdefault(row['id'],[]);assert row['seq']==len(group)+1;group.append(row)
    assert set(groups)==set(range(1,len(groups)+1));calls=[]
    for identity,group in groups.items():
        assert group[0]['event']=='begin' and group[-1]['event']=='end' and sum(r['event']=='begin' for r in group)==sum(r['event']=='end' for r in group)==1
        api=[r for r in group if r['event']=='api'];assert [r['name'] for r in api]==list(APIS[:len(api)]) and len(api)<=32
        outcome=group[-1]['name'];assert outcome in ('created','cached','empty','exception');reject=[r for r in group if r['event']=='reject'];accept=[r for r in group if r['event']=='accept']
        config=[r for r in group if r['event']=='configuration'];algorithm=[r for r in group if r['event']=='algorithm'];checks=[r for r in group if r['event']=='check']
        assert [r['name'] for r in config]==[field[0] for field in FIELDS[:len(config)]] and len(config)<=9
        if outcome=='created':
            assert len(api)==32 and len(config)==9 and len(algorithm)==1 and len(algorithm[0]['opaque'])==128 and [(r['name'],r['a']) for r in checks]==[('M32',0),('M128',0)] and all(0<=r['b']<=4194304 for r in checks)
            assert [(r['name'],r['a'],r['b']) for r in config]==list(FIELDS) and all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in api)
        if outcome in ('created','cached'):assert not reject and len(accept)==1 and accept[0]['name']==outcome
        elif outcome=='empty':assert len(reject)==1 and not accept
        else:assert not reject and not accept
        if outcome=='cached':assert len(api)==3 and not config and not algorithm and not checks
        calls.append({'id':identity,'outcome':outcome,'API_calls':len(api),'API_results':api,'configuration':config,'algorithm':algorithm,'checks':checks,'reject':reject[0] if reject else None,'events':len(group)})
    return {'status':'separate_API_reconstruction_factory_events_replayed','factory_calls':len(calls),'events':len(rows),'calls':calls,'opaque_semantics_or_numerical_equivalence_established':False,'model_or_quality_or_performance_qualified':False}
def cpu_case(report,raw,mode,enabled,unsupported=False):
    success=not unsupported and mode in ('success','cached');thrown=not unsupported and mode.startswith('throw_');assert report['mode']==mode and report['created'] is success and report['info_okay'] is success and report['cached'] is (success and mode=='cached') and report['thrown'] is thrown and report['algorithm']==(67 if success else -1) and report['workspace']==0 and report['unexpected_matmuls']==0
    if not enabled:assert events(raw)==[];return {'status':'reconstruction_trace_disabled_silent'}
    q=replay(raw);assert q['factory_calls']==(2 if mode=='cached' and not unsupported else 1);row=q['calls'][0];assert row['outcome']==('created' if success else 'exception' if thrown else 'empty')
    if unsupported:assert row['API_calls']==0 and row['reject']['name']=='unsupported_CUDART'
    elif mode.startswith(('fail_','throw_')):
        index=int(mode.split('_')[1]);assert row['API_calls']==index-(1 if thrown else 0)
        if not thrown:assert row['reject']['name']==GUARDS[index-1]
    elif mode.startswith(('get_value_','get_bytes_')):assert row['API_calls']==22+int(mode.split('_')[-1]) and row['reject']['name']=='configuration_get'
    elif mode in ('major','minor','version'):assert row['reject']['name']=='device_or_version'
    elif mode.endswith('_32'):assert row['API_calls']==31 and row['reject']['name']=='algorithm_check_M32'
    elif mode.endswith('_128'):assert row['API_calls']==32 and row['reject']['name']=='algorithm_check_M128'
    return q

def sdk_case(report,raw,enabled):
    assert report['status']=='actual_separate_Z128_reconstruction_constructor_collected'
    assert report['created'] is report['cached'] is report['info_okay'] is report['cached_opaque_exact'] is True
    assert report['device']==0 and report['algorithm']==67 and 0<=report['workspace']<=4194304 and report['cublaslt_version']==130400 and report['cuda_error']==0 and re.fullmatch('[0-9a-f]{128}',report['opaque'])
    assert report['GPU_matmul_or_model_executed'] is report['numerical_equivalence_established'] is False
    assert [(r['name'],r['value'],r['written']) for r in report['configuration']]==list(FIELDS) and all(r['status']==0 and r['requested']==r['written'] for r in report['configuration'])
    if not enabled:assert events(raw)==[];return {'status':'SDK_explicit_factory_trace_unset_silent','public_configuration':report['configuration'],'numerical_equivalence_established':False}
    q=replay(raw);assert q['factory_calls']==2 and [r['outcome'] for r in q['calls']]==['created','cached'] and q['calls'][0]['algorithm'][0]['opaque']==report['opaque'] and q['calls'][0]['checks'][-1]['b']==report['workspace'];return q
