"""Parse actual plugin entry/exit observations; synthetic CPU controls are never model measurements."""
import json
PREFIX='GARNET_MODELOPT_INIT '
ROUTES={'fp8','nvfp4','fp8_m128','fp8_z128','fp8_z128_reconstructed','fp8_kv128'}
FIELDS={'id','phase','route','instance','rows','columns','policy','context_present','factory_called','result','exception'}
def events(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    return [json.loads(line[len(PREFIX):]) for line in raw.splitlines() if line.startswith(PREFIX)]
def replay(rows,*,expected_calls=None):
    opened={};closed=set();failures=[];routes={};factories=exceptions=cached=0
    for row in rows:
        assert set(row)==FIELDS and type(row['id']) is int and row['id']>0 and row['route'] in ROUTES and row['phase'] in ('begin','end')
        for name in ('rows','columns','policy','result'):assert type(row[name]) is int
        for name in ('context_present','factory_called','exception'):assert type(row[name]) is bool
        assert row['rows']>0 and row['columns']>0 and type(row['instance']) is str and row['instance'] and row['id'] not in closed
        if row['route']=='fp8_m128':assert (row['rows'],row['columns'],row['policy'])==(5120,6144,128)
        elif row['route']=='fp8_z128':assert (row['rows'],row['columns'],row['policy'])==(6144,5120,0x5a313238)
        elif row['route']=='fp8_z128_reconstructed':assert (row['rows'],row['columns'],row['policy'])==(6144,5120,0x52313238)
        elif row['route']=='fp8_kv128':assert (row['rows'],row['columns'],row['policy'])==(1024,5120,0x4b313238)
        elif row['route']=='fp8':assert row['policy'] in (0,1)
        else:assert row['policy'] in (0,1,2)
        if row['phase']=='begin':
            assert row['id'] not in opened and row['result']==-1 and row['factory_called'] is row['exception'] is False;opened[row['id']]=row
        else:
            assert row['id'] in opened;before=opened.pop(row['id'])
            for name in ('route','instance','rows','columns','policy'):assert row[name]==before[name]
            assert row['result'] in (0,1) and row['factory_called']==(not before['context_present'])
            assert row['result']==(0 if row['context_present'] and not row['exception'] else 1)
            if before['context_present']:assert row['context_present'] and not row['exception']
            routes[row['route']]=routes.get(row['route'],0)+1;factories+=int(row['factory_called']);exceptions+=int(row['exception']);cached+=int(before['context_present'])
            if row['result']:failures.append(row)
            closed.add(row['id'])
    assert not opened and closed and sorted(closed)==list(range(1,len(closed)+1))
    if expected_calls is not None:assert type(expected_calls) is int and len(closed)==expected_calls
    return {'status':'actual_plugin_initialization_entries_exits_replayed','initialize_calls':len(closed),'events':len(rows),'routes':routes,'factory_calls':factories,'cached_context_initializations':cached,'exception_returns':exceptions,'failures':failures,'actual_failed_plugin_route_observed':bool(failures),'arithmetic_or_quality_or_performance_qualified':False}
def cpu_contract(report,stderr,enabled):
    assert report=={'status':'actual_kv128_plugin_initialize_cpu_controls_passed','actual_initialize_calls':54,'factory_calls':42,'successes':18,'failures':36,'cached_clone_initializations':6,'unexpected_GPU_operator_calls':0,'trace_enabled':enabled,'model_or_performance_qualified':False}
    rows=events(stderr)
    if not enabled:assert stderr==b'' and rows==[];return {'trace_enabled':False,'initialize_calls':54,'factory_calls':42,'GPU_operator_calls':0}
    assert all(line.startswith(PREFIX.encode()) for line in stderr.splitlines());q=replay(rows,expected_calls=54)
    assert q['routes']=={name:9 for name in ROUTES} and q['factory_calls']==42 and q['cached_context_initializations']==12 and q['exception_returns']==24 and len(q['failures'])==36
    return {'trace_enabled':True,'initialize_calls':54,'factory_calls':42,'GPU_operator_calls':0,'trace_replay':q,'synthetic_factory_faults_only':True}
