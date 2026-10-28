"""Independent complete production-dispatch raw replay; no timing admission.

Archive/source/tool identity must additionally be checked by the enclosing run
verifier. A successful raw replay alone never qualifies a runtime DLL or model.
"""
import hashlib,json
from pathlib import Path
from integrated_top1_oracle import SHAPES,expected_rows

def replay(raw,fixture,enabled):
    assert type(enabled) is bool and fixture['status']=='integrated_top1_independent_fixtures_predeclared_gpu_unrun'
    keys=[(d,r,v,p) for d in ('fp32','bf16') for r,v in SHAPES for p in range(11)]
    frozen={}
    for c in fixture['cases']:
        key=tuple(c[n] for n in ('dtype','rows','vocab','pattern'))
        assert key not in frozen
        frozen[key]=expected_rows(*key)
        assert frozen[key]==(c['ids'],c['score_bits'])
    assert list(frozen)==keys and len(keys)==fixture['cases_per_process']==352
    records=[json.loads(line) for line in raw.splitlines()]
    assert len(records)==1787
    start={'kind':'dispatch_start','parallel_enabled':enabled,'legacy_host_calls':0,'parallel_host_calls':0,'environment_changed_after_snapshot':True}
    end={'kind':'dispatch_end','parallel_enabled':enabled,'legacy_host_calls':594 if enabled else 1060,
         'parallel_host_calls':466 if enabled else 0,'counter_scope':'host dispatch attempts including capture construction; excludes graph replays'}
    summary={'kind':'summary','status':'integrated_batch_top1_correctness_passed','cases':352,'graphs':2,'rejected_calls':12,'performance_admitted':False}
    assert records[0]==start and records[-2:]==[end,summary]
    starts=[];cases=[];normal=[];extras=[];graphs=[];rejections=[];observations=0
    for record in records[1:-2]:
        kind=record['kind']
        if kind in ('case_started','observation','case'):
            key=tuple(record[n] for n in ('dtype','rows','vocab','pattern'));assert key in frozen
            ids,bits=frozen[key];rows=key[1]
        if kind=='case_started':starts.append(key)
        elif kind=='case':
            assert record['ids']==ids and record['score_bits']==bits and record['repeats']==3
            assert record['cpu_legacy_candidate_exact'] and record['readonly'] and record['canaries'];cases.append(key)
        elif kind=='observation':
            assert record['cpu_ids']==ids and record['cpu_score_bits']==bits
            for field in ('candidate_ids','legacy_ids','candidate_score_bits','legacy_score_bits','cpu_ids','cpu_score_bits'):
                assert all(type(x) is int for x in record[field])
            for mode in ('candidate','legacy'):
                gpu_ids=record[mode+'_ids'];gpu_bits=record[mode+'_score_bits']
                assert len(gpu_ids)==len(gpu_bits)==rows+4 and gpu_ids[2:-2]==ids and gpu_bits[2:-2]==bits
                assert all(gpu_ids[i]==0x5a5a5a5a5a5a5a5a and gpu_bits[i]==0x5a5a5a5a for i in (0,1,rows+2,rows+3))
            item=(key,record['phase'],record['repeat']);(normal if record['phase']=='normal' else extras).append(item);observations+=1
        elif kind=='graph':
            assert record=={'kind':'graph','dtype':record['dtype'],'rows':4,'vocab':248320,'replays':3,'exact':True,'nullable_score_exact':True}
            graphs.append(record['dtype'])
        elif kind=='rejection':
            assert record['status']==record['expected']==1
            rejections.append((record['dtype'],record['label']))
        else:raise AssertionError('Unexpected record kind')
    assert starts==cases==keys
    assert normal==[(key,'normal',r) for key in keys for r in range(3)]
    expected_extras=[]
    for dtype in ('fp32','bf16'):
        key=(dtype,4,248320,0)
        expected_extras.append((key,'nullable_score',0));expected_extras.extend((key,'graph',r) for r in range(3))
        expected_extras.append(((dtype,4,248320,10),'after_rejections',0))
    assert extras==expected_extras and observations==fixture['raw_observations_per_process']==1066
    assert graphs==['fp32','bf16']
    labels=['dimensions-0-248320','dimensions--1-248320','dimensions-4-0','dimensions-4--1','null-input','null-ids']
    assert rejections==[(d,label) for d in ('fp32','bf16') for label in labels]
    # Derive routing independently from the declared shape inventory.
    parallel=(sum(3*11*2 for r,v in SHAPES if 1<=r<=128 and v>=8192)+4) if enabled else 0
    assert end['parallel_host_calls']==parallel and end['legacy_host_calls']==3*len(keys)+4-parallel
    return {'status':'integrated_top1_raw_observations_replayed','parallel_enabled':enabled,'cases':len(cases),
            'raw_observations':observations,'observations_sha256':hashlib.sha256(raw).hexdigest(),
            'legacy_host_calls':end['legacy_host_calls'],'parallel_host_calls':parallel,'graphs':len(graphs),
            'invalid_calls':len(rejections),'model_qualified':False,'performance_admitted':False}

if __name__=='__main__':
    import sys
    assert len(sys.argv)==5 and sys.argv[3] in ('0','1')
    result=replay(Path(sys.argv[1]).read_bytes(),json.loads(Path(sys.argv[2]).read_bytes()),sys.argv[3]=='1')
    Path(sys.argv[4]).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
