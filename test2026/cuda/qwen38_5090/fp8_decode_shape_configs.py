"""Replay actual SDK configuration queries; eligibility is not numerical proof."""
from collections import Counter
import re
GEOMETRIES=(('GDN_QKV',10240,5120),('GDN_Z',6144,5120),('FA_KV',1024,5120),('FA_Q',12288,5120),('FA_O',5120,6144))
ATTRIBUTES=('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION')
KNOWN=(35,13,17,3,2,0,1)

def replay(report,gpu_uuid):
    assert report['status']=='actual_live_FP8_M4_M16_geometry_configurations_collected_no_matmul'
    assert report['device']==0 and report['compute_capability']==[12,0] and report['cublasLt_version']==130400
    assert report['uuid_hex']==gpu_uuid.removeprefix('GPU-').replace('-','').lower()
    assert report['runtime_version']==report['CUDART_VERSION']>=12080 and report['driver_version']>=12080
    assert report['cublasLt_DLL'].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublaslt64_13.dll'
    assert (report['weight_dtype'],report['activation_dtype'],report['output_dtype'],report['compute_dtype'],report['TRANSA'],report['TRANSB'])==('E4M3','E4M3','BF16','FP32','T','N')
    assert report['matmul_executed'] is report['model_loaded'] is report['performance_qualified'] is False
    entries=report['geometries'];assert [(r['label'],r['N'],r['K'],r['M']) for r in entries]==[(label,n,k,m) for label,n,k in GEOMETRIES for m in (4,16)]
    checked=[];valid=0;rows=[]
    for row in entries:
        assert row['workspace_limit']==4<<20 and type(row['found']) is int and 1<=row['found']<=8 and row['found']==len(row['choices'])
        for i,choice in enumerate(row['choices']):
            assert choice['index']==i and type(choice['state']) is int and re.fullmatch('[0-9a-f]{128}',choice['opaque']) and type(choice['heuristic_workspace']) is int and choice['heuristic_workspace']>=0
            if choice['state']:
                assert choice['configuration']==choice['checks']==[];continue
            valid+=1;assert len(choice['configuration'])==7 and all(type(n) is int for n in choice['configuration'])
            assert [r['M'] for r in choice['checks']]==[4,16]
            for check in choice['checks']:
                assert all(type(check[k]) is int for k in ('status','state','workspace')) and check['workspace']>=0;checked.append(check['status'])
    for label,n,k in GEOMETRIES:
        a,b=[r for r in entries if r['label']==label]
        def selected(r):return next((c for c in r['choices'] if c['state']==0),None)
        x,y=selected(a),selected(b)
        eligible=x is not None and all(c['status']==c['state']==0 and c['workspace']<=4<<20 for c in x['checks'])
        rows.append(dict(label=label,N=n,K=k,M4_configuration=x['configuration'] if x else None,M16_configuration=y['configuration'] if y else None,M4_and_M16_configurations_equal=x['configuration']==y['configuration'] if x and y else None,live_M4_eligible_at_M4_and_M16=eligible,M4_checks=x['checks'] if x else None,known_M4_reference_matches=(tuple(x['configuration'])==KNOWN if x else False) if label in ('GDN_QKV','GDN_Z','FA_KV') else None))
    events=report['API_events'];assert all(set(r)=={'name','status'} and type(r['status']) is int for r in events)
    assert all(r['status']==0 for r in events if r['name']!='cublasLtMatmulAlgoCheck')
    assert [r['status'] for r in events if r['name']=='cublasLtMatmulAlgoCheck']==checked
    expected=Counter({'cudaGetDevice':1,'cudaGetDeviceProperties':1,'cudaDriverGetVersion':1,'cudaRuntimeGetVersion':1,'cublasLtCreate':1,'cublasLtDestroy':1,'cublasLtMatmulDescCreate':5,'cublasLtMatmulDescDestroy':5,'cublasLtMatmulDescSetAttribute:TRANSA':5,'cublasLtMatmulDescSetAttribute:TRANSB':5,'cublasLtMatmulPreferenceCreate':5,'cublasLtMatmulPreferenceDestroy':5,'cublasLtMatmulPreferenceSetAttribute':5,'cublasLtMatrixLayoutDestroy':25,'cublasLtMatmulAlgoGetHeuristic':10,'cublasLtMatmulAlgoConfigGetAttribute':valid*7,'cublasLtMatmulAlgoCheck':valid*2,**{'cublasLtMatrixLayoutCreate:'+name:5 for name in ('W','X4','Y4','X16','Y16')}})
    assert Counter(r['name'] for r in events)==expected
    return dict(status='actual_live_FP8_decode_configuration_queries_replayed_no_matmul',rows=rows,heuristic_candidates=sum(r['found'] for r in entries),valid_candidates=valid,actual_AlgoChecks=len(checked),API_events=len(events),all_known_M4_references_match=all(r['known_M4_reference_matches'] for r in rows[:3]),numerical_or_model_or_performance_qualified=False,scope='Only live SDK descriptor/algorithm/workspace eligibility. No tensor inputs, matmul, fullmodel or benchmark executed. Use fullconfiguration, not algorithmID alone; actualmatrix and unchanged modelgates required before any policy expansion or timing.')
