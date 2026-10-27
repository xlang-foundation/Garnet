"""Parse printed SDK parameters without inferring unprinted algorithm attributes."""
import json,re
def parse(raw):
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    assert 'cuBLASLt v13.4 device #0:' in raw
    pids=set(re.findall(r'\[cublasLt\]\[(\d+)\]',raw));assert len(pids)==1
    pending=None;calls=[]
    for number,line in enumerate(raw.splitlines(),1):
        if '[Api][cublasLtMatmulAlgoGetHeuristic]' in line:
            assert pending is None
            m=re.search(r'Bdesc=\[type=R_8F_E4M3 rows=5120 cols=(32|128) ld=5120\]',line);assert m
            assert 'Adesc=[type=R_8F_E4M3 rows=5120 cols=1024 ld=5120]' in line and 'maxWorkspaceSizeinBytes=4194304' in line
            pending=dict(tokens=int(m[1]),line=number)
        elif '[Info][cublasLtMatmulAlgoGetHeuristic]' in line and 'heuristicResults=' in line:
            assert pending is not None and 'candidates' not in pending
            m=re.search(r'heuristicResults=\[([\d,]+)\]',line);assert m
            ids=[int(v) for v in m[1].split(',')];assert 1<=len(ids)<=8;pending['candidates']=ids
        elif '[Trace][cublasLtMatmul]' in line:
            assert pending is not None and 'candidates' in pending;t=pending['tokens']
            for text in ('Adesc=[type=R_8F_E4M3 rows=5120 cols=1024 ld=5120]',f'Bdesc=[type=R_8F_E4M3 rows=5120 cols={t} ld=5120]',f'Cdesc=[type=R_16BF rows=1024 cols={t} ld=1024]',f'Ddesc=[type=R_16BF rows=1024 cols={t} ld=1024]','computeType=COMPUTE_32F scaleType=R_32F transa=OP_T','workSpaceSizeInBytes=4194304 beta=0 outOfPlace=0'):assert text in line,text
            m=re.search(r' algo=\[([^\]]+)\]',line);assert m
            config=dict(re.findall(r'(\w+)=([^\s]+)',m[1]));required={'algoId','tile','stages','reductionScheme','numSplitsK'};assert required<=set(config)<=(required|{'ctaSwizzling'})
            for k in ('algoId','numSplitsK','ctaSwizzling'):
                if k in config:assert config[k].isdigit();config[k]=int(config[k])
            assert re.fullmatch(r'MATMUL_TILE_\d+x\d+',config['tile']) and re.fullmatch(r'MATMUL_STAGES_\d+x\d+',config['stages']) and config['reductionScheme']=='REDUCTION_SCHEME_COMPUTE_TYPE'
            assert config['algoId']==pending['candidates'][0]
            weight=re.search(r' A=(0X[0-9A-F]+) ',line);assert weight
            calls.append(dict(tokens=t,line=number,query_line=pending['line'],heuristic_candidates=pending['candidates'],printed_algorithm=config,weight_buffer=weight[1]));pending=None
    assert pending is None and {r['tokens'] for r in calls}=={32,128};profiles={}
    for t in (32,128):
        rows=[r for r in calls if r['tokens']==t];configs={json.dumps(r['printed_algorithm'],sort_keys=True) for r in rows};weights=sorted({r['weight_buffer'] for r in rows});assert len(rows)>=2 and len(weights)==2
        profiles[str(t)]=dict(matmul_trace_calls=len(rows),weight_buffers=weights,unique_printed_configurations=[json.loads(s) for s in sorted(configs)])
    return dict(status='actual_cuBLASLt_KV_selection_vendor_parameters_parsed',process_id=int(next(iter(pids))),calls=calls,profiles=profiles,unprinted_attributes_not_assumed=['CUSTOM_OPTION','INNER_SHAPE_ID','CLUSTER_SHAPE_ID','CTA_SWIZZLING when omitted'],chronology_scope='Line order in one vendor file; wall clock is not merged with Python stdout or other hosts',SDK_logging_is_observation_not_performance=True,full_model_or_numerical_or_performance_qualified=False)
