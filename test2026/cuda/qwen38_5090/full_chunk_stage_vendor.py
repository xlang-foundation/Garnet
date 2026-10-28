"""Logger-only full-stage parity and actual printed Q/Gate vendor observations."""
import json,math,re
from .full_chunk_stage_layout import STAGES
from .full_chunk_stage_replay import replay,frames,array

def weight_metadata(row):
    expected=dict(logical_shape=[12288,5120],packed_shape=[12288,1280],packed_dtype='INT32',packed_bytes=62914560,recipe='modelopt_fp8_w8a8',scale_bytes=0,scale_shape=[],joined_mlp=False)
    for k,v in expected.items():assert type(row[k]) is type(v) and row[k]==v,(k,row[k])
    for k in ('weight_scale','input_scale'):assert type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0

def stage_parity(current,prior,read,oldread):
    assert current['tokens']==prior['tokens'];left=frames(current,read);right=frames(prior,oldread);assert len(left)==len(right)
    count=words=0
    for a,b in zip(left,right):
        assert (a['start'],a['count'])==(b['start'],b['count'])
        for name,width in STAGES:
            x=array(a['arrays'][name],read);y=array(b['arrays'][name],oldread);assert x==y,('Vendor logging changed raw stage',current['tokens'],a['start'],name);count+=1;words+=a['count']*width
    assert current['readonly_observations']==prior['readonly_observations']
    return dict(complete_stage_entries=count,complete_F32_words=words,all_bytes_exact=True,complete_readonly_records_exact=True)

def parse_qg(raw,tokens):
    assert tokens in (32,128)
    if isinstance(raw,bytes):raw=raw.decode('utf8')
    assert 'cuBLASLt v13.4 device #0:' in raw
    pids=set(re.findall(r'\[cublasLt\]\[(\d+)\]',raw));assert len(pids)==1
    calls=[]
    for number,line in enumerate(raw.splitlines(),1):
        if '[Trace][cublasLtMatmul]' not in line or 'Adesc=[type=R_8F_E4M3 rows=5120 cols=12288 ld=5120]' not in line:continue
        m=re.search(r'Bdesc=\[type=R_8F_E4M3 rows=5120 cols=(\d+) ld=5120\]',line);assert m
        if int(m[1])!=tokens:continue
        for value in (f'Cdesc=[type=R_16BF rows=12288 cols={tokens} ld=12288]',f'Ddesc=[type=R_16BF rows=12288 cols={tokens} ld=12288]','computeType=COMPUTE_32F scaleType=R_32F transa=OP_T','workSpaceSizeInBytes=4194304 beta=0 outOfPlace=0'):assert value in line,value
        a=re.search(r' algo=\[([^\]]+)\]',line);weight=re.search(r' A=(0X[0-9A-F]+) ',line);assert a and weight
        config=dict(re.findall(r'(\w+)=([^\s]+)',a[1]));assert {'algoId','tile','stages','reductionScheme','numSplitsK'}<=set(config)
        calls.append(dict(line=number,tokens=tokens,weight_buffer=weight[1],printed_algorithm=config))
    assert calls,('No actual target QG trace',tokens)
    return dict(process_id=int(next(iter(pids))),geometry=dict(N=12288,K=5120,M=tokens),matmul_trace_calls=len(calls),weight_buffers=sorted({r['weight_buffer'] for r in calls}),unique_printed_configurations=[json.loads(s) for s in sorted({json.dumps(r['printed_algorithm'],sort_keys=True) for r in calls})],calls=calls,unprinted_attributes_not_assumed=True,heuristic_to_trace_association_not_inferred=True)

def compare(reports,reads,prior,oldreads,proofs,logs):
    assert set(reports)==set(reads)==set(logs)=={'stages32','stages128'}
    composed={**prior,**reports};composed_reads={**oldreads,**reads};arrays=replay(composed,composed_reads,proofs);parity={};vendor={}
    for phase,t in (('stages32',32),('stages128',128)):
        weight_metadata(reports[phase]['qg_weight_metadata']);parity[phase]=stage_parity(reports[phase],prior[phase],reads[phase],oldreads[phase]);vendor[phase]=parse_qg(logs[phase],t)
    assert reports['stages32']['qg_weight_metadata']==reports['stages128']['qg_weight_metadata']
    return dict(status='actual_full_chunk_QG_vendor_logging_exact_stage_primary_parity',arrays=arrays,stage_parity=parity,QG_weight_metadata=reports['stages32']['qg_weight_metadata'],vendor=vendor,inherited_defaultoff_controls=['control32','control128'],fresh_logged_profiles=['stages32','stages128'],algorithm_cause_established=False,model_or_performance_qualified=False,two_times_target_achieved=False)
