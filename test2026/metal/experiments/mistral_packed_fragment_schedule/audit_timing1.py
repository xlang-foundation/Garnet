"""Audit serial, balanced operator timings without claiming model speed."""
from pathlib import Path
import collections,copy,hashlib,itertools,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
need(read(ROOT/'correctness-audit1.json')['passed'] and read(ROOT/'timing1-postflight.json')['passed'],'missing admission/postflight')
for name,h in read(ROOT/'timing1-preflight.json')['files_sha256'].items():need(sha(Path(name))==h,'changed timing input '+name)
statuses=[read(ROOT/('timing-'+co+'1-status.json')) for co in ['a','b']]
need(all(s['status']=='terminal' and s['returncode']==0 for s in statuses),'unfinished GPU process')
need(statuses[1]['started_ns']>=statuses[0]['finished_ns'] and statuses[0]['pid']!=statuses[1]['pid'],'not serial fresh processes')
baseline=read(ROOT/'correctness1/result1.json')['pipeline_resources']
data={co:read(ROOT/('timing-'+co+'1/result1.json')) for co in ['a','b']}
keys=set(itertools.product(['gate','down','gate-prefix-proxy'],[128,256,512],[False,True],range(20),range(-1,4)))
def verify(d,wall_ns):
    need(d['completed'] and d['timing'] and d['cases_count']==0 and d['samples_count']==1800,'wrong timing mode/count')
    need(d['pipeline_resources']==baseline,'different qualified pipelines/resources')
    samples=d['samples'];actual={(e['shape'],e['rows'],e['sparse'],e['round'],e['mode']) for e in samples}
    need(len(samples)==len(actual)==1800 and actual==keys,'missing/duplicate cell/round/mode')
    positions=collections.defaultdict(collections.Counter)
    for e in samples:
        need(e['layer']==15 and math.isfinite(e['gpu_ms']) and e['gpu_ms']>0,'wrong/nonfinite clock or data layer')
        shape,rows=e['shape'],e['rows'];tile=128 if shape=='gate-prefix-proxy' else (256 if rows>=(384 if shape=='down' else 256) else 128)
        need(e['tile']==tile,'different native-selected tile')
        index=(e['position']+(e['round']+5)%5)%5
        if (e['round']+5)&1:index=4-index
        need(e['mode']==index-1 and 0<=e['position']<5,'wrong rotating/reversing order')
        positions[(shape,rows,e['sparse'],e['mode'])][e['position']]+=1
    need(all(c==collections.Counter({i:4 for i in range(5)}) for c in positions.values()),'sample positions not balanced')
    need(sum(e['gpu_ms'] for e in samples)<=wall_ns/1e6,'GPU durations exceed serial process wall time')
    result={}
    for shape,rows,sparse in itertools.product(['gate','down','gate-prefix-proxy'],[128,256,512],[False,True]):
        values={mode:[e['gpu_ms'] for e in samples if (e['shape'],e['rows'],e['sparse'],e['mode'])==(shape,rows,sparse,mode)] for mode in range(-1,4)}
        med={m:statistics.median(v) for m,v in values.items()}
        rounds={(e['round'],e['mode']):e['gpu_ms'] for e in samples if (e['shape'],e['rows'],e['sparse'])==(shape,rows,sparse)}
        result[shape+'-r'+str(rows)+'-'+('permuted' if sparse else 'contiguous')]=dict(shape=shape,rows=rows,sparse=sparse,median_gpu_ms={str(m):med[m] for m in med},range_gpu_ms={str(m):[min(v),max(v)] for m,v in values.items()},control_over_candidate={str(m):med[-1]/med[m] for m in range(4)},paired_round_ratio_median={str(m):statistics.median(rounds[(r,-1)]/rounds[(r,m)] for r in range(20)) for m in range(4)})
    return result
results={co:verify(data[co],statuses[i]['finished_ns']-statuses[i]['started_ns']) for i,co in enumerate(['a','b'])};rejected=[]
mutations={
    'missing_sample':lambda x:x['samples'].pop(),
    'duplicate_sample':lambda x:x['samples'].__setitem__(1,copy.deepcopy(x['samples'][0])),
    'wrong_tile':lambda x:x['samples'][0].update(tile=64),
    'wrong_order':lambda x:x['samples'][0].update(position=4),
    'nonfinite_duration':lambda x:x['samples'][0].update(gpu_ms=float('nan')),
    'negative_duration':lambda x:x['samples'][0].update(gpu_ms=-1),
    'wrong_qualified_resources':lambda x:x['pipeline_resources'][0].update(shared_bytes=0),
}
for name,mutate in mutations.items():
    d=copy.deepcopy(data['a']);mutate(d)
    try:verify(d,statuses[0]['finished_ns']-statuses[0]['started_ns'])
    except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
    else:raise ValueError('Corruption admitted: '+name)
largest={shape:{str(mode):[results[co][shape+'-r512-contiguous']['control_over_candidate'][str(mode)] for co in ['a','b']] for mode in range(4)} for shape in ['gate','down','gate-prefix-proxy']}
result=dict(passed=True,scope=__doc__,cohorts=results,largest_contiguous_control_over_candidate=largest,measured_samples=3600,warmup_samples=900,measured_rounds_per_mode_cell_process=20,each_mode_sample_positions={str(i):4 for i in range(5)},negative_controls_rejected=rejected,
    mode_names={'-1':'unaltered native packed kernel','0':'unchanged arithmetic plus disabled writer observer','1':'explicit RHS half4 loads','2':'scalar RHS K32 preload and per-accumulator K16 schedule','3':'vector RHS plus K32 scheduling'},
    numerical_gate_sha256=sha(ROOT/'correctness-audit1.json'),sample_result_sha256={co:sha(ROOT/('timing-'+co+'1/result1.json')) for co in ['a','b']},preflight_sha256=sha(ROOT/'timing1-preflight.json'),
    selection_changed=False,reason='Largest real gate/down cells do not support a replacement: vector-only is essentially flat and K32 variants regress. Small-row/proxy gains are retained, not promoted to full-model claims.',
    timings_are_operator_gpu_durations_only=True,absolute_gpu_start_end_timestamps_retained=False,proxy_is_actual_QO=False,baseline_per_operator_kernel_selection_proven=False,native_runtime_changed=False,serving_speed_claim=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit1.json';need(not dest.exists(),'Never overwrite audit');dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cohorts'},indent=2))
