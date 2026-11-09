"""Audit prepared decode geometry operator timing; never infer token throughput."""
from pathlib import Path
import collections,copy,hashlib,json,math,statistics,sys
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit2.json'
files={}
def require(v,m):
    if not v:raise ValueError(m)
def raw(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
SHAPES=[('gate',4096,14336),('up',4096,14336),('down',14336,4096)]
SELECTIONS=[(4,1,True),(4,8,True)]
NAMES=['warm-a2','warm-b2','pressure-a2','pressure-b2']
def verify(d,mode,order,qualified_resources):
    require(d['completed'] is True and d['mode']==mode and d['order']==order and d['rounds']==20 and d['warmups']==5,'wrong process/mode/order/count')
    require(d['contiguous_layout'] is True and d['all_pipelines_prepared_before_collection'] is True,'missing serving layout/preparation barrier')
    require(d['native_math']=='MSL3.1 fastMathEnabled=NO, FP32 operands/FMA/SIMD reduction','changed arithmetic policy')
    require(d['case_combinations']==6 and d['cases']==[] and d['fp64']==[] and d['complete_parent_word_comparisons']==0,'timing incorrectly mixed correctness observations')
    require(d['pressure_bytes']==(256*1024*1024 if mode=='pressure' else 0),'wrong pressure payload')
    require(d['after_close_allocated_bytes']<=d['baseline_allocated_bytes']+4*1024*1024,'buffer leak')
    require(d['no_runtime_integration'] is True and d['serving_speed_claim'] is False and d['goal_complete'] is False,'overstated scope')
    want={(indexed,k,n,k,n,active,variant) for _,k,n in SHAPES for indexed in [False,True] for active in [1,2,3,4] for variant in [0,4,9]};seen=set()
    for p in d['pipeline_resources']:
        key=(p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['variant']);require(key not in seen,'duplicate resource');seen.add(key)
        require(key in qualified_resources and p==qualified_resources[key],'timing pipeline differs from exact qualified profile')
    require(seen==want and d['prepared_pipeline_count']==len(want)==48,'missing full prepared pipeline family')
    samples=d['samples'];require(len(samples)==240,'incomplete measurements');at=0;last=0;values=collections.defaultdict(list)
    for name,k,n in SHAPES:
        for physical,mask,indexed in SELECTIONS:
            for round_ in range(20):
                for position in range(2):
                    e=samples[at];at+=1;index=(position+(round_+5)%2)%2
                    if ((round_+5)+order)&1:index=1-index
                    variant=[0,9][index]
                    require((e['layer'],e['shape'],e['inner'],e['columns'],e['physical_rows'],e['mask'],e['indexed'],e['variant'],e['round'],e['position'])==(15,name,k,n,physical,mask,indexed,variant,round_,position),'wrong geometry/sample order')
                    start,end,ms,wall=(e[q] for q in ['gpu_start_seconds','gpu_end_seconds','gpu_ms','wall_including_pressure_ms']);ps,pe=e['pressure_gpu_start_seconds'],e['pressure_gpu_end_seconds']
                    require(all(type(x) in (float,int) and math.isfinite(x) for x in [start,end,ms,wall,ps,pe]) and start>=last and end>start and ms>0 and wall>=ms and math.isclose(ms,1000*(end-start),rel_tol=2e-12,abs_tol=1e-9),'invalid/overlapping GPU clock or false duration')
                    if mode=='pressure':require(last<=ps<pe<=start and wall>=ms+1000*(pe-ps),'pressure command not distinct/preceding kernel')
                    else:require(ps==pe==0,'unexpected pressure in warm sample')
                    last=end;values[(name,physical,mask,indexed,variant)].append(ms)
    medians={key:statistics.median(v) for key,v in values.items()};require(all(len(v)==20 for v in values.values()),'sample cell differs')
    cells=[]
    for name,k,n in SHAPES:
        for physical,mask,indexed in SELECTIONS:
            control=medians[(name,physical,mask,indexed,0)]
            for variant in [9]:
                candidate=medians[(name,physical,mask,indexed,variant)];cells.append(dict(shape=name,physical_rows=physical,mask=mask,indexed=indexed,variant=variant,control_median_gpu_ms=control,candidate_median_gpu_ms=candidate,operator_ratio_control_over_candidate=control/candidate))
    c1=[]
    for mask in [1,8]:
        control=sum(medians[(name,4,mask,True,0)] for name,_,_ in SHAPES)
        for variant in [9]:
            candidate=sum(medians[(name,4,mask,True,variant)] for name,_,_ in SHAPES)
            c1.append(dict(mask=mask,variant=variant,three_projection_control_ms=control,three_projection_candidate_ms=candidate,operator_ratio_control_over_candidate=control/candidate,worst_individual_projection_ratio=min(medians[(name,4,mask,True,0)]/medians[(name,4,mask,True,variant)] for name,_,_ in SHAPES)))
    return dict(samples=240,measured_cells=12,medians=cells,c1_three_projection_sums=c1,after_close_allocated_bytes=d['after_close_allocated_bytes'])
