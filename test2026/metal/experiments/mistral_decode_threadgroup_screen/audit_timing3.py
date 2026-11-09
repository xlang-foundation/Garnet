"""Audit prepared decode geometry operator timing; never infer token throughput."""
from pathlib import Path
import collections,copy,hashlib,json,math,statistics,sys
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit3.json'
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
SELECTIONS=[(1,1,False),(4,15,False),(4,1,True),(4,8,True),(4,3,True),(4,7,True),(4,15,True)]
NAMES=['warm-a2','warm-b2','pressure-a2','pressure-b2']
def verify(d,mode,order,qualified_resources):
    require(d['completed'] is True and d['mode']==mode and d['order']==order and d['rounds']==20 and d['warmups']==5,'wrong process/mode/order/count')
    require(d['contiguous_layout'] is True and d['all_pipelines_prepared_before_collection'] is True,'missing serving layout/preparation barrier')
    require(d['native_math']=='MSL3.1 fastMathEnabled=NO, FP32 operands/FMA/SIMD reduction','changed arithmetic policy')
    require(d['case_combinations']==21 and d['cases']==[] and d['fp64']==[] and d['complete_parent_word_comparisons']==0,'timing incorrectly mixed correctness observations')
    require(d['pressure_bytes']==(256*1024*1024 if mode=='pressure' else 0),'wrong pressure payload')
    require(d['after_close_allocated_bytes']<=d['baseline_allocated_bytes']+4*1024*1024,'buffer leak')
    require(d['no_runtime_integration'] is True and d['serving_speed_claim'] is False and d['goal_complete'] is False,'overstated scope')
    want={(indexed,k,n,k,n,active,groups) for _,k,n in SHAPES for indexed in [False,True] for active in [1,2,3,4] for groups in [0,2,4,8]};seen=set()
    for p in d['pipeline_resources']:
        key=(p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['groups']);require(key not in seen,'duplicate resource');seen.add(key)
        require(key in qualified_resources and p==qualified_resources[key],'timing pipeline differs from exact qualified profile')
    require(seen==want and d['prepared_pipeline_count']==len(want)==64,'missing full prepared pipeline family')
    samples=d['samples'];require(len(samples)==1260,'incomplete measurements');at=0;last=0;values=collections.defaultdict(list)
    for name,k,n in SHAPES:
        for physical,mask,indexed in SELECTIONS:
            for round_ in range(20):
                for position in range(3):
                    e=samples[at];at+=1;index=(position+(round_+5)%3)%3
                    if ((round_+5)+order)&1:index=2-index
                    groups=[2,4,8][index]
                    require((e['layer'],e['shape'],e['inner'],e['columns'],e['physical_rows'],e['mask'],e['indexed'],e['groups'],e['round'],e['position'])==(15,name,k,n,physical,mask,indexed,groups,round_,position),'wrong geometry/sample order')
                    start,end,ms,wall=(e[q] for q in ['gpu_start_seconds','gpu_end_seconds','gpu_ms','wall_including_pressure_ms']);ps,pe=e['pressure_gpu_start_seconds'],e['pressure_gpu_end_seconds']
                    require(all(type(x) in (float,int) and math.isfinite(x) for x in [start,end,ms,wall,ps,pe]) and start>=last and end>start and ms>0 and wall>=ms and math.isclose(ms,1000*(end-start),rel_tol=2e-12,abs_tol=1e-9),'invalid/overlapping GPU clock or false duration')
                    if mode=='pressure':require(last<=ps<pe<=start and wall>=ms+1000*(pe-ps),'pressure command not distinct/preceding kernel')
                    else:require(ps==pe==0,'unexpected pressure in warm sample')
                    last=end;values[(name,physical,mask,indexed,groups)].append(ms)
    medians={key:statistics.median(v) for key,v in values.items()};require(all(len(v)==20 for v in values.values()),'sample cell differs')
    cells=[]
    for name,k,n in SHAPES:
        for physical,mask,indexed in SELECTIONS:
            control=medians[(name,physical,mask,indexed,4)]
            for groups in [2,8]:
                candidate=medians[(name,physical,mask,indexed,groups)];cells.append(dict(shape=name,physical_rows=physical,mask=mask,indexed=indexed,groups=groups,control_median_gpu_ms=control,candidate_median_gpu_ms=candidate,operator_ratio_control_over_candidate=control/candidate))
    c1=[]
    for mask in [1,8]:
        control=sum(medians[(name,4,mask,True,4)] for name,_,_ in SHAPES)
        for groups in [2,8]:
            candidate=sum(medians[(name,4,mask,True,groups)] for name,_,_ in SHAPES)
            c1.append(dict(mask=mask,groups=groups,three_projection_control_ms=control,three_projection_candidate_ms=candidate,operator_ratio_control_over_candidate=control/candidate,worst_individual_projection_ratio=min(medians[(name,4,mask,True,4)]/medians[(name,4,mask,True,groups)] for name,_,_ in SHAPES)))
    return dict(samples=1260,measured_cells=63,medians=cells,c1_three_projection_sums=c1,after_close_allocated_bytes=d['after_close_allocated_bytes'])
def main():
    require(sha(ROOT/'audit_timing2.py')=='e8b380a1eb129aa28f40c5f34fc81761084b00d02b0085b9fe2a91dc60be4335','changed frozen auditor')
    require(not DEST.exists(),'never overwrite audit');status=read(ROOT/'timing2-status.json');require(status['status']=='terminal' and status['returncode']==0 and len(status['processes'])==4,'unfinished collection')
    require(read(ROOT/'timing2-postflight.json')['passed'],'missing postflight')
    for name,h in read(ROOT/'timing2-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed bound input '+name)
    prior_end=0
    for name,p in zip(NAMES,status['processes']):require(p['name']==name and p['status']=='terminal' and p['returncode']==0 and p['started_ns']>=prior_end and p['finished_ns']>p['started_ns'],'overlapping/failed native processes');prior_end=p['finished_ns']
    correctness=read(ROOT/'correctness-audit1.json');require(correctness['passed'],'unqualified arithmetic')
    resources=read(ROOT/'correctness2/result1.json')['pipeline_resources'];qualified={(p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['groups']):p for p in resources}
    policy=read(ROOT/'timing-policy2.json');require(policy['control_groups']==4 and policy['candidates']==[2,8] and policy['full_preparation_barrier_required'],'changed declared decision')
    data={name:read(ROOT/name/'result1.json') for name in NAMES};reports={name:verify(data[name],'pressure' if name.startswith('pressure') else 'warm',int(name.endswith('b2')),qualified) for name in NAMES}
    rejected=[]
    def event(d):return d['samples'][0]
    mutations={
        'missing_sample':lambda d:d['samples'].pop(),
        'wrong_native_group':lambda d:event(d).update(groups=16),
        'wrong_order':lambda d:event(d).update(position=2),
        'changed_active_rows':lambda d:event(d).update(mask=0),
        'false_gpu_rate':lambda d:event(d).update(gpu_ms=1e-12),
        'nonfinite_clock':lambda d:event(d).update(gpu_start_seconds=float('nan')),
        'pressure_overlap':lambda d:event(d).update(pressure_gpu_end_seconds=event(d)['gpu_end_seconds']+1),
        'pressure_omitted':lambda d:event(d).update(pressure_gpu_start_seconds=0,pressure_gpu_end_seconds=0),
        'wrong_pressure_payload':lambda d:d.update(pressure_bytes=1),
        'wrong_warmup_count':lambda d:d.update(warmups=0),
        'partial_profile_preparation':lambda d:d.update(all_pipelines_prepared_before_collection=False),
        'missing_prepared_profile':lambda d:d['pipeline_resources'].pop(),
        'wrong_compiled_stride':lambda d:d['pipeline_resources'][0].update(input_stride=1),
        'wrong_math':lambda d:d.update(native_math='relaxed'),
        'retained_buffer':lambda d:d.update(after_close_allocated_bytes=1<<40),
    }
    for name,mutate in mutations.items():
        d=copy.deepcopy(data['pressure-a2']);mutate(d)
        try:verify(d,'pressure',0,qualified)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corrupt timing admitted '+name)
    decisions={}
    for groups in [2,8]:
        rows=[r for p in reports.values() for r in p['c1_three_projection_sums'] if r['groups']==groups]
        decisions[str(groups)]=dict(qualified_for_native_integration_screen=all(r['operator_ratio_control_over_candidate']>=1.03 and r['worst_individual_projection_ratio']>=1/1.02 for r in rows),minimum_c1_three_projection_ratio=min(r['operator_ratio_control_over_candidate'] for r in rows),maximum_c1_three_projection_ratio=max(r['operator_ratio_control_over_candidate'] for r in rows),worst_individual_projection_ratio=min(r['worst_individual_projection_ratio'] for r in rows))
    for p in [ROOT/'audit_timing2.py',ROOT/'host-derivation3.json',ROOT/'compile3-command.json']:raw(p)
    result=dict(passed=True,scope=__doc__,processes=reports,measured_samples=5040,warmup_dispatches=1260,negative_controls_rejected=rejected,decisions=decisions,files_sha256=files,full_preparation_before_collection=True,earlier_timing1_retained_but_excluded=True,pressure_does_not_prove_all_hardware_caches_cold=True,native_runtime_integrated=False,token_throughput_measured=False,serving_speed_claim=False,coding_quality_qualified=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','processes']},indent=2))
if __name__=='__main__':main()
