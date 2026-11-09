"""Audit balanced fresh-process FP32 gate/up/SiLU GPU spans; not model throughput."""
from pathlib import Path
import collections,copy,hashlib,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit1.json';files={}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def check(report,cohort,readonly):
    need(report['completed'] and report['mode']==cohort['mode'] and report['order']==cohort['order'] and not report['padded'] and report['candidate_selected']==2,'wrong/unfinished cohort')
    need(report['rounds']==20 and report['warmups']==5 and report['case_combinations']==3 and len(report['samples'])==120 and not report['cases'] and not report['fp64'],'wrong timing/warmup coverage')
    need(report['pipelines_prepared_before_collection'] and report['pipeline_count']==len(report['pipelines'])==24,'late/missing pipeline preparation')
    need(all(p['simd_width']==32 and p['max_threads']>=128 and p['threadgroup_bytes']==0 for p in report['pipelines']),'invalid native resource profile')
    need(report['readonly_checks']==[readonly],'timing input/weight parent identity changed')
    need(report['pressure_bytes']==(268435456 if cohort['mode']=='pressure' else 0),'wrong pressure allocation')
    keys=set();samples=collections.defaultdict(list);previous_end=0
    for sample in report['samples']:
        case=sample['case'];need(case in [0,1,2] and sample['mask']==[1,8,15][case] and sample['layer']==15,'wrong actual native occupancy/physical row')
        need(type(sample['round']) is int and 0<=sample['round']<20 and sample['method'] in [-1,2] and sample['position'] in [0,1],'warmup or wrong method counted')
        key=case,sample['round'],sample['method'];need(key not in keys,'duplicate timing sample');keys.add(key)
        expected_method=2 if (sample['position']^cohort['order']) else -1;need(sample['method']==expected_method,'pair order differs from declared cohort')
        start,end=sample['gpu_start_seconds'],sample['gpu_end_seconds'];gpu=sample['gpu_ms'];wall=sample['wall_including_pressure_ms']
        need(all(type(x) in [int,float] and math.isfinite(x) for x in [start,end,gpu,wall]) and end>start>0 and start>=previous_end and wall>=gpu>0,'invalid/overlapping GPU or wall interval')
        need(math.isclose(gpu,(end-start)*1000,rel_tol=2e-12,abs_tol=1e-9),'GPU duration not bound to raw command clock');previous_end=end
        a,b=sample['pressure_gpu_start_seconds'],sample['pressure_gpu_end_seconds']
        need((0<a<b<=start) if cohort['mode']=='pressure' else a==b==0,'pressure included in measured span or missing')
        samples[sample['mask'],sample['method']].append(gpu)
    need(keys=={(case,round,method) for case in range(3) for round in range(20) for method in [-1,2]},'unbalanced complete sample family')
    need(report['after_close_allocated_bytes']<=report['baseline_allocated_bytes']+4194304,'retained native GPU resources')
    cells={}
    for mask in [1,8,15]:
        control=statistics.median(samples[mask,-1]);candidate=statistics.median(samples[mask,2])
        cells[str(mask)]=dict(control_gpu_median_ms=control,candidate_gpu_median_ms=candidate,control_over_candidate=control/candidate,samples_per_method=20,actual_active_rows=mask.bit_count())
    return cells
def main():
    need(not DEST.exists(),'never overwrite audit');pre=read(ROOT/'timing1-preflight.json');post=read(ROOT/'timing1-postflight.json');policy=read(ROOT/'timing-policy1.json')
    need(pre['passed'] and pre['source_asset_hashing_during_collection'] is False and post['passed'] and post['all_owned_processes_terminal'],'unfinished or contaminated collection')
    for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed timing input '+name)
    correctness=read(ROOT/'correctness-audit1.json');need(correctness['passed'] and correctness['eligible_methods']==[2],'no exact candidate')
    original=read(ROOT/'contiguous1/result1.json');readonly=next(e for e in original['readonly_checks'] if e['layer']==15 and e['columns']==14336)
    data={};reports={};previous=0;pids=set()
    for cohort in policy['cohorts']:
        name=cohort['name'];status=read(ROOT/(name+'-status.json'));command=read(ROOT/(name+'-command.json'))
        need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=previous and status['pid'] not in pids,'probes not fresh/serial/completed');previous=status['finished_ns'];pids.add(status['pid'])
        need(command[-5:]==[cohort['mode'],str(cohort['order']),'20','contiguous','2'],'wrong fixed benchmark arguments')
        data[name]=read(ROOT/name/'result1.json');reports[name]=check(data[name],cohort,readonly)
    rejected=[];cohort=policy['cohorts'][0];source=data[cohort['name']]
    mutations={
        'missing_sample':lambda r:r['samples'].pop(),
        'duplicated_sample':lambda r:r['samples'].__setitem__(1,copy.deepcopy(r['samples'][0])),
        'wrong_occupancy':lambda r:r['samples'][0].update(mask=3),
        'wrong_pair_position':lambda r:r['samples'][0].update(position=1-r['samples'][0]['position']),
        'warmup_as_measurement':lambda r:r['samples'][0].update(round=-1),
        'false_gpu_duration':lambda r:r['samples'][0].update(gpu_ms=1e-6),
        'negative_wall_interval':lambda r:r['samples'][0].update(wall_including_pressure_ms=-1),
        'nonfinite_clock':lambda r:r['samples'][0].update(gpu_start_seconds=float('nan')),
        'late_pipeline':lambda r:r.update(pipelines_prepared_before_collection=False),
        'wrong_math_method':lambda r:r.update(candidate_selected=0),
        'changed_weights':lambda r:r['readonly_checks'][0]['sha256'].__setitem__(1,'0'*64),
        'retained_resources':lambda r:r.update(after_close_allocated_bytes=r['baseline_allocated_bytes']+4194305),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(source);mutate(bad)
        try:check(bad,cohort,readonly)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    pressure=policy['cohorts'][2];bad=copy.deepcopy(data[pressure['name']]);bad['samples'][0]['pressure_gpu_end_seconds']=bad['samples'][0]['gpu_end_seconds']+1
    try:check(bad,pressure,readonly)
    except ValueError:rejected.append('pressure_overlaps_measured_span')
    else:raise ValueError('overlapping pressure admitted')
    ratios=[cell['control_over_candidate'] for cohort in reports.values() for cell in cohort.values()]
    gate=all(r>=policy['minimum_gpu_speedup'] and r>=1/(1+policy['maximum_individual_regression']) for r in ratios)
    output=dict(passed=True,scope=__doc__,cohorts=reports,measured_commands=480,warmup_commands=120,exact_candidate_method=2,minimum_observed_control_over_candidate=min(ratios),maximum_observed_control_over_candidate=max(ratios),declared_operator_gain_gate_passed=gate,negative_controls_rejected=rejected,files_sha256=files,whole_decode_measured=False,serving_speed_claim=False,no_native_runtime_integration=True,goal_complete=False)
    DEST.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
