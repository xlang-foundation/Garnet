"""Audit serial native blocked/N4 timing and the declared C1/C4 integration screen."""
from pathlib import Path
import copy,hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit1.json'
sys.path.insert(0,str(HERE));from audit_timing_transport1 import verify,require
files={}
def raw(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
NAMES=['warm-a1','warm-b1','pressure-a1','pressure-b1']
def main():
    require(not DEST.exists(),'never overwrite audit');status=read(ROOT/'timing1-status.json');require(status['status']=='terminal' and status['returncode']==0 and len(status['processes'])==4,'unfinished timing')
    require(read(ROOT/'timing1-postflight.json')['passed'],'missing postflight')
    for name,h in read(ROOT/'timing1-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed input '+name)
    previous=0
    for name,p in zip(NAMES,status['processes']):require(p['name']==name and p['status']=='terminal' and p['returncode']==0 and p['started_ns']>=previous and p['finished_ns']>p['started_ns'],'overlapping/failed native processes');previous=p['finished_ns']
    gate=read(ROOT/'correctness-audit1.json');require(gate['passed'] and gate['lossless_q4_and_coefficients_verified'],'unqualified arithmetic/words')
    resource=read(ROOT/'contiguous1/result1.json')['pipeline_resources'];qualified={(p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['variant']):p for p in resource}
    policy=read(ROOT/'timing-policy1.json');require(policy['control_variant']==4 and policy['candidate_variant']==9 and policy['physical_simd_groups']==4,'changed declared policy')
    data={name:read(ROOT/name/'result1.json') for name in NAMES};reports={}
    for name,d in data.items():
        report=verify(d,'pressure' if name.startswith('pressure') else 'warm',int(name.endswith('b1')),qualified)
        cells=[c for c in report['medians'] if c['physical_rows']==4 and c['mask']==15 and c['indexed'] is False]
        require(len(cells)==3 and {c['shape'] for c in cells}=={'gate','up','down'},'missing C4 projection sum')
        control=sum(c['control_median_gpu_ms'] for c in cells);candidate=sum(c['candidate_median_gpu_ms'] for c in cells)
        report['c4_three_projection_sum']=dict(control_ms=control,candidate_ms=candidate,operator_ratio_control_over_candidate=control/candidate,worst_individual_projection_ratio=min(c['operator_ratio_control_over_candidate'] for c in cells));reports[name]=report
    rejected=[]
    def event(d):return d['samples'][0]
    mutations={
        'missing_sample':lambda d:d['samples'].pop(),
        'wrong_variant':lambda d:event(d).update(variant=0),
        'wrong_order':lambda d:event(d).update(position=1),
        'wrong_active_rows':lambda d:event(d).update(mask=0),
        'false_gpu_time':lambda d:event(d).update(gpu_ms=1e-12),
        'nonfinite_time':lambda d:event(d).update(gpu_start_seconds=float('nan')),
        'pressure_overlap':lambda d:event(d).update(pressure_gpu_end_seconds=event(d)['gpu_end_seconds']+1),
        'missing_pressure':lambda d:event(d).update(pressure_gpu_start_seconds=0,pressure_gpu_end_seconds=0),
        'wrong_pressure_size':lambda d:d.update(pressure_bytes=1),
        'wrong_warmups':lambda d:d.update(warmups=0),
        'late_preparation':lambda d:d.update(all_pipelines_prepared_before_collection=False),
        'missing_profile':lambda d:d['pipeline_resources'].pop(),
        'wrong_stride':lambda d:d['pipeline_resources'][0].update(input_stride=1),
        'changed_arithmetic':lambda d:d.update(native_math='half relaxed'),
        'retained_buffers':lambda d:d.update(after_close_allocated_bytes=1<<40),
    }
    for name,mutate in mutations.items():
        d=copy.deepcopy(data['pressure-a1']);mutate(d)
        try:verify(d,'pressure',0,qualified)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corrupt timing admitted '+name)
    c1=[c for p in reports.values() for c in p['c1_three_projection_sums']];c4=[p['c4_three_projection_sum'] for p in reports.values()]
    require(len(c1)==8 and len(c4)==4,'wrong decision coverage')
    decision=dict(c1_qualified_for_integration_screen=all(c['three_projection_candidate_ms']<=.97*c['three_projection_control_ms'] and c['worst_individual_projection_ratio']>=1/1.02 for c in c1),c4_qualified_for_integration_screen=all(c['candidate_ms']<=.97*c['control_ms'] and c['worst_individual_projection_ratio']>=1/1.02 for c in c4),minimum_c1_ratio=min(c['operator_ratio_control_over_candidate'] for c in c1),maximum_c1_ratio=max(c['operator_ratio_control_over_candidate'] for c in c1),minimum_c4_ratio=min(c['operator_ratio_control_over_candidate'] for c in c4),maximum_c4_ratio=max(c['operator_ratio_control_over_candidate'] for c in c4))
    for p in [Path(__file__),HERE/'audit_timing_transport1.py',ROOT/'timing-audit-derivation1.json']:raw(p)
    result=dict(passed=True,scope=__doc__,processes=reports,measured_samples=3360,warmup_dispatches=840,negative_controls_rejected=rejected,decision=decision,files_sha256=files,full_profile_preparation_before_collection=True,pressure_not_proof_of_all_caches_cold=True,one_time_repacking_excluded_from_projection_time=True,full_model_memory_admission_pending=True,native_runtime_integrated=False,token_throughput_measured=False,coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['processes','files_sha256']},indent=2))
if __name__=='__main__':main()
