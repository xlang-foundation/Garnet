"""Audit N4 against the actual native occupancy-one scalar control."""
from pathlib import Path
import hashlib,json,sys,copy
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'native-c1-timing-audit2.json'
sys.path.insert(0,str(HERE));from audit_native_c1_transport2 import verify,require
files={}
def raw(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    require(not DEST.exists(),'never overwrite audit');status=read(ROOT/'native-c1-timing2-status.json');require(status['status']=='terminal' and status['returncode']==0 and len(status['processes'])==4,'unfinished timing')
    require(read(ROOT/'native-c1-timing2-postflight.json')['passed'],'missing postflight')
    for name,h in read(ROOT/'native-c1-timing2-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed input '+name)
    qualified={(p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['variant']):p for p in read(ROOT/'contiguous1/result1.json')['pipeline_resources']}
    names=['native-c1-warm-a2','native-c1-warm-b2','native-c1-pressure-a2','native-c1-pressure-b2'];previous=0;reports={};data={}
    for name,p in zip(names,status['processes']):
        require(p['name']==name and p['status']=='terminal' and p['returncode']==0 and p['started_ns']>=previous and p['finished_ns']>p['started_ns'],'failed/overlapping process');previous=p['finished_ns'];d=read(ROOT/name/'result1.json');require(d['native_c1_control_variant']==0,'wrong scalar control');data[name]=d;reports[name]=verify(d,'pressure' if 'pressure' in name else 'warm',int(name.endswith('b2')),qualified)
    rows=[c for p in reports.values() for c in p['c1_three_projection_sums']];require(len(rows)==8,'wrong native C1 coverage')
    decision=dict(qualified_for_native_c1_integration_screen=all(c['three_projection_candidate_ms']<=.97*c['three_projection_control_ms'] and c['worst_individual_projection_ratio']>=1/1.02 for c in rows),minimum_ratio=min(c['operator_ratio_control_over_candidate'] for c in rows),maximum_ratio=max(c['operator_ratio_control_over_candidate'] for c in rows),worst_individual_projection_ratio=min(c['worst_individual_projection_ratio'] for c in rows))
    mutations={
        'wrong_scalar_control':lambda d:d['samples'][0].update(variant=4),
        'missing_sample':lambda d:d['samples'].pop(),
        'false_gpu_duration':lambda d:d['samples'][0].update(gpu_ms=0),
        'changed_placement':lambda d:d['samples'][0].update(mask=3),
        'pressure_overlap':lambda d:d['samples'][0].update(pressure_gpu_end_seconds=d['samples'][0]['gpu_end_seconds']+1),
        'late_prepare':lambda d:d.update(all_pipelines_prepared_before_collection=False),
        'changed_native_profile':lambda d:d['pipeline_resources'][0].update(input_stride=1),
        'changed_arithmetic':lambda d:d.update(native_math='relaxed'),
    };rejected=[]
    for name,fn in mutations.items():
        d=copy.deepcopy(data['native-c1-pressure-a2']);fn(d)
        try:verify(d,'pressure',0,qualified)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corrupt native C1 evidence admitted '+name)
    for p in [Path(__file__),HERE/'audit_native_c1_transport2.py',ROOT/'native-c1-audit-derivation2.json']:raw(p)
    result=dict(passed=True,scope=__doc__,processes=reports,measured_samples=960,warmup_dispatches=240,decision=decision,negative_controls_rejected=rejected,files_sha256=files,earlier_c1_blocked4_comparison_is_surrogate=True,c4_blocked4_comparison_unchanged=True,token_speed_measured=False,full_model_memory_admission_pending=True,native_runtime_integrated=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','processes']},indent=2))
if __name__=='__main__':main()
