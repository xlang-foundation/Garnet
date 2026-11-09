"""Four fresh serial processes, reverse pair order, actual C1/C4 native controls."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(M/'mistral-wide-half-prefix-http1/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert not(R/'timing1-preflight.json').exists()
assert read(R/'correctness-audit1.json')['passed'] and read(R/'correctness-audit1.json')['eligible_methods']==[2]
checks=dict(read(R/'correctness1-preflight.json')['files_sha256'])
for name,h in checks.items():assert sha(Path(name))==h,name
for p in [R/'correctness-audit1.json',R/'audit_correctness1.py',R/'run_timing1.py']:checks[str(p)]=sha(p)
cohorts=[dict(name='warm-a1',mode='warm',order=0),dict(name='warm-b1',mode='warm',order=1),dict(name='pressure-a1',mode='pressure',order=0),dict(name='pressure-b1',mode='pressure',order=1)]
policy=dict(cohorts=cohorts,candidate_method=2,control_method=-1,masks=[1,8,15],physical_rows=4,columns=14336,inner=4096,layer=15,layout='contiguous',warmups=5,rounds=20,pair_order='Control then candidate in order0; candidate then control in order1. Fresh paired cohorts reverse order.',minimum_gpu_speedup=1/0.97,maximum_individual_regression=0.02,only_occupancies_1_and_4_timed=True,no_comparator_execution=True)
save(R/'timing-policy1.json',policy);checks[str(R/'timing-policy1.json')]=sha(R/'timing-policy1.json')
save(R/'timing1-preflight.json',dict(passed=True,files_sha256=checks,source_asset_hashing_during_collection=False,no_comparator_execution=True))
for cohort in cohorts:
    assert not engine_processes(),engine_processes()
    name=cohort['name'];command=[str(R/'screen1'),str(R),str(T/'assets1'),str(T/'captures1'),str(R/name),cohort['mode'],str(cohort['order']),'20','contiguous','2'];save(R/(name+'-command.json'),command)
    with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
    assert code==0,(name,code)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(R/'timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
