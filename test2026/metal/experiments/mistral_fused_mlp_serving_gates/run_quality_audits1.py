"""Grade the retained coding collection with unchanged sandbox evaluators."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(sys.argv[1]).resolve()
PY=ROOT.parents[1]/'qwen-local-coder/asset-env/bin/python'
read=lambda p:json.loads(p.read_text())
assert read(ROOT/'launch1-status.json')['status']=='terminal' and read(ROOT/'launch1-status.json')['returncode']==0
assert read(ROOT/'postflight1.json')['passed']
sys.path.insert(0,str(ROOT/'tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
steps=[
    ('strict-evaluate1',[str(PY),'-B',str(ROOT/'tools1/evaluate_local_coder_quality.py'),'--suite',str(ROOT/'tools1/suite.json'),'--responses',str(ROOT/'responses1'),'--output',str(ROOT/'strict-quality1.json')]),
    ('block-diagnose1',[str(PY),'-B',str(ROOT/'tools1/diagnose_local_coder_blocks.py'),'--suite',str(ROOT/'tools1/suite.json'),'--responses',str(ROOT/'responses1'),'--strict-report',str(ROOT/'strict-quality1.json'),'--output',str(ROOT/'block-diagnosis1.json')]),
    ('fused-audit1',[str(PY),'-B',str(ROOT/'audit_quality1.py')]),
]
for name,command in steps:
    status=ROOT/(name+'-status.json');assert not status.exists()
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(status,state)
        code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(status,state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
print(json.dumps(dict(collection_valid=read(ROOT/'fused-quality-audit1.json')['passed'],strict_passed=read(ROOT/'strict-quality1.json')['passed'],strict_total=8,quality_gate_passed=False)),flush=True)
