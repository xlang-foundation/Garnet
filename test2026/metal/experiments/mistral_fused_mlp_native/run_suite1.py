"""Build the clean candidate's native tests and execute them serially."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
assert json.loads((ROOT/'model1-status.json').read_text())['status']=='terminal'
assert json.loads((ROOT/'model1-audit.json').read_text())['passed']
assert not engine_processes(),engine_processes()
for name,command in [
    ('suite-build1',['cmake','--build',str(ROOT/'build2'),'--parallel','4']),
    ('suite1',['ctest','--test-dir',str(ROOT/'build2'),'--output-on-failure','--parallel','1']),
    ('use-proof1',[str(ROOT/'use_proof_test1')]),
]:
    status=ROOT/(name+'-status.json');assert not status.exists(),'Never overwrite a suite run'
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(status,state)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(status,state)
    print(json.dumps(dict(step=name,**state)),flush=True)
    assert code==0,(name,code)
assert not engine_processes(),engine_processes()
