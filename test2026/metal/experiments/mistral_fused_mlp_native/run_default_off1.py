"""Verify the new compile option defaults off and preserves the native suite."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
assert read(ROOT.parent/'mistral-fused-mlp-prefix-http1/launch1-status.json')['status']=='terminal'
assert not engine_processes(),engine_processes()
command=read(ROOT/'configure2-command.json');command=[a for a in command if not a.startswith('-DGARNET_METAL_FUSED_MISTRAL_MLP=')]
command[command.index('-B')+1]=str(ROOT/'build-default-off1')
for name,cmd in [('default-off-configure1',command),
                 ('default-off-build1',['cmake','--build',str(ROOT/'build-default-off1'),'--parallel','4']),
                 ('default-off-suite1',['ctest','--test-dir',str(ROOT/'build-default-off1'),'--output-on-failure','--parallel','1'])]:
    dest=ROOT/(name+'-status.json');assert not dest.exists()
    if name=='default-off-build1':assert 'GARNET_METAL_FUSED_MISTRAL_MLP:BOOL=OFF' in (ROOT/'build-default-off1/CMakeCache.txt').read_text()
    save(ROOT/(name+'-command.json'),cmd)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(dest,state)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(dest,state)
    print(json.dumps(dict(step=name,**state)),flush=True)
    assert code==0,(name,code)
assert not engine_processes(),engine_processes()
