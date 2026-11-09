"""Check uninstrumented preparation before any timing, without repeating GEMM."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert read(ROOT/'correctness-audit1.json')['passed']
assert sha(ROOT/'combined1.metal')==read(ROOT/'correctness-audit1.json')['combined_sha256']
command=read(ROOT/'compile1-command.json');command[command.index(str(ROOT/'screen1.mm'))]=str(ROOT/'normal_prepare1.mm');command[-1]=str(ROOT/'normal_prepare1')
for name,cmd in [('normal-prepare-compile1',command),('normal-prepare1',[str(ROOT/'normal_prepare1'),str(ROOT/'combined1.metal'),str(T/'assets1'),str(T/'captures1'),str(ROOT/'normal-prepare1'),'correctness'])]:
    assert not (ROOT/(name+'-status.json')).exists();save(ROOT/(name+'-command.json'),cmd)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
        code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
assert not engine_processes(),engine_processes()
