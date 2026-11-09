"""Bind and run one standalone centered group64 arithmetic pilot, serially."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;T=M/'mistral-fused-mlp-trained1';N=M/'mistral-fused-q4-decode-native1';B=M/'mistral-decode-threadgroup-screen1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(N/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (R/'run2-status.json').exists() and read(R/'compile3-status.json')['returncode']==0
checks={}
def bind(p,h=None):
    v=sha(p)
    if h is not None:assert h==v,str(p)
    checks[str(p)]=v
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/n,h)
for key in ['assets_sha256','capture_files_sha256']:
    for n,h in read(T/'replay-bindings1.json')[key].items():
        if 'layer-15/' in n:bind(T/n,h)
for p in [R/'run2.py',R/'plan1.json',R/'plan2.json',R/'pilot-output1/result1.json',R/'group64-vector2.metal',R/'pilot3.mm',R/'pilot3',R/'compile3-command.json',R/'ordinary2.metal',R/'indexed2.metal']:bind(p)
for n,h in read(R/'plan1.json')['control_shader_sha256'].items():bind(B/n,h)
save(R/'preflight2.json',dict(passed=True,files_sha256=checks,pilot_only=True,no_model_or_comparator_execution=True))
command=[str(R/'pilot3'),str(R),str(T/'assets1'),str(T/'captures1'),str(R/'pilot-output2')];save(R/'run2-command.json',command)
with (R/'run2.stdout').open('x') as out,(R/'run2.stderr').open('x') as err:
    p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/'run2-status.json',state);print(json.dumps(state),flush=True)
    code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/'run2-status.json',state)
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight2.json',dict(passed=True,all_owned_processes_terminal=True,checked_files=len(checks),pilot_only=True));print(json.dumps(state),flush=True)
raise SystemExit(code)
