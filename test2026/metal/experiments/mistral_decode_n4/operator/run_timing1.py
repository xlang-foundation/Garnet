"""Collect paired row-major/N4 operator timing in four serial fresh processes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;T=M/'mistral-fused-mlp-trained1'
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
assert not engine_processes(),engine_processes()
assert not(ROOT/'timing1-status.json').exists();gate=read(ROOT/'correctness-audit1.json');assert gate['passed'] and gate['full_parent_and_padding_exact'] and gate['lossless_q4_and_coefficients_verified']
checks={}
for name,h in read(ROOT/'correctness1-preflight.json')['files_sha256'].items():assert sha(Path(name))==h,name;checks[name]=h
for p in [ROOT/'audit_correctness1.py',ROOT/'audit_transport1.py',ROOT/'audit-derivation1.json',ROOT/'correctness-audit1.json',ROOT/'contiguous1/result1.json',ROOT/'padded1/result1.json',ROOT/'run_timing1.py']:checks[str(p)]=sha(p)
save(ROOT/'timing1-preflight.json',dict(passed=True,files_sha256=checks,processes_serial=True,inference_or_comparator_execution=False))
state=dict(status='running',started_ns=time.time_ns(),processes=[]);save(ROOT/'timing1-status.json',state)
for name,mode,order in [('warm-a1','warm',0),('warm-b1','warm',1),('pressure-a1','pressure',0),('pressure-b1','pressure',1)]:
    assert not engine_processes(),engine_processes()
    cmd=[str(ROOT/'screen1'),str(ROOT),str(T/'assets1'),str(T/'captures1'),str(ROOT/name),mode,str(order),'20','contiguous'];save(ROOT/(name+'-command.json'),cmd)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        record=dict(name=name,mode=mode,order=order,status='running',pid=p.pid,started_ns=time.time_ns());state['processes'].append(record);save(ROOT/'timing1-status.json',state);print(json.dumps(record),flush=True)
        code=p.wait();record.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),record);save(ROOT/'timing1-status.json',state);assert code==0,(name,code)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
state.update(status='terminal',returncode=0,finished_ns=time.time_ns());save(ROOT/'timing1-status.json',state);print(json.dumps(dict(status='terminal',returncode=0,processes=4)),flush=True)
