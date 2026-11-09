"""Collect four fresh geometry processes serially; hash only outside collection."""
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
assert not(ROOT/'timing2-status.json').exists()
gate=read(ROOT/'correctness-audit1.json');assert gate['passed'] and gate['all_complete_parents_exact_legacy_fp32']
assert read(ROOT/'compile3-status.json')['returncode']==0
checks={}
for name,h in read(ROOT/'correctness2-preflight.json')['files_sha256'].items():assert sha(Path(name))==h,name;checks[name]=h
for p in [ROOT/'audit_correctness1.py',ROOT/'correctness-audit1.json',ROOT/'correctness1/result1.json',ROOT/'correctness2/result1.json',ROOT/'run_timing2.py',ROOT/'screen3.mm',ROOT/'screen3',ROOT/'compile3-status.json',ROOT/'host-derivation3.json',ROOT/'timing-policy2.json']:checks[str(p)]=sha(p)
save(ROOT/'timing2-preflight.json',dict(passed=True,files_sha256=checks,processes_serial=True,inference_or_comparator_execution=False))
state=dict(status='running',started_ns=time.time_ns(),processes=[]);save(ROOT/'timing2-status.json',state)
for name,mode,order in [('warm-a2','warm',0),('warm-b2','warm',1),('pressure-a2','pressure',0),('pressure-b2','pressure',1)]:
    assert not engine_processes(),engine_processes()
    cmd=[str(ROOT/'screen3'),str(ROOT),str(T/'assets1'),str(T/'captures1'),str(ROOT/name),mode,str(order),'20'];save(ROOT/(name+'-command.json'),cmd)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        record=dict(name=name,mode=mode,order=order,status='running',pid=p.pid,started_ns=time.time_ns());state['processes'].append(record);save(ROOT/'timing2-status.json',state);print(json.dumps(record),flush=True)
        code=p.wait();record.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),record);save(ROOT/'timing2-status.json',state);assert code==0,(name,code)
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'timing2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
state.update(status='terminal',returncode=0,finished_ns=time.time_ns());save(ROOT/'timing2-status.json',state);print(json.dumps(dict(status='terminal',returncode=0,processes=4)),flush=True)
