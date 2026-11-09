"""Bind actual weights/activations, compile, then run two serial correctness probes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;J=M/'mistral-prefill-completion-wide1';T=M/'mistral-fused-mlp-trained1'
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
assert not engine_processes(),engine_processes();assert not(R/'correctness1-preflight.json').exists()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
prep=read(R/'preparation1.json')
for name,h in prep['inputs_sha256'].items():bind(Path(name),h)
for name,h in prep['generated_sha256'].items():bind(R/name,h)
for name,h in read(T/'replay-bindings1.json')['assets_sha256'].items():bind(T/name,h)
for name,h in read(T/'replay-bindings1.json')['capture_files_sha256'].items():bind(T/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/name,h)
for name in ['screen1.mm','run_correctness1.py','plan1.json','preparation1.json']:bind(R/name)
save(R/'correctness1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True))
def run(name,command):
    assert not(R/(name+'-status.json')).exists();save(R/(name+'-command.json'),command)
    with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
    assert code==0,(name,code)
run('compile1',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-I'+str(J/'source2/third_party'),'-I'+str(J/'source2'),str(R/'screen1.mm'),'-o',str(R/'screen1')]);checks[str(R/'screen1')]=sha(R/'screen1')
save(R/'correctness1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_execution=True))
for layout in ['contiguous','padded']:
    assert not engine_processes(),engine_processes()
    run(layout+'1',[str(R/'screen1'),str(R),str(T/'assets1'),str(T/'captures1'),str(R/(layout+'1')),'correctness','0','3',layout,'-1'])
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(R/'correctness1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
