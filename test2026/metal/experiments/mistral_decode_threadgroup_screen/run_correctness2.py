"""Compile the isolated geometry screen, then collect one serial correctness arm."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;W=M.parents[2];J=M/'mistral-prefill-completion-wide1';T=M/'mistral-fused-mlp-trained1'
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
assert not(ROOT/'compile2-status.json').exists() and not(ROOT/'correctness2-status.json').exists()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
bind(J/'provenance1.json','3dd125f8f9ed9bef0a944cdb14eca7ff0f7af3b0bc19f818fc884763d99a9411')
bind(T/'provenance1.json','191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a')
for name,h in read(ROOT/'plan1.json')['files_sha256'].items():bind(Path(name),h)
for group in ['assets_sha256','capture_files_sha256']:
    for name,h in read(T/'replay-bindings1.json')[group].items():bind(T/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
for name in ['src/model/float16_codec.h','third_party/nlohmann/json.hpp']:bind(J/'source2'/name,read(J/'source-bindings2.json')['files_sha256'][name])
for name in ['plan1.json','prepare1.py','preparation1.json','ordinary1.metal','indexed1.metal','screen1.mm','screen2.mm','host-derivation2.json','run_correctness2.py']:bind(ROOT/name)
def run(name,argv):
    save(ROOT/(name+'-command.json'),argv)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
command=['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-I',str(J/'source2/third_party'),'-I',str(J/'source2'),str(ROOT/'screen2.mm'),'-framework','Foundation','-framework','Metal','-o',str(ROOT/'screen2')]
save(ROOT/'correctness2-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_inference=True,performance_measurement=False))
run('compile2',command);bind(ROOT/'screen2');save(ROOT/'correctness2-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_inference=True,performance_measurement=False))
run('correctness2',[str(ROOT/'screen2'),str(ROOT),str(T/'assets1'),str(T/'captures1'),str(ROOT/'correctness2'),'correctness','0','3'])
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'correctness2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
