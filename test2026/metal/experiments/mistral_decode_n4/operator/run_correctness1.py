"""Bind immutable operands; build and audit both layout classes serially."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;W=M.parents[2];J=M/'mistral-prefill-completion-wide1';G=M/'mistral-decode-threadgroup-screen1';T=M/'mistral-fused-mlp-trained1'
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
assert not(ROOT/'compile1-status.json').exists() and not(ROOT/'correctness1-status.json').exists()
checks={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checks[str(p)]=actual
plan=read(ROOT/'plan1.json');bind(J/'provenance1.json',plan['base_engine_provenance_sha256']);bind(G/'provenance1.json',plan['previous_geometry_provenance_sha256']);bind(T/'provenance1.json','191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a')
for name,h in read(ROOT/'preparation1.json')['source_headers_sha256'].items():bind(Path(name),h)
for group in ['assets_sha256','capture_files_sha256']:
    for name,h in read(T/'replay-bindings1.json')[group].items():bind(T/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
for name in ['src/model/float16_codec.h','third_party/nlohmann/json.hpp']:bind(J/'source2'/name,read(J/'source-bindings2.json')['files_sha256'][name])
for name in ['plan1.json','prepare1.py','preparation1.json','ordinary1.metal','indexed1.metal','interleaved1.inc','screen1.mm','run_correctness1.py','timing-policy1.json']:bind(ROOT/name)
def run(name,argv):
    save(ROOT/(name+'-command.json'),argv)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True)
        code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    assert code==0,(name,code)
save(ROOT/'correctness1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_inference=True,performance_measurement=False))
run('compile1',['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-I',str(J/'source2/third_party'),'-I',str(J/'source2'),str(ROOT/'screen1.mm'),'-framework','Foundation','-framework','Metal','-o',str(ROOT/'screen1')]);bind(ROOT/'screen1');save(ROOT/'correctness1-preflight.json',dict(passed=True,files_sha256=checks,no_comparator_inference=True,performance_measurement=False))
for layout in ['contiguous','padded']:
    run(layout+'1',[str(ROOT/'screen1'),str(ROOT),str(T/'assets1'),str(T/'captures1'),str(ROOT/(layout+'1')),'correctness','0','3',layout])
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'correctness1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True))
save(ROOT/'correctness1-status.json',dict(status='terminal',returncode=0,layouts=['contiguous','padded']))
print('Both native layout arms terminal',flush=True)
