"""Bind fixtures, build the new factor screen and run one serial GPU audit."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent
N=M/'mistral-fused-mlp-native1';T=M/'mistral-fused-mlp-trained1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
sys.path.insert(0,str(M/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (ROOT/'compile1-status.json').exists() and not (ROOT/'correctness1-status.json').exists()
checked={}
def bind(p,h=None):
    actual=sha(p)
    if h is not None:assert actual==h,str(p)
    checked[str(p)]=actual
bind(N/'provenance1.json','b94f6e7ee34d17079439e88cd0d5e419ff9c11e8d2514fbe2dbd00f589917ebe')
bind(T/'provenance1.json','191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a')
for group in ['assets_sha256','capture_files_sha256']:
    for name,h in read(T/'replay-bindings1.json')[group].items():bind(T/name,h)
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():bind(W/'Garnet'/name,h)
for name in ['prepare1.py','screen1.mm','combined1.metal','plan1.json','run_correctness1.py']:bind(ROOT/name)
for name,h in read(ROOT/'plan1.json')['primary_reference_files_sha256'].items():bind(Path(name),h)
for name in ['src/model/float16_codec.h','src/backends/metal/packed_affine_layout_guard.h','third_party/nlohmann/json.hpp']:
    bind(N/'source2'/name,read(N/'source-bindings2.json')['files_sha256'][name])
def run(name,command):
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
command=['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-I',str(N/'source2/third_party'),'-I',str(N/'source2'),str(ROOT/'screen1.mm'),'-framework','Foundation','-framework','Metal','-o',str(ROOT/'screen1')]
save(ROOT/'correctness1-preflight.json',dict(passed=True,files_sha256=checked,no_comparator_execution=True,diagnostic_only=True))
run('compile1',command);bind(ROOT/'screen1')
save(ROOT/'correctness1-preflight.json',dict(passed=True,files_sha256=checked,no_comparator_execution=True,diagnostic_only=True))
run('correctness1',[str(ROOT/'screen1'),str(ROOT/'combined1.metal'),str(T/'assets1'),str(T/'captures1'),str(ROOT/'correctness1'),'correctness'])
assert not engine_processes(),engine_processes()
for name,h in checked.items():assert sha(Path(name))==h,name
save(ROOT/'correctness1-postflight.json',dict(passed=True,checked_files=len(checked),process_terminal=True))
