"""Compare N4 with the actual occupancy-one scalar kernel, in serial fresh processes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;J=M/'mistral-prefill-completion-wide1';T=M/'mistral-fused-mlp-trained1'
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
assert not engine_processes(),engine_processes();assert not(ROOT/'native-c1-timing2-status.json').exists()
require=read(ROOT/'correctness-audit1.json');assert require['passed'] and require['full_parent_and_padding_exact']
def run(name,command):
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        status=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/(name+'-status.json'),status);print(json.dumps(dict(step=name,**status)),flush=True);code=p.wait();status.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),status);assert code==0,(name,code)
run('native-c1-compile2',['xcrun','clang++','-std=c++20','-O3','-fobjc-arc','-I',str(J/'source2/third_party'),'-I',str(J/'source2'),str(ROOT/'screen-c1-native2.mm'),'-framework','Foundation','-framework','Metal','-o',str(ROOT/'screen-c1-native2')])
checks={}
for name,h in read(ROOT/'correctness1-preflight.json')['files_sha256'].items():assert sha(Path(name))==h,name;checks[name]=h
for p in [ROOT/'screen-c1-native2.mm',ROOT/'screen-c1-native2',ROOT/'host-c1-derivation2.json',ROOT/'native-c1-policy2.json',ROOT/'run_native_c1_timing2.py',ROOT/'correctness-audit1.json']:checks[str(p)]=sha(p)
save(ROOT/'native-c1-timing2-preflight.json',dict(passed=True,files_sha256=checks))
state=dict(status='running',processes=[]);save(ROOT/'native-c1-timing2-status.json',state)
for label,mode,order in [('native-c1-warm-a2','warm',0),('native-c1-warm-b2','warm',1),('native-c1-pressure-a2','pressure',0),('native-c1-pressure-b2','pressure',1)]:
    assert not engine_processes(),engine_processes()
    run(label,[str(ROOT/'screen-c1-native2'),str(ROOT),str(T/'assets1'),str(T/'captures1'),str(ROOT/label),mode,str(order),'20','contiguous']);state['processes'].append(dict(name=label,**read(ROOT/(label+'-status.json'))));save(ROOT/'native-c1-timing2-status.json',state)
for name,h in checks.items():assert sha(Path(name))==h,name
assert not engine_processes(),engine_processes();save(ROOT/'native-c1-timing2-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True));state.update(status='terminal',returncode=0);save(ROOT/'native-c1-timing2-status.json',state);print('Native scalar C1 comparison terminal',flush=True)
