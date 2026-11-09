"""Run only the new isolated native operator correctness screen, once."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
M=ROOT.parent
sys.path.insert(0,str(M/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
binding=json.loads((ROOT/'bindings1.json').read_text())
assert not engine_processes(),engine_processes()
assert not (ROOT/'correctness1-status.json').exists()
for name,digest in binding['assets_sha256'].items():assert sha(ROOT/name)==digest,name
for name,digest in binding['source_headers_sha256'].items():assert sha(M/'mistral-half-attention-native1/source7/src/backends/metal'/name)==digest,name
assert sha(ROOT/'combined1.metal')==binding['combined_sha256']
assert sha(ROOT/'ordinary1.metal')==binding['ordinary_sha256']
headers=M/'mistral-half-attention-native1/source7'
command=[str(ROOT/'host1'),str(ROOT/'combined1.metal'),str(ROOT/'ordinary1.metal'),str(ROOT/'assets1'),str(ROOT/'correctness1')]
compile_command=['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',str(headers/'third_party'),'-I',str(headers),str(ROOT/'host1.mm'),'-framework','Foundation','-framework','Metal','-o',str(ROOT/'host1')]
save(ROOT/'compile2-command.json',dict(command=compile_command,returncode=0,stderr_empty=(ROOT/'compile2.stderr').stat().st_size==0))
files={str(p):sha(p) for p in [ROOT/'host1',ROOT/'host1.mm',ROOT/'combined1.metal',ROOT/'ordinary1.metal',ROOT/'bindings1.json',ROOT/'prepare1.py',ROOT/'run_correctness1.py',headers/'src/model/float16_codec.h',headers/'third_party/nlohmann/json.hpp']}
save(ROOT/'correctness1-command.json',dict(command=command,files_sha256=files,assets_verified=True,started_ns=time.time_ns(),parent_pid=os.getpid()))
state=dict(status='starting',started_ns=time.time_ns())
try:
    with (ROOT/'correctness1.stdout').open('x') as out,(ROOT/'correctness1.stderr').open('x') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        state.update(status='running',pid=process.pid);save(ROOT/'correctness1-status.json',state)
        state.update(returncode=process.wait(),finished_ns=time.time_ns(),status='terminal')
    save(ROOT/'correctness1-status.json',state)
    assert state['returncode']==0,state
    for p,digest in files.items():assert sha(Path(p))==digest,p
    assert not engine_processes(),engine_processes()
    save(ROOT/'correctness1-postflight.json',dict(passed=True,checked_files=len(files),no_competing_engines=True))
    print(json.dumps(state))
finally:
    save(ROOT/'correctness1-status.json',state)
