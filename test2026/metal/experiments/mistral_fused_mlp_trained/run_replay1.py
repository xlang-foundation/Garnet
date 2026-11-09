"""One serial untimed full-MLP replay against the exact trained native values."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
assert not (ROOT/'replay1-status.json').exists(),'Never reuse a replay'
assert not engine_processes(),engine_processes()
assert read(ROOT/'capture-audit1.json')['passed'] and read(ROOT/'replay1-compile-status.json')['returncode']==0
binding=read(ROOT/'replay-bindings1.json');inputs={}
for mapping in [binding['assets_sha256'],binding['capture_files_sha256']]:
    for name,digest in mapping.items():assert sha(ROOT/name)==digest,name;inputs[str(ROOT/name)]=digest
for name in ['replay1','replay1.mm','run_replay1.py','combined1.metal','ordinary1.metal','replay-bindings1.json','capture-audit1.json']:
    inputs[str(ROOT/name)]=sha(ROOT/name)
assert inputs[str(ROOT/'combined1.metal')]==binding['combined_sha256']
assert inputs[str(ROOT/'ordinary1.metal')]==binding['ordinary_sha256']
command=[str(ROOT/'replay1'),str(ROOT/'combined1.metal'),str(ROOT/'ordinary1.metal'),str(ROOT/'assets1'),str(ROOT/'captures1'),str(ROOT/'replay-output1')]
save(ROOT/'replay1-command.json',dict(command=command,files_sha256=inputs,diagnostic_only=True))
state=dict(status='starting',started_ns=time.time_ns(),parent_pid=os.getpid())
try:
    with (ROOT/'replay1.stdout').open('x') as out,(ROOT/'replay1.stderr').open('x') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        state.update(status='running',pid=process.pid);save(ROOT/'replay1-status.json',state)
        state.update(returncode=process.wait(),status='terminal',finished_ns=time.time_ns());save(ROOT/'replay1-status.json',state)
    assert state['returncode']==0,state
    assert not engine_processes(),engine_processes()
    for path,digest in inputs.items():assert sha(Path(path))==digest,path
    save(ROOT/'replay1-postflight.json',dict(passed=True,checked_files=len(inputs),process_terminal=True))
    print(json.dumps(dict(**state,candidate_exact=read(ROOT/'replay-output1/result1.json')['candidate_exact'])),flush=True)
finally:save(ROOT/'replay1-status.json',state)
