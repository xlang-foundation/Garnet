"""One untimed native model run; observe real MLP data without new arithmetic."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
M=ROOT.parent
BASE=M/'mistral-half-attention-native1'
sys.path.insert(0,str(M/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,d):
    temporary=p.with_name(p.name+'.tmp');temporary.write_text(json.dumps(d,indent=2)+'\n');temporary.replace(p)
assert not (ROOT/'capture1-status.json').exists(),'Never reuse a capture run'
assert not engine_processes(),engine_processes()
assert read(ROOT/'build1-status.json')['returncode']==0
unit=read(ROOT/'copy-test1.stdout');assert unit['passed'] and unit['all_native_owners_zero']
assert sha(BASE/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
source=read(ROOT/'source-bindings1.json');checked={}
def bind(p,expected=None):
    value=sha(p)
    if expected is not None:assert value==expected,str(p)
    checked[str(p)]=value
for name,digest in source['source_files_sha256'].items():bind(ROOT/'source1'/name,digest)
for name,digest in source['original_dirty_sha256'].items():bind(M.parents[2]/'Garnet'/name,digest)
original=read(BASE/'model-bindings7.json')
for name,digest in original['model_files_sha256'].items():bind(Path(original['model'])/name,digest)
bind(Path(original['xlang']),original['xlang_sha256'])
library=ROOT/'library1';library.mkdir();shutil.copyfile(ROOT/'build1/libgarnet.dylib',library/'libgarnet.dylib')
for p in [library/'libgarnet.dylib',ROOT/'profile1.json',ROOT/'worker-capture1.py',ROOT/'copy_test1',ROOT/'copy-test1.stdout',ROOT/'run_capture1.py',ROOT/'source-bindings1.json',BASE/'model-on7.stdout',BASE/'model-state-audit7.json']:
    bind(p)
fixture=Path(read(ROOT/'profile1.json')['diagnostic_fixture']);bind(fixture,original['fixture_sha256'])
(ROOT/'captures1').mkdir()
command=[original['xlang'],'-B',str(ROOT/'worker-capture1.py'),str(library),str(ROOT/'source1'),original['model'],str(ROOT/'profile1.json')]
save(ROOT/'capture1-preflight.json',dict(passed=True,command=command,files_sha256=checked,copy_unit=unit,no_comparator_execution=True,scope='Diagnostic capture only, not a benchmark.'))
state=dict(status='starting',parent_pid=os.getpid(),started_ns=time.time_ns(),saved_layers=[],completed_requests=[])
try:
    with (ROOT/'capture1.stdout').open('x') as log,(ROOT/'capture1.stderr').open('x') as err:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,env=dict(os.environ,GARNET_METAL_PROFILE_OPS='0',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        state.update(status='running',pid=process.pid);save(ROOT/'capture1-status.json',state)
        for line in process.stdout:
            log.write(line);log.flush()
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event.get('event')=='diagnostic_mlp_saved':
                state['saved_layers'].append(event['layer']);save(ROOT/'capture1-status.json',state);print(json.dumps(dict(saved_layer=event['layer'],rows=event['metadata']['rows'])),flush=True)
            elif event.get('event')=='completed':
                state['completed_requests'].append(event['id']);save(ROOT/'capture1-status.json',state)
            elif event.get('event')=='error':
                print(json.dumps(event),flush=True)
        state.update(status='terminal',returncode=process.wait(),finished_ns=time.time_ns());save(ROOT/'capture1-status.json',state)
    assert state['returncode']==0 and state['saved_layers']==[0,15,31] and len(state['completed_requests'])==4,state
    assert not engine_processes(),engine_processes()
    for p,digest in checked.items():assert sha(Path(p))==digest,p
    save(ROOT/'capture1-postflight.json',dict(passed=True,checked_files=len(checked),process_terminal=True))
    print(json.dumps(state),flush=True)
finally:save(ROOT/'capture1-status.json',state)
