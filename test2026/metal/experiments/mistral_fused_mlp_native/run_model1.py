"""One untimed fused MLP full-model regression; frozen controls are read only."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mistral-half-attention-native1'
WORKSPACE=ROOT.parents[3]
sys.path.insert(0,str(ROOT.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,value):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
assert not (ROOT/'model1-status.json').exists(),'Never reuse a model run'
assert not engine_processes(),engine_processes()
assert read(ROOT/'build2-status.json')['returncode']==0
unit=read(ROOT/'native-test1.stdout')
assert unit['passed'] and unit['cases']==54 and unit['all_native_owners_zero']
assert sha(BASE/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
foundation=read(BASE/'provenance7.json')
source=read(ROOT/'source-bindings2.json');original=read(BASE/'model-bindings7.json');checked={}
def bind(p,expected=None):
    actual=sha(p)
    if expected is not None:assert actual==expected,str(p)
    checked[str(p)]=actual
for name,digest in source['files_sha256'].items():bind(ROOT/'source2'/name,digest)
for name,digest in read(BASE/'source-audit7.json')['original_dirty_sha256'].items():bind(WORKSPACE/'Garnet'/name,digest)
for name,digest in original['model_files_sha256'].items():bind(Path(original['model'])/name,digest)
bind(Path(original['xlang']),original['xlang_sha256'])
bind(Path(original['fixture']),original['fixture_sha256'])
for name in ['model-on7.stdout','model-state-audit7.json','model-bindings7.json']:
    bind(BASE/name,foundation['files_sha256'][name])
assert read(BASE/'model-state-audit7.json')['passed']
library=ROOT/'library2';library.mkdir();shutil.copyfile(ROOT/'build2/libgarnet.dylib',library/'libgarnet.dylib')
for name in ['profile-model1.json','worker-model1.py','native_test1','native-test1.stdout','run_model1.py','source-bindings2.json']:
    bind(ROOT/name)
bind(library/'libgarnet.dylib');bind(ROOT/'build2/libgarnet.dylib')
command=[original['xlang'],'-B',str(ROOT/'worker-model1.py'),str(library),str(ROOT/'source2'),original['model'],str(ROOT/'profile-model1.json')]
save(ROOT/'model1-command.json',command)
save(ROOT/'model1-preflight.json',dict(passed=True,files_sha256=checked,command=command,native_unit=unit,no_comparator_execution=True,scope='Untimed full-model correctness regression, not a speed benchmark.'))
state=dict(status='starting',parent_pid=os.getpid(),started_ns=time.time_ns(),completed_requests=[],fused_blocks=0,state_witnesses=[])
process=None
try:
    with (ROOT/'model1.stdout').open('x') as log,(ROOT/'model1.stderr').open('x') as err:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,env=dict(os.environ,GARNET_METAL_PROFILE_OPS='0',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        state.update(status='running',pid=process.pid);save(ROOT/'model1-status.json',state)
        for line in process.stdout:
            log.write(line);log.flush()
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            kind=event.get('event');state['last_event']=kind
            if kind=='diagnostic_fused_mlp':state['fused_blocks']+=event['blocks']
            if kind=='diagnostic_full_state':state['state_witnesses'].append(event['invocation'])
            if kind=='completed':state['completed_requests'].append(event['id'])
            if kind in ['native_fused_mlp_prepared','ready','diagnostic_fused_mlp','completed','error','closed']:
                save(ROOT/'model1-status.json',state)
                print(json.dumps({k:event[k] for k in ['event','id','invocation','blocks','sites','error'] if k in event}),flush=True)
        state.update(status='terminal',returncode=process.wait(),finished_ns=time.time_ns());save(ROOT/'model1-status.json',state)
    assert not engine_processes(),engine_processes()
    for name,digest in checked.items():assert sha(Path(name))==digest,name
    save(ROOT/'model1-postflight.json',dict(passed=True,checked_files=len(checked),process_terminal=True))
    assert state['returncode']==0 and len(state['completed_requests'])==4 and state['last_event']=='closed',state
    print(json.dumps(state),flush=True)
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        state.update(status='terminal',returncode=process.returncode,interrupted_by_collector=True,finished_ns=time.time_ns())
    save(ROOT/'model1-status.json',state)
