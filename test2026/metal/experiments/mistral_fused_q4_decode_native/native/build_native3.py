"""Build isolated native fusion, serial ON/OFF suites, then trained native replay."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;J=M/'mistral-prefill-completion-wide1';T=M/'mistral-fused-mlp-trained1';S=R/'source3';B=R/'build3'
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
bindings=read(R/'source-bindings3.json')
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
for name,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():assert sha(M.parents[2]/'Garnet'/name)==h,name
trained={}
for key in ['assets_sha256','capture_files_sha256']:
    for name,h in read(T/'replay-bindings1.json')[key].items():assert sha(T/name)==h;trained[str(T/name)]=h
save(R/'build-preflight3.json',dict(passed=True,source_bindings_sha256=sha(R/'source-bindings3.json'),trained_files_sha256=trained,no_comparator_execution=True))
def run(name,command,expected=0):
    assert not(R/(name+'-status.json')).exists();save(R/(name+'-command.json'),command)
    with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
    assert code==expected,(name,code,expected)
command=[str(S) if v==str(J/'source2') else str(B) if v==str(J/'build2') else v for v in read(J/'configure2-command.json')]+['-DGARNET_METAL_FUSED_Q4_DECODE=ON']
run('invalid-reuse-off3',[str(R/'invalid-reuse-off3') if v==str(B) else '-DGARNET_METAL_MISTRAL_DECODE_REUSE=OFF' if v=='-DGARNET_METAL_MISTRAL_DECODE_REUSE=ON' else v for v in command],1)
assert 'Fused Q4 decode requires' in (R/'invalid-reuse-off3.stderr').read_text()
run('configure3',command);run('build3',['/opt/homebrew/bin/cmake','--build',str(B),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite3',['/opt/homebrew/bin/ctest','--test-dir',str(B),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 53' in (R/'suite3.stdout').read_text()
off=R/'build-default-off3'
run('default-off-configure3',[str(off) if v==str(B) else '-DGARNET_METAL_FUSED_Q4_DECODE=OFF' if v=='-DGARNET_METAL_FUSED_Q4_DECODE=ON' else v for v in command])
run('default-off-build3',['/opt/homebrew/bin/cmake','--build',str(off),'--parallel','4'])
assert not engine_processes(),engine_processes();run('default-off-suite3',['/opt/homebrew/bin/ctest','--test-dir',str(off),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 52' in (R/'default-off-suite3.stdout').read_text()
assert not engine_processes(),engine_processes();run('trained-native3',[str(B/'garnet_fused_q4_decode_test'),str(T/'assets1'),str(T/'captures1')])
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
for name,h in trained.items():assert sha(Path(name))==h,name
assert not engine_processes(),engine_processes()
library=R/'library3';assert not library.exists();library.mkdir();shutil.copyfile(B/'libgarnet.dylib',library/'libgarnet.dylib')
match=re.search(r'fused_q4_decode PASS checks=(\d+) fp32_parent_words=(\d+) owners=0',(R/'trained-native3.stdout').read_text());assert match
save(R/'build-audit3.json',dict(passed=True,source_bindings_sha256=sha(R/'source-bindings3.json'),library_sha256=sha(library/'libgarnet.dylib'),tests_enabled=53,tests_default_off=52,trained_native_checks=int(match[1]),trained_native_parent_words=int(match[2]),private_fp32_input_bytes=65536,extra_weight_cache_bytes=0,native_option_default_on=False,model_or_serving_measured=False,goal_complete=False))
