"""Build the isolated optional N4 engine and run both native suites serially."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,shutil
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;J=M/'mistral-prefill-completion-wide1';S=ROOT/'source1';B=ROOT/'build1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(M/'mistral-wide-half-prefix-http1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
bindings=read(ROOT/'source-bindings2.json')
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
def run(name,command,expected=0):
    assert not(ROOT/(name+'-status.json')).exists();save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True);state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    assert code==expected,(name,code,expected)
command=[str(S) if v==str(J/'source2') else str(B) if v==str(J/'build2') else v for v in read(J/'configure2-command.json')]+['-DGARNET_METAL_MISTRAL_N4_DECODE=ON']
run('invalid-reuse-off1',[str(ROOT/'invalid-reuse-off1') if v==str(B) else '-DGARNET_METAL_MISTRAL_DECODE_REUSE=OFF' if v=='-DGARNET_METAL_MISTRAL_DECODE_REUSE=ON' else v for v in command],1)
assert 'N4 decode requires' in (ROOT/'invalid-reuse-off1.stderr').read_text()
run('configure1',command);run('build1',['/opt/homebrew/bin/cmake','--build',str(B),'--parallel','4'])
assert not engine_processes(),engine_processes();run('suite1',['/opt/homebrew/bin/ctest','--test-dir',str(B),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 54' in (ROOT/'suite1.stdout').read_text()
off=ROOT/'build-default-off1';run('default-off-configure1',[str(off) if v==str(B) else '-DGARNET_METAL_MISTRAL_N4_DECODE=OFF' if v=='-DGARNET_METAL_MISTRAL_N4_DECODE=ON' else v for v in command])
run('default-off-build1',['/opt/homebrew/bin/cmake','--build',str(off),'--parallel','4'])
assert not engine_processes(),engine_processes();run('default-off-suite1',['/opt/homebrew/bin/ctest','--test-dir',str(off),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 54' in (ROOT/'default-off-suite1.stdout').read_text()
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
assert not engine_processes(),engine_processes();library=ROOT/'library1';assert not library.exists();library.mkdir();shutil.copyfile(B/'libgarnet.dylib',library/'libgarnet.dylib')
save(ROOT/'build-audit1.json',dict(passed=True,source_bindings_sha256=sha(ROOT/'source-bindings2.json'),library_sha256=sha(library/'libgarnet.dylib'),tests_enabled=54,tests_default_off=54,native_n4_exact_word_comparisons=1253376,only_four_active_mlp_rows=True,cache_budget_bytes=3221225472,model_or_serving_measured=False,goal_complete=False))
