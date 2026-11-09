"""Build and serially verify the default-off wide half-profile extension."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;S=ROOT/'source2';B=ROOT/'build2'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
sys.path.insert(0,str(ROOT/'tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
bindings=read(ROOT/'source-bindings2.json')
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
def run(name,command,expected=0):
    status=ROOT/(name+'-status.json');assert not status.exists(),name
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(status,state)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(status,state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==expected,(name,code,expected)
command=read(ROOT/'configure2-command.json')
run('invalid-half-off2',[str(ROOT/'invalid-half-off2') if v==str(B) else '-DGARNET_METAL_CHUNK_HALF_ATTENTION=OFF' if v=='-DGARNET_METAL_CHUNK_HALF_ATTENTION=ON' else v for v in command],1)
assert 'Wide half attention requires half attention' in (ROOT/'invalid-half-off2.stderr').read_text()
run('configure2',command)
run('build2',['/opt/homebrew/bin/cmake','--build',str(B),'--parallel','4'])
assert not engine_processes(),engine_processes()
run('suite2',['/opt/homebrew/bin/ctest','--test-dir',str(B),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 52' in (ROOT/'suite2.stdout').read_text()
log=(B/'Testing/Temporary/LastTest.log').read_text()
for expected in ['half_native512/128 exact partition first=0 active=512 gated=0','half_native512/128 exact partition first=1920 active=257 gated=0','half_native512/128 exact partition first=3584 active=512 gated=1','Native half attention PASS cells=12']:
    assert expected in log,expected
off=ROOT/'build-default-off2';offcommand=[str(off) if v==str(B) else '-DGARNET_METAL_CHUNK_HALF_WIDE=OFF' if v=='-DGARNET_METAL_CHUNK_HALF_WIDE=ON' else v for v in command]
run('default-off-configure2',offcommand)
run('default-off-build2',['/opt/homebrew/bin/cmake','--build',str(off),'--parallel','4'])
assert not engine_processes(),engine_processes()
run('default-off-suite2',['/opt/homebrew/bin/ctest','--test-dir',str(off),'--output-on-failure','--parallel','1'])
assert '100% tests passed out of 52' in (ROOT/'default-off-suite2.stdout').read_text()
assert 'Native half attention PASS cells=8' in (off/'Testing/Temporary/LastTest.log').read_text()
assert not engine_processes(),engine_processes()
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
library=ROOT/'library2';assert not library.exists();library.mkdir();shutil.copyfile(B/'libgarnet.dylib',library/'libgarnet.dylib')
save(ROOT/'build-audit2.json',dict(passed=True,source_bindings_sha256=sha(ROOT/'source-bindings2.json'),library_sha256=sha(library/'libgarnet.dylib'),suite_stdout_sha256=sha(ROOT/'suite2.stdout'),default_off_suite_stdout_sha256=sha(ROOT/'default-off-suite2.stdout'),tests_wide_enabled=52,tests_default_off=52,exact_partition_cases=3,source_files_unchanged=len(bindings['files_sha256']),new_option_default_enabled=False,shader_source_unchanged=True,private_half_scratch_bytes=68*1024*1024+384,new_model_or_serving_execution=False,goal_complete=False))
