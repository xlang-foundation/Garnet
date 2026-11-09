"""Build the isolated native policy and run its tests serially."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;S=ROOT/'source1';B=ROOT/'build1'
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
bindings=read(ROOT/'source-bindings1.json')
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
def run(name,command):
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
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
run('portable-compile1',['xcrun','clang++','-std=c++17','-O2',str(S/'test2026/native/scheduler_fixed_rows_test.cpp'),str(S/'src/runtime/scheduler/continuous_batch_scheduler.cpp'),'-o',str(ROOT/'portable1')])
run('portable1',[str(ROOT/'portable1')])
run('configure1',read(ROOT/'configure1-command.json'))
run('build1',['/opt/homebrew/bin/cmake','--build',str(B),'--parallel','4'])
assert not engine_processes(),engine_processes()
run('suite1',['/opt/homebrew/bin/ctest','--test-dir',str(B),'--output-on-failure','--parallel','1'])
assert not engine_processes(),engine_processes()
for name,h in bindings['files_sha256'].items():assert sha(S/name)==h,name
library=ROOT/'library1';assert not library.exists();library.mkdir();shutil.copyfile(B/'libgarnet.dylib',library/'libgarnet.dylib')
save(ROOT/'build-audit1.json',dict(passed=True,source_bindings_sha256=sha(ROOT/'source-bindings1.json'),library_sha256=sha(library/'libgarnet.dylib'),portable_binary_sha256=sha(ROOT/'portable1'),suite_stdout_sha256=sha(ROOT/'suite1.stdout'),source_files_unchanged=len(bindings['files_sha256']),default_policy_enabled=False,new_model_or_serving_execution=False,goal_complete=False))
