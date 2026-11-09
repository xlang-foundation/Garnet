"""Build a prepared clean source and run its trained native fusion unit."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
TRAINED=ROOT.parent/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
sys.path.insert(0,str(ROOT.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert sha(TRAINED/'provenance1.json')=='191c7098368d3a1d2fdfdac3bafad93c3f897d8c8674cba96e8e4c83b843b04a'
checked={}
for group in ['assets_sha256','capture_files_sha256']:
    for name,h in read(TRAINED/'replay-bindings1.json')[group].items():assert sha(TRAINED/name)==h;checked[str(TRAINED/name)]=h
for name,h in read(ROOT/'source-bindings2.json')['files_sha256'].items():assert sha(ROOT/'source2'/name)==h
save(ROOT/'native-unit1-bindings.json',dict(passed=True,files_sha256=checked,test_sha256=sha(ROOT/'native_test1.mm')))
def run(name,command):
    status=ROOT/(name+'-status.json');assert not status.exists()
    if name!='configure2':save(ROOT/(name+'-command.json'),command)
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
run('configure2',read(ROOT/'configure2-command.json'))
run('build2',['cmake','--build',str(ROOT/'build2'),'--target','garnet','--parallel','4'])
flags={}
for line in (ROOT/'build2/CMakeFiles/garnet_metal.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
command=['/usr/bin/c++']+flags['OBJCXX_DEFINES']+flags['OBJCXX_INCLUDES']+['-I',str(ROOT/'source2/src/model')]+flags['OBJCXX_FLAGS']+['-c',str(ROOT/'native_test1.mm'),'-o',str(ROOT/'native_test1.o')]
run('native-test1-compile',command)
run('native-test1-link',['/usr/bin/c++','-arch','arm64',str(ROOT/'native_test1.o'),str(ROOT/'build2/libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(ROOT/'native_test1')])
run('native-test1',[str(ROOT/'native_test1'),str(TRAINED/'assets1'),str(TRAINED/'captures1')])
run('use-proof1-compile',['/usr/bin/c++','-std=c++17','-O2','-I',str(ROOT/'source2/src/backends/metal'),str(ROOT/'use_proof_test1.cpp'),'-o',str(ROOT/'use_proof_test1')])
for name,h in checked.items():assert sha(Path(name))==h,name
assert not engine_processes(),engine_processes()
