"""Two serial fresh-process operator screens, balanced over20 measured rounds."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;T=M/'mistral-fused-mlp-trained1';N=M/'mistral-fused-mlp-native1'
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
audit=read(ROOT/'correctness-audit1.json');assert audit['passed'] and audit['cases']==1620
assert sha(ROOT/'combined1.metal')==audit['combined_sha256']
assert not (ROOT/'timing1-preflight.json').exists()
checks=dict(read(ROOT/'correctness1-preflight.json')['files_sha256'])
for name,h in checks.items():assert sha(Path(name))==h,name
text=(ROOT/'screen1.mm').read_text();old='for(int round=-5;round<15;round++)';new='for(int round=-5;round<20;round++)';assert text.count(old)==1
source=ROOT/'timing1.mm';assert not source.exists();source.write_text(text.replace(old,new))
save(ROOT/'timing1-derivation.json',dict(passed=True,source_sha256=sha(ROOT/'screen1.mm'),timing_source_sha256=sha(source),only_change=dict(old=old,new=new),reason='20rounds cover the complete10-round rotating/reversing order twice; each mode occupies each sample position four times. GPU shaders, all shared host setup/encoding and correctness path are unchanged.'))
def run(name,command):
    assert not (ROOT/(name+'-status.json')).exists()
    save(ROOT/(name+'-command.json'),command)
    with (ROOT/(name+'.stdout')).open('x') as out,(ROOT/(name+'.stderr')).open('x') as err:
        p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MTL_CAPTURE_ENABLED='0'))
        state=dict(status='running',pid=p.pid,parent_pid=os.getpid(),started_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
        try:code=p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            state.update(status='terminal',returncode=p.returncode,finished_ns=time.time_ns());save(ROOT/(name+'-status.json'),state)
    print(json.dumps(dict(step=name,**state)),flush=True);assert code==0,(name,code)
command=read(ROOT/'compile1-command.json');command[command.index(str(ROOT/'screen1.mm'))]=str(source);command[-1]=str(ROOT/'timing1')
run('timing-compile1',command)
for p in [ROOT/'correctness-audit1.json',source,ROOT/'timing1',ROOT/'timing1-derivation.json',ROOT/'run_timing1.py']:checks[str(p)]=sha(p)
save(ROOT/'timing1-preflight.json',dict(passed=True,files_sha256=checks,qualified_shader_unchanged=True,measured_rounds_per_mode_cell=20,warmup_rounds_per_mode_cell=5,source_model_hashing_during_collection=False,no_comparator_execution=True))
for cohort in ['a','b']:
    assert not engine_processes(),engine_processes()
    run('timing-'+cohort+'1',[str(ROOT/'timing1'),str(ROOT/'combined1.metal'),str(T/'assets1'),str(T/'captures1'),str(ROOT/('timing-'+cohort+'1')),'timing'])
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_gpu_processes_terminal=True))
