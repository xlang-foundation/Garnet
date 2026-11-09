"""Time two serial fresh processes after counted/normal preparation admission."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;T=M/'mistral-fused-mlp-trained1'
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
assert not (ROOT/'timing1-preflight.json').exists()
audit=read(ROOT/'correctness-audit1.json');normal=read(ROOT/'normal-preparation-audit1.json')
assert audit['passed'] and normal['passed'] and normal['normal_and_counted_half_bits_exact']
assert sha(ROOT/'screen1')==audit['binary_sha256'] and sha(ROOT/'combined1.metal')==audit['combined_sha256']==normal['qualified_shader_sha256']
checks=dict(read(ROOT/'correctness1-preflight.json')['files_sha256'])
for name,h in checks.items():assert sha(Path(name))==h,name
for name in ['correctness-audit1.json','normal-preparation-audit1.json','audit_correctness1.py','audit_normal_prepare1.py','normal_prepare1.mm','normal_prepare1','run_timing1.py']:
    checks[str(ROOT/name)]=sha(ROOT/name)
save(ROOT/'timing1-preflight.json',dict(passed=True,files_sha256=checks,same_qualified_host_binary=True,normal_preparation_bits_checked=True,measured_rounds_per_mode_cell=20,warmups_per_mode_cell=5,source_model_hashing_during_collection=False,no_comparator_execution=True))
for cohort in ['a','b']:
    assert not engine_processes(),engine_processes()
    name='timing-'+cohort+'1';assert not (ROOT/(name+'-status.json')).exists()
    command=[str(ROOT/'screen1'),str(ROOT/'combined1.metal'),str(T/'assets1'),str(T/'captures1'),str(ROOT/name),'timing'];save(ROOT/(name+'-command.json'),command)
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
assert not engine_processes(),engine_processes()
for name,h in checks.items():assert sha(Path(name))==h,name
save(ROOT/'timing1-postflight.json',dict(passed=True,checked_files=len(checks),all_gpu_processes_terminal=True))
