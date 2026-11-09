"""Compile before collection; execute two fresh pressure cohorts serially."""
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
assert not (ROOT/'pressure1-preflight.json').exists()
audit=read(ROOT/'correctness-audit1.json');normal=read(ROOT/'normal-preparation-audit1.json')
assert audit['passed'] and normal['passed'] and normal['normal_and_counted_half_bits_exact']
assert read(ROOT/'timing1-postflight.json')['passed']
for co in ['a','b']:
    s=read(ROOT/('timing-'+co+'1-status.json'));assert s['status']=='terminal' and s['returncode']==0
derivation=read(ROOT/'pressure-derivation1.json')
assert derivation['passed'] and sha(ROOT/'combined1.metal')==derivation['qualified_shader_prefix_sha256']==audit['combined_sha256']==normal['qualified_shader_sha256']
assert (ROOT/'pressure1.metal').read_bytes().startswith((ROOT/'combined1.metal').read_bytes())
assert sha(ROOT/'pressure1.metal')==derivation['new_shader_sha256'] and sha(ROOT/'screen1.mm')==derivation['original_host_sha256']
host=(ROOT/'screen1.mm').read_text()
for change in derivation['host_changes']:
    assert host.count(change['old'])==1;host=host.replace(change['old'],change['new'])
assert host==(ROOT/'pressure1.mm').read_text() and sha(ROOT/'pressure1.mm')==derivation['pressure_host_sha256']
command=read(ROOT/'compile1-command.json')
command=[str(ROOT/'pressure1.mm') if v==str(ROOT/'screen1.mm') else str(ROOT/'pressure1') if v==str(ROOT/'screen1') else v for v in command]
save(ROOT/'pressure-compile1-command.json',command)
assert not (ROOT/'pressure1').exists()
with (ROOT/'pressure-compile1.stdout').open('x') as out,(ROOT/'pressure-compile1.stderr').open('x') as err:
    began=time.time_ns();p=subprocess.run(command,stdout=out,stderr=err)
    save(ROOT/'pressure-compile1-status.json',dict(status='terminal',returncode=p.returncode,started_ns=began,finished_ns=time.time_ns()))
assert p.returncode==0
checks=dict(read(ROOT/'timing1-preflight.json')['files_sha256'])
for name,h in checks.items():assert sha(Path(name))==h,name
for name in ['timing1-preflight.json','timing1-postflight.json','timing-a1/result1.json','timing-b1/result1.json','timing-a1-status.json','timing-b1-status.json','prepare_pressure1.py','pressure-derivation1.json','pressure1.metal','pressure1.mm','pressure1','run_pressure1.py','pressure-compile1-command.json','pressure-compile1-status.json']:
    checks[str(ROOT/name)]=sha(ROOT/name)
save(ROOT/'pressure1-preflight.json',dict(passed=True,files_sha256=checks,qualified_shader_prefix_unchanged=True,normal_preparation_bits_checked=True,measured_rounds_per_mode_cell=20,warmups_per_mode_cell=5,source_model_hashing_during_collection=False,no_comparator_execution=True,sweep_bytes_read=256*1024*1024,sweep_bytes_written=256*1024*1024,sweep_outside_operator_interval=True,no_claim_all_caches_cold=True))
for cohort in ['a','b']:
    assert not engine_processes(),engine_processes()
    name='pressure-'+cohort+'1';assert not (ROOT/(name+'-status.json')).exists()
    command=[str(ROOT/'pressure1'),str(ROOT/'pressure1.metal'),str(T/'assets1'),str(T/'captures1'),str(ROOT/name),'timing'];save(ROOT/(name+'-command.json'),command)
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
save(ROOT/'pressure1-postflight.json',dict(passed=True,checked_files=len(checks),all_gpu_processes_terminal=True))
