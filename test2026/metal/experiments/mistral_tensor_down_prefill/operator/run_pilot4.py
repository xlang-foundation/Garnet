"""Bind trained fixtures and run one isolated operator correctness pilot."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;assert not p.exists(),n;p.write_text(json.dumps(v,indent=2)+'\n')
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();plan=read(R/'plan4.json');assert read(R/'host-compile4-status.json')['returncode']==0
assets=Path(plan['trained_assets']);captures=Path(plan['trained_captures']);paths=[R/n for n in ['run_pilot4.py','pilot4.mm','pilot4','combined4.metal','walk4.metal','prepare1.py','plan4.json','host-compile4-command.json','host-compile4-status.json']]
paths+=[Q/'build-audit1.json',Q/'source1/src/backends/metal/packed_affine_shader.h',Q/'source1/src/model/float16_codec.h']
for shape in ['gate','down']:
 paths+=[assets/'layer-15'/(shape+s) for s in ['-weight.u32','-scales.f16','-biases.f16']]
paths+=[captures/'layer-15'/n for n in ['norm-input.f32','gate.f32','silu.f32','down.f32']]
checks={str(p):sha(p) for p in paths};assert checks[str(R/'combined4.metal')]==plan['combined_sha256'] and checks[str(R/'walk4.metal')]==plan['new_shader_sha256']
save('pilot4-preflight.json',dict(passed=True,files_sha256=checks,correctness_only=True,no_comparator_execution=True,performance_admitted=False))
cmd=[str(R/'pilot4'),str(R/'combined4.metal'),str(assets),str(captures),str(R/'pilot4-output')];save('pilot4-command.json',cmd);started=time.time_ns()
with (R/'pilot4.stdout').open('x') as out,(R/'pilot4.stderr').open('x') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ));save('pilot4-pid.json',dict(pid=p.pid,started_ns=started));print('Pilot',p.pid,'running',flush=True);rc=p.wait()
save('pilot4-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=started,finished_ns=time.time_ns()));print('Pilot terminal',rc,flush=True)
if rc:print((R/'pilot4.stderr').read_text()[:10000])
assert rc==0 and read(R/'pilot4-output/result1.json')['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('pilot4-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
