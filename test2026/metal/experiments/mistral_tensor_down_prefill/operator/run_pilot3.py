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
assert not engine_processes(),engine_processes();plan=read(R/'plan3.json');assert read(R/'host-compile3-status.json')['returncode']==0
assets=Path(plan['trained_assets']);captures=Path(plan['trained_captures']);paths=[R/n for n in ['run_pilot3.py','pilot3.mm','pilot3','combined3.metal','shared3.metal','prepare1.py','plan3.json','host-compile3-command.json','host-compile3-status.json']]
paths+=[Q/'build-audit1.json',Q/'source1/src/backends/metal/packed_affine_shader.h',Q/'source1/src/model/float16_codec.h']
for shape in ['gate','down']:
 paths+=[assets/'layer-15'/(shape+s) for s in ['-weight.u32','-scales.f16','-biases.f16']]
paths+=[captures/'layer-15'/n for n in ['norm-input.f32','gate.f32','silu.f32','down.f32']]
checks={str(p):sha(p) for p in paths};assert checks[str(R/'combined3.metal')]==plan['combined_sha256'] and checks[str(R/'shared3.metal')]==plan['new_shader_sha256']
save('pilot3-preflight.json',dict(passed=True,files_sha256=checks,correctness_only=True,no_comparator_execution=True,performance_admitted=False))
cmd=[str(R/'pilot3'),str(R/'combined3.metal'),str(assets),str(captures),str(R/'pilot3-output')];save('pilot3-command.json',cmd);started=time.time_ns()
with (R/'pilot3.stdout').open('x') as out,(R/'pilot3.stderr').open('x') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ));save('pilot3-pid.json',dict(pid=p.pid,started_ns=started));print('Pilot',p.pid,'running',flush=True);rc=p.wait()
save('pilot3-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=started,finished_ns=time.time_ns()));print('Pilot terminal',rc,flush=True)
if rc:print((R/'pilot3.stderr').read_text()[:10000])
assert rc==0 and read(R/'pilot3-output/result1.json')['completed']
assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('pilot3-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
