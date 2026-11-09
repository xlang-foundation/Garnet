"""Compile and collect one serial learned whole-MLP correctness pilot."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;P=M/'mistral-tensor-view-prefill-screen1';Q=M/'mistral-causal-batch4-helper1';T=M/'mistral-fused-mlp-trained1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=R/n;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
def run(name,cmd):
 save(name+'-command.json',cmd);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as o,(R/(name+'.stderr')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True);save(name+'-pid.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'.stderr')).read_text()[:5000],flush=True)
 assert rc==0,name
cmd=read(P/'host-compile3-command.json');cmd=[str(R/'pilot1.mm') if a==str(P/'pilot3.mm') else str(R/'pilot1') if a==str(P/'pilot3') else a for a in cmd];run('host-compile1',cmd)
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();plan=read(R/'plan1.json');paths=[R/n for n in ['run_pilot1.py','prepare1.py','fused_tensor1.metal','combined1.metal','pilot1.mm','pilot1','plan1.json','host-compile1-command.json','host-compile1-status.json']]
paths += [T/'ordinary1.metal',Q/'source1/src/backends/metal/packed_affine_shader.h',Q/'source1/src/backends/metal/fused_mistral_mlp_shader.h',Q/'source1/src/model/float16_codec.h',Q/'source1/src/backends/metal/packed_affine_layout_guard.h']
provenance=read(T/'provenance1.json');fixtures=[]
for layer in [0,15,31]:
 fixtures += [T/'assets1'/('layer-'+str(layer))/(shape+s) for shape in ['gate','up','down'] for s in ['-weight.u32','-scales.f16','-biases.f16']]
 fixtures += [T/'captures1'/('layer-'+str(layer))/(name+'.f32') for name in ['norm-input','gate','up','silu','down']]
for p in fixtures+[T/'ordinary1.metal']:assert sha(p)==provenance['files_sha256'][str(p.relative_to(T))],p
checks={str(p):sha(p) for p in paths+fixtures};save('pilot1-preflight.json',dict(passed=True,files_sha256=checks,trained_provenance_sha256=sha(T/'provenance1.json'),correctness_only=True,no_comparator_execution=True))
run('pilot1',[str(R/'pilot1'),str(R/'combined1.metal'),str(T/'ordinary1.metal'),str(T/'assets1'),str(T/'captures1'),str(R/'pilot1-output')]);d=read(R/'pilot1-output/result1.json');assert d['completed'] and d['candidate_exact']
assert not engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('pilot1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
