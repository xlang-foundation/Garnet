"""Compile the isolated host, bind frozen fixtures and run one serial operator correctness process."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;P=M/'mistral-tensor-view-prefill-screen1';Q=M/'mistral-causal-batch4-helper1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')
def run(name,cmd):
 save(name+'-command.json',cmd);start=time.time_ns()
 with (R/(name+'.stdout')).open('x') as o,(R/(name+'.stderr')).open('x') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True);save(name+'-pid.json',dict(pid=p.pid,started_ns=start));print(name,p.pid,'running',flush=True);rc=p.wait()
 save(name+'-status.json',dict(status='terminal',returncode=rc,pid=p.pid,started_ns=start,finished_ns=time.time_ns()));print(name,rc,'terminal',flush=True)
 if rc:print((R/(name+'.stderr')).read_text()[:6000],flush=True)
 assert rc==0,name
cmd=read(P/'host-compile3-command.json');cmd=[str(R/'pilot1.mm') if a==str(P/'pilot3.mm') else str(R/'pilot1') if a==str(P/'pilot3') else a for a in cmd];run('host-compile1',cmd)
sys.path.insert(0,str(M/'mistral-fused-q4-decode-native1/tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();plan=read(R/'plan1.json');paths=[R/n for n in ['run_pilot1.py','prepare1.py','pilot1.mm','pilot1','combined1.metal','row_tiles1.metal','plan1.json','host-compile1-command.json','host-compile1-status.json']]
paths+=[Q/'source1/src/backends/metal/packed_affine_shader.h',Q/'source1/src/model/float16_codec.h']
F=M/'mistral-fused-mlp-trained1';old=read(F/'provenance1.json');assets=Path(plan['trained_assets']);captures=Path(plan['trained_captures'])
fixtures=[assets/'layer-15'/(shape+s) for shape in ['gate','down'] for s in ['-weight.u32','-scales.f16','-biases.f16']]+[captures/'layer-15'/n for n in ['norm-input.f32','gate.f32','silu.f32','down.f32']]
for p in fixtures:assert sha(p)==old['files_sha256'][str(p.relative_to(F))],p
checks={str(p):sha(p) for p in paths+fixtures};assert checks[str(R/'combined1.metal')]==plan['shader_sha256'] and checks[str(R/'row_tiles1.metal')]==plan['variant_sha256']
save('pilot1-preflight.json',dict(passed=True,files_sha256=checks,correctness_only=True,no_comparator_execution=True,original_trained_provenance_sha256=sha(F/'provenance1.json'),full_half_weight_expansion=False,performance_admitted=False))
run('pilot1',[str(R/'pilot1'),str(R/'combined1.metal'),str(assets),str(captures),str(R/'pilot1-output')]);assert read(R/'pilot1-output/result1.json')['completed'];assert not engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save('pilot1-postflight.json',dict(passed=True,checked_files=len(checks),all_owned_processes_terminal=True,no_comparator_execution=True))
