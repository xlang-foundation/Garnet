"""Bind inputs; build once; run two fresh Metal processes strictly serially."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,time
R=Path(__file__).resolve().parent;P=R.parent/'mistral-wide-fp32-projection-screen1'
W=Path('/Users/shawn/CantorAI');G=W/'Garnet'
BASE=R.parent/'mistral-fused-q4-decode-native1/tools1/run_local_coder_baseline.py'
spec=importlib.util.spec_from_file_location('baseline',BASE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 return h.hexdigest()
def save(name,v):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(v,indent=2)+'\n')
def idle():
 assert not b.engine_processes(),b.engine_processes()
 commands=subprocess.check_output(['ps','-axo','pid=,comm='],text=True)
 extra=[line.strip() for line in commands.splitlines() if any(line.strip().endswith('/'+name) for name in ['screen1','screen2','screen3','screen4'])]
 assert not extra,extra
 return commands
old=json.loads((P/'preflight1.json').read_text());audit=json.loads((R.parent/'mistral-half-attention-native1/source-audit7.json').read_text())
dirty=audit['original_dirty_sha256']
def dirty_check():
 for name,h in dirty.items():assert sha(G/name)==h,name
 return True
def bindings():
 files=old['files_sha256'].copy()
 for name,h in json.loads((R/'derivation1.json').read_text())['base_files_sha256'].items():
  assert sha(Path(name))==h,name;files[name]=h
 for name,h in files.items():assert sha(Path(name))==h,name
 model=W/'out/models/Mistral-7B-Instruct-v0.3-MLX-4bit/model.safetensors'
 assert sha(model)==old['model_sha256'];files[str(model)]=old['model_sha256']
 return files
idle();dirty_check();files=bindings()
save('preflight1.json',dict(files_sha256=files,model_sha256=old['model_sha256'],original_dirty_sha256=dirty,original_dirty_unchanged=True,no_comparator_execution=True,processes=idle()))
def launch(name,argv):
 save(name+'-command.json',argv);start=time.time()
 with (R/(name+'.stdout')).open('w') as o,(R/(name+'.stderr')).open('w') as e:
  p=subprocess.Popen(argv,stdout=o,stderr=e,start_new_session=True)
  save(name+'-pid.json',dict(pid=p.pid,argv=argv,started_unix=start))
  rc=p.wait()
 save(name+'-status.json',dict(exit_code=rc,elapsed_seconds=time.time()-start,pid=p.pid,terminal=True))
 assert rc==0,(name,rc)
launch('build1',['/usr/bin/clang++','-std=c++17','-O2','-fobjc-arc','-Wno-deprecated-declarations','-I'+str(G/'third_party'),str(R/'screen1.mm'),'-framework','Foundation','-framework','Metal','-o',str(R/'screen1')])
save('execution-bindings1.json',{name:sha(R/name) for name in ['screen1.mm','screen1','prepare1.py','run1.py','audit1.py','control1.metal','current4x4.metal','candidate8x2.metal','plan1.json','derivation1.json']})
for cohort in [1,2]:
 idle();launch(f'run1-cohort{cohort}',[str(R/'screen1'),str(R),str(R/f'run1-cohort{cohort}'),str(cohort)])
# No hashing, builds, or concurrent GPU/model workers during either collection.
assert files==bindings();dirty_check()
save('postflight1.json',dict(files_sha256=files,original_dirty_unchanged=True,no_comparator_execution=True,all_owned_engines_terminal=True,processes=idle()))
print('Two fresh serial cohorts complete; postflight bindings unchanged.',flush=True)
