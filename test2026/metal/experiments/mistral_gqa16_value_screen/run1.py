"""Build once; two serial fresh-process operator cohorts; no comparator execution."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,time
R=Path(__file__).resolve().parent;W=Path('/Users/shawn/CantorAI');G=W/'Garnet'
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
 found=b.engine_processes();assert not found,found
 commands=subprocess.check_output(['ps','-axo','pid=,comm='],text=True)
 extra=[line.strip() for line in commands.splitlines() if any(line.strip().endswith('/'+name) for name in ['screen1','screen2','screen3','screen4'])]
 arguments=subprocess.check_output(['ps','-axww','-o','pid=,args='],text=True)
 argv={line.strip().split(None,1)[0]:line.strip().split(None,1)[1] for line in arguments.splitlines() if len(line.strip().split(None,1))==2}
 for line in commands.splitlines():
  fields=line.strip().split(None,1)
  if len(fields)==2 and Path(fields[1]).name=='xlang3' and '/WorldSense/' not in fields[1]+' '+argv.get(fields[0],''):extra.append(line.strip())
 assert not extra,extra
 return commands
plan=json.loads((R/'plan1.json').read_text());old=json.loads((R.parent/'mistral-half-attention-native1/source-audit7.json').read_text())
dirty=old['original_dirty_sha256']
def check():
 for name,h in dirty.items():assert sha(G/name)==h,name
 for name,h in plan['base_source_sha256'].items():assert sha(Path(name))==h,name
 return True
idle();check()
save('preflight1.json',dict(passed=True,base_source_sha256=plan['base_source_sha256'],original_dirty_sha256=dirty,original_dirty_unchanged=True,hashes_during_collection=False,no_comparator_execution=True,processes=idle()))
def launch(name,argv):
 save(name+'-command.json',argv);start=time.time()
 with (R/(name+'.stdout')).open('w') as o,(R/(name+'.stderr')).open('w') as e:
  p=subprocess.Popen(argv,stdout=o,stderr=e,start_new_session=True)
  save(name+'-pid.json',dict(pid=p.pid,argv=argv,started_unix=start))
  rc=p.wait()
 save(name+'-status.json',dict(exit_code=rc,elapsed_seconds=time.time()-start,pid=p.pid,terminal=True))
 assert rc==0,(name,rc)
launch('build1',['/usr/bin/clang++','-std=c++17','-O2','-fobjc-arc','-Wno-deprecated-declarations','-I'+str(G/'third_party'),str(R/'screen1.mm'),'-framework','Foundation','-framework','Metal','-o',str(R/'screen1')])
save('execution-bindings1.json',{name:sha(R/name) for name in ['screen1.mm','screen1','run1.py','control1.metal','candidate1.metal','value1.metal','helpers1.inc','main1.inc','plan1.json']})
for cohort in [1,2]:
 idle();launch(f'run1-cohort{cohort}',[str(R/'screen1'),str(R),str(R/f'run1-cohort{cohort}'),str(cohort)])
check()
for name,h in json.loads((R/'execution-bindings1.json').read_text()).items():assert sha(R/name)==h,name
save('postflight1.json',dict(passed=True,base_source_sha256=plan['base_source_sha256'],original_dirty_unchanged=True,no_comparator_execution=True,all_owned_engines_terminal=True,execution_bindings_unchanged=True,processes=idle()))
print('Both fresh serial GQA value cohorts complete.',flush=True)
