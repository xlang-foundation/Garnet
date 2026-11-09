"""Compile an extra lifecycle test against frozen ON/OFF native static libraries."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-verification-split-score-native1';O=M/'mistral-fused-q4-decode-native1';W=M.parents[2]
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
sys.path.insert(0,str(O/'tools1'));from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes();assert not (R/'preflight1.json').exists()
checks={}
for n,h in read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256'].items():assert sha(W/'Garnet'/n)==h;checks[str(W/'Garnet'/n)]=h
for p in [R/'plan1.json',R/'test1.cpp',R/'run1.py',N/'build-audit1.json',N/'provenance1.json',N/'source2/src/backends/metal/metal_context.h']:checks[str(p)]=sha(p)
commands={}
for mode,build in [('on','build-on1'),('off','build-default-off1')]:
 b=N/build;flags=b/'CMakeFiles/garnet_mistral_verification_split_score_test.dir/flags.make';lib=b/'libgarnet_metal.a';checks[str(flags)]=sha(flags);checks[str(lib)]=sha(lib);s=flags.read_text();value=lambda k:shlex.split(re.search('^'+k+' = (.*)$',s,re.M).group(1))
 commands[mode]=['/usr/bin/c++']+value('CXX_DEFINES')+value('CXX_INCLUDES')+value('CXX_FLAGS')+[str(R/'test1.cpp'),str(lib),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(R/('test-'+mode+'1'))]
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,no_model_or_comparator_execution=True))
def run(name,cmd):
 save(R/(name+'-command.json'),cmd)
 with (R/(name+'.stdout')).open('x') as out,(R/(name+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);state=dict(status='running',pid=p.pid,started_ns=time.time_ns());save(R/(name+'-status.json'),state);print(json.dumps(dict(step=name,**state)),flush=True);code=p.wait();state.update(status='terminal',returncode=code,finished_ns=time.time_ns());save(R/(name+'-status.json'),state)
 assert code==0,(name,code)
for mode in ['on','off']:run('compile-'+mode+'1',commands[mode])
for mode in ['on','off']:assert not engine_processes(),engine_processes();run('test-'+mode+'1',[str(R/('test-'+mode+'1'))])
for n,h in checks.items():assert sha(Path(n))==h,n
rows=[]
for mode in ['on','off']:
 s=(R/('test-'+mode+'1.stdout')).read_text();x=re.search(r'PASS cases=(\d+) fp32_parent_words=(\d+) rejections=(\d+) split_dispatches=(\d+) owners=0',s);assert x,(mode,s);cases,words,rejections,dispatches=map(int,x.groups());assert cases==12 and words>0 and rejections>=84 and dispatches==(24 if mode=='on' else 0);rows.append(dict(mode=mode,cases=cases,fp32_parent_words=words,rejections=rejections,split_dispatches=dispatches))
save(R/'audit1.json',dict(passed=True,rows=rows,original_dirty_unchanged=True,all_owned_processes_terminal=True,no_model_or_comparator_execution=True,performance_measured=False,goal_complete=False));print(json.dumps(rows),flush=True)
