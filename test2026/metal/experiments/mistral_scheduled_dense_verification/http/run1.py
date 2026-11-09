"""Two fresh B1 serving engines, each cold+two warmed requests; strictly serial."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,threading,time
from http.server import ThreadingHTTPServer
R=Path(__file__).resolve().parent;M=R.parent;C=M/'mistral-scheduled-dense-verification1';N=M/'mistral-fused-q4-decode-native1';SG=M/'mistral-fused-mlp-serving1';W=M.parents[2]
sys.path.insert(0,str(SG/'tools1'));sys.path.insert(0,str(R))
from http_adapter1 import ScheduledWorker,ScheduledHandler
from local_coder_stream_benchmark import run_request
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
assert not engine_processes(),engine_processes();assert read(C/'qualification-audit2.json')['passed'];assert not (R/'preflight1.json').exists()
plan=read(R/'plan1.json');case=next(p for p in read(SG/'serving-manifest1.json')['plans'] if p['id']=='p2048-c1-o128');body=read(Path(case['cases']))[0];workload=read(Path(case['workload']))['requests'][0]
checks=read(C/'qualification1-preflight.json')['files_sha256']
for p in [R/'plan1.json',R/'profile1.json',R/'worker1.py',R/'http_adapter1.py',R/'run1.py',R/'preparation1.json',C/'audit_qualification2.py',C/'qualification-audit2.json',Path(case['cases']),Path(case['workload']),SG/'comparison1.json',SG/'provenance1.json']+sorted((SG/'tools1').glob('*.py')):checks[str(p)]=sha(p)
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'preflight1.json',dict(passed=True,files_sha256=checks,collection_serial=True,no_comparator_execution=True,source_model_hashing_during_collection=False,max_active_rows1_not_matched_to_reference_max4=True))
rows=[]
for engine in [1,2]:
 assert not engine_processes(),engine_processes();root=R/f'engine{engine}';root.mkdir();cmd=[str(W/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(R/'worker1.py'),str(C/'build-on1'),str(C/'source1'),str(R/'profile1.json')];save(root/'worker-command.json',cmd)
 worker=ScheduledWorker(cmd,root/'server',Path(read(R/'profile1.json')['runtime_inputs']['target_model']),first_token_timeout=300,model_family='mistral_7b');state=dict(status='running',pid=worker.process.pid,started_ns=time.time_ns());save(root/'status1.json',state);print(json.dumps(dict(engine=engine,**state)),flush=True)
 server=None
 try:
  deadline=time.monotonic()+120
  while not worker.healthy():
   if worker.failure:raise RuntimeError(worker.failure)
   assert time.monotonic()<deadline,'Worker ready deadline';time.sleep(.1)
  save(root/'ready1.json',worker.ready_info);assert worker.ready_info['active_requests']==1
  server=ThreadingHTTPServer(('127.0.0.1',0),ScheduledHandler);server.worker=worker;server.model=body['model'];server.model_family='mistral_7b';thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='http://127.0.0.1:'+str(server.server_port)
  for wave in [0,1,2]:
   dest=root/f'wave{wave}';dest.mkdir();result=run_request(url,body,dest/'request-0',threading.Barrier(1),300,workload['prompt_tokens'],False);assert result['status']=='completed' and result['generated_tokens']==128,(engine,wave,result)
   rows.append(dict(engine=engine,wave=wave,phase='cold' if wave==0 else 'warmed',result=result));print(json.dumps(dict(engine=engine,wave=wave,ttft=result['ttft_content_seconds'],decode_tps=result['decode_tokens_per_second'])),flush=True)
 finally:
  if server is not None:server.shutdown();server.server_close();thread.join(timeout=10)
  worker.close();state.update(status='terminal',returncode=worker.process.returncode,finished_ns=time.time_ns());save(root/'status1.json',state)
 assert worker.process.returncode==0,(engine,worker.process.returncode)
 assert not engine_processes(),engine_processes()
for n,h in checks.items():assert sha(Path(n))==h,n
save(R/'postflight1.json',dict(passed=True,checked_files=len(checks),all_owned_engines_terminal=True,no_comparator_execution=True));save(R/'result1.json',dict(completed=True,rows=rows,scope=plan['scope'],maximum_profile_matching=False,matched_c4_admitted=False,goal_complete=False));print('Two serial native B1 serving engines terminal',flush=True)
