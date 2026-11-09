"""Diagnostic-only serial uncached / prefill-only HTTP return campaign.
Read-only native state observations perturb timing. No benchmark or quality admission.
"""
from pathlib import Path
import json,sys,time,subprocess,os,copy,threading,concurrent.futures,socket
from urllib.request import urlopen
r=Path(__file__).resolve().parent;w=Path('/Users/shawn/CantorAI');tools=r/'tools1';sys.path.insert(0,str(tools))
from run_local_coder_baseline import engine_processes,stop_owned
from local_coder_stream_benchmark import run_request
fixture=json.loads((r/'fixture1.json').read_text());base=json.loads((r/'profile-template1.json').read_text());bindings=json.loads((r/'bindings1.json').read_text());report=dict(status='running',scope=__doc__,phases=[])
def save():(r/'status1.json').write_text(json.dumps(report,indent=2)+'\n')
def first_event(path,future):
 while True:
  if path.exists():
   for line in path.read_text().splitlines():
    try:data=json.loads(line)['data']
    except json.JSONDecodeError:continue
    if data!='[DONE]':
     item=json.loads(data)
     if item.get('usage',{}).get('completion_tokens',0)>=1:return
  if future.done():raise RuntimeError('A ended before first-token observation: '+str(future.result()))
  time.sleep(.01)
def call(folder,entry,barrier,url):
 p=folder/entry['label'];p.mkdir();result=run_request(url,entry['body'],p/'request-0',barrier,600,entry['prompt_tokens'],True)
 if result['status']!='completed':raise RuntimeError('HTTP request failed '+entry['label']+': '+repr(result))
 return result
assert not (r/'status1.json').exists(), 'Never overwrite a collection'
try:
 for label,cache in [('uncached',False),('cached',True)]:
  competing=engine_processes()
  if competing:raise RuntimeError('Competing engine: '+repr(competing))
  folder=r/label;folder.mkdir();active=copy.deepcopy(base);active.update(prefix_cache=cache,native_cached_chunks=cache,profile_id=base['profile_id']+'-'+label);active['memory_inventory_path']=str(folder/'engine-inventory1.json')
  profile=folder/'profile.json';profile.write_text(json.dumps(active,indent=2)+'\n')
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  url='http://127.0.0.1:'+str(port)
  cmd=[str(w/'out/research/qwen-local-coder/asset-env/bin/python'),'-B',str(tools/'local_coder_batch_http.py'),'--checkpoint',fixture['model'],'--log-root',str(folder/'server'),'--first-token-timeout','600','--port',str(port),'--model','local-mistral-7b','--model-family','mistral_7b','--',str(w/'out/build/macos-arm64-release/bin/xlang3'),'-B',str(tools/'local_coder_batch_worker.py'),bindings['library'],bindings['source'],fixture['model'],str(profile)]
  (folder/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');state=dict(label=label,cache=cache,status='starting',command=cmd,url=url,transitions=[]);report['phases'].append(state);save()
  log=(folder/'server.log').open('x');process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GARNET_METAL_PROFILE_OPS='0'));state['pid']=process.pid;save();print('starting',label,process.pid,flush=True)
  try:
   while True:
    if process.poll() is not None or (folder/'server/worker-exit.json').exists():raise RuntimeError('Native startup terminated '+label)
    try:
     with urlopen(url+'/health',timeout=2) as response:
      if response.status==200:state['health']=json.load(response);break
    except OSError:pass
    time.sleep(.25)
   state['status']='running';save();print('ready',label,flush=True)
   with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    barrier=threading.Barrier(3);jobs={e['label']:pool.submit(call,folder,e,barrier,url) for e in fixture['requests'][:3]}
    first_event(folder/'D/request-0.events.jsonl',jobs['D'])
    state['transitions'].append(dict(trigger='D-first-token-all-three-seeds-prefilled',submitted='A',monotonic=time.monotonic()));save()
    entry=fixture['requests'][3];jobs['A']=pool.submit(call,folder,entry,threading.Barrier(1),url)
    first_event(folder/'A/request-0.events.jsonl',jobs['A']);entry=fixture['requests'][4];jobs['E']=pool.submit(call,folder,entry,threading.Barrier(1),url)
    state['transitions'].append(dict(trigger='A-first-token',submitted='E',monotonic=time.monotonic()));save();print(label,'queued E after first A token',flush=True)
    previous='E'
    for entry in fixture['requests'][5:]:
     jobs[previous].result();state['transitions'].append(dict(trigger=previous+'-complete',submitted=entry['label'],monotonic=time.monotonic()));save();jobs[entry['label']]=pool.submit(call,folder,entry,threading.Barrier(1),url);previous=entry['label'];print(label,'submitted',entry['label'],flush=True)
    results={name:future.result() for name,future in jobs.items()};state['completed_requests']=len(results)
   state['status']='collected';save();print('collected',label,flush=True)
  finally:
   state['shutdown']=stop_owned(process,graceful_native=True);log.close();save();print('closed',label,state['shutdown'],flush=True)
 report['status']='collected_pending_independent_audit';save()
except BaseException as e:report.update(status='failed',error=repr(e));save();raise
print('native HTTP diagnostic campaign collected',flush=True)
