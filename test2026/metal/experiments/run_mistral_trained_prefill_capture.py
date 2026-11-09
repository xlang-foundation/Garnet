from pathlib import Path
import json,subprocess,threading,queue,time,os,signal,sys
s=Path(__file__).resolve().parent;w=Path('/Users/shawn/CantorAI');b=json.loads((s/'bindings1.json').read_text());fixture=json.loads((s/'fixture1.json').read_text());status={'status':'starting','waves':[]}
def save():(s/'status1.json').write_text(json.dumps(status,indent=2)+'\n')
# Capture/control are diagnostics, never admitted as performance observations.
for label in ['control','capture']:
 cmd=[b['xlang'],'-B',str(s/'worker1.py'),b['library'],b['source'],str(w/'out/models/Mistral-7B-Instruct-v0.3-MLX-4bit'),str(s/(label+'-profile1.json'))]
 (s/(label+'-command1.json')).write_text(json.dumps({'argv':cmd,'environment_overrides':{'MTL_CAPTURE_ENABLED':'1','GARNET_METAL_PROFILE_OPS':'0','PYTHONDONTWRITEBYTECODE':'1'}},indent=2)+'\n')
 row={'label':label,'status':'starting','completed':[]};status['waves'].append(row);save();events=queue.Queue();out=(s/(label+'-events1.jsonl')).open('x');err=(s/(label+'-stderr1.log')).open('x')
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=dict(os.environ,MTL_CAPTURE_ENABLED='1',GARNET_METAL_PROFILE_OPS='0',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True,bufsize=0);row['pid']=p.pid;save();print(label,'starting',p.pid,flush=True)
 def reader():
  try:
   for raw in iter(p.stdout.readline,b''):
    line=raw.decode('utf8');out.write(line);out.flush()
    try:events.put(json.loads(line))
    except ValueError:pass
  finally:events.put({'event':'process_stdout_eof'})
 t=threading.Thread(target=reader,daemon=True);t.start();deadline=time.monotonic()+600;ready=False;done={}
 try:
  while len(done)<4:
   if time.monotonic()>deadline:raise TimeoutError('Diagnostic native collection deadline exceeded')
   try:event=events.get(timeout=1)
   except queue.Empty:
    if p.poll() is not None:raise RuntimeError('Native worker exited before completion: '+str(p.returncode))
    continue
   kind=event.get('event')
   if kind=='ready':
    assert not ready;ready=True;row['status']='collecting';save()
    requests=[dict(id='trained-'+str(x['row']),body=dict(rendered_prompt=x['rendered_prompt'],max_tokens=8)) for x in fixture['requests']]
    payload=(''.join(json.dumps(x)+'\n' for x in requests)).encode();p.stdin.write(payload);p.stdin.flush();print(label,'ready; all4 requests submitted',flush=True)
   elif kind=='error':raise RuntimeError('Native worker error: '+repr(event))
   elif kind=='completed':
    key=event['id'];assert key not in done;done[key]=event;row['completed'].append({'id':key,'tokens':len(event['generated_ids'])});save();print(label,'completed',key,len(event['generated_ids']),flush=True)
   elif kind=='diagnostic_prefill_capture_boundary':print(label,'reached capture boundary16',flush=True)
   elif kind=='diagnostic_prefill_capture_stopped':print(label,'capture document stopped',flush=True)
   elif kind=='process_stdout_eof':raise RuntimeError('Native stdout ended before completions')
  (s/(label+'-results1.json')).write_text(json.dumps(done,indent=2)+'\n');p.stdin.close();p.wait(timeout=90);t.join(timeout=10);assert p.returncode==0,p.returncode;row.update(status='completed',returncode=p.returncode);save()
 except BaseException as e:
  row.update(status='failed',error=repr(e));save();raise
 finally:
  if p.poll() is None:
   p.stdin.close()
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  t.join(timeout=10);out.close();err.close();row['final_returncode']=p.returncode;save();print(label,'closed',p.returncode,flush=True)
status['status']='collected_pending_audit';save()
