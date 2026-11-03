"""Finite native HTTP routing/refill/prefix study, not a performance campaign."""
import argparse,concurrent.futures,hashlib,json,os,shutil,signal,subprocess,sys,threading,time,urllib.request,urllib.error
from pathlib import Path
from run_local_coder_baseline import engine_processes

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','build','checkpoint','profile','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8129);args=parser.parse_args()
    live=engine_processes()
    if live:raise RuntimeError('Inference engine already live: '+str(live))
    if args.output.exists():raise ValueError('Study output already exists')
    cases=json.loads(args.cases.read_text());assert len(cases)==3
    args.output.mkdir(parents=True);repo=Path(__file__).resolve().parents[2];frozen=args.output/'frozen';frozen.mkdir()
    for source in (args.profile,args.cases,Path(__file__),Path(__file__).with_name('local_coder_batch_http.py'),Path(__file__).with_name('local_coder_batch_transport.py'),Path(__file__).with_name('local_coder_batch_worker.py'),Path(__file__).with_name('run_local_coder_baseline.py'),repo/'test2026/cuda/qwen38_5090/native_http_bridge.py'):
        shutil.copy2(source,frozen/source.name)
    manifest=dict(scope=__doc__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()},library_sha256=hashlib.sha256((args.build/'libgarnet.dylib').read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='running',responses=[],issues=[])
    def save():(args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();url='http://127.0.0.1:'+str(args.port)
    command=[sys.executable,str(frozen/'local_coder_batch_http.py'),'--checkpoint',str(args.checkpoint),'--first-token-timeout',str(json.loads(args.profile.read_text()).get('first_token_timeout_seconds',3600)),'--log-root',str(args.output/'server'),'--port',str(args.port),'--',str(args.runtime),'-B',str(frozen/'local_coder_batch_worker.py'),str(args.build),str(repo),str(args.checkpoint),str(frozen/args.profile.name)]
    # Frozen sibling modules take precedence; retain repository imports for
    # model factories and other runtime dependencies fixed by the source commit.
    environment=dict(os.environ);environment['PYTHONPATH']=str(repo/'test2026/metal')+os.pathsep+str(repo/'test2026/cuda/qwen38_5090')
    serverlog=(args.output/'server.log').open('wb');process=subprocess.Popen(command,stdout=serverlog,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
    def health():
        with urllib.request.urlopen(url+'/health',timeout=2) as response:return json.load(response)
    def call(case,label,first=None):
        path=args.output/label;path.mkdir();body=case['body'];(path/'request.json').write_text(json.dumps(body,indent=2)+'\n')
        events=[];done=False;began=time.monotonic()
        request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=600) as response,(path/'events.jsonl').open('w') as raw:
            for line in response:
                text=line.decode().strip()
                if not text.startswith('data: '):continue
                payload=text[6:];raw.write(json.dumps(dict(received=time.monotonic(),data=payload))+'\n');raw.flush()
                if payload=='[DONE]':done=True;break
                event=json.loads(payload)
                if 'error' in event:raise RuntimeError(event['error'])
                events.append(event)
                if 'token_id' in event and first is not None:first.set()
        assert done and events
        last=events[-1];ids=[event['token_id'] for event in events if 'token_id' in event]
        assert ids==case['expected_ids'] and ids==last['response_token_ids']
        assert last['prompt_token_ids']==case['prompt_ids']
        assert last['usage']['completion_tokens']==len(ids) and last['usage']['prompt_tokens']==len(case['prompt_ids'])
        cached=last['usage']['prompt_tokens_details']['cached_tokens'];assert 0<=cached<=len(case['prompt_ids'])
        result=dict(label=label,http_id=last['id'],generated_ids=ids,cached_tokens=cached,total_seconds=time.monotonic()-began)
        (path/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
    try:
        deadline=time.monotonic()+600
        while True:
            if process.poll() is not None:raise RuntimeError('Native HTTP process exited before ready')
            try:ready=health();break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>=deadline:raise TimeoutError('Native batch startup deadline')
                time.sleep(.5)
        report['health']=ready;save()
        first=threading.Event()
        with concurrent.futures.ThreadPoolExecutor(3) as executor:
            a=executor.submit(call,cases[0],'A',first);b=executor.submit(call,cases[1],'B')
            if not first.wait(timeout=600):raise TimeoutError('A produced no first token')
            c=executor.submit(call,cases[2],'C')
            report['responses']=[a.result(),b.result(),c.result()];save()
        repeated=call(cases[2],'C-returned');assert repeated['cached_tokens']==len(cases[2]['prompt_ids'])
        report['responses'].append(repeated)
        # Context rejection is isolated to that request; the worker stays ready.
        invalid=dict(cases[0]['body']);invalid['max_tokens']=1024
        request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(invalid).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=10) as response:rejection=response.read().decode()
        (args.output/'rejected-profile.sse').write_text(rejection);assert 'exceeds' in rejection
        assert health()['native_worker']['prefix_cache'] is True
        report['status']='collected_pending_native_audit'
    except Exception as error:
        report['status']='failed';report['issues'].append(repr(error));raise
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                report['issues'].append('Forced shutdown after owned process deadline');os.killpg(process.pid,signal.SIGKILL);process.wait()
        report['server_exit_code']=process.returncode;serverlog.close();save()
    return 0
if __name__=='__main__':sys.exit(main())
