"""Launch one native batch worker, collect concurrent frozen coding tasks, stop.

Evaluation and native-event audits run separately after GPU shutdown.
"""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
from run_local_coder_baseline import engine_processes

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','build','checkpoint','profile','manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=8129);args=parser.parse_args()
    live=engine_processes()
    if live:raise RuntimeError('Inference engine already live: '+str(live))
    args.output.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2];source=Path(__file__).parent;frozen=args.output/'frozen';frozen.mkdir()
    for path in (args.profile,args.manifest,Path(__file__),source/'local_coder_batch_http.py',source/'local_coder_batch_worker.py',source/'local_coder_batch_transport.py',source/'run_local_coder_quality.py',source/'local_coder_stream_benchmark.py',repo/'test2026/cuda/qwen38_5090/native_http_bridge.py'):
        shutil.copy2(path,frozen/path.name)
    shutil.copy2(args.build/'libgarnet.dylib',args.output/'libgarnet.dylib')
    manifest=dict(scope=__doc__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir()},library_sha256=hashlib.sha256((args.output/'libgarnet.dylib').read_bytes()).hexdigest())
    (args.output/'artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='starting',issues=[])
    def save():(args.output/'collection.json').write_text(json.dumps(report,indent=2)+'\n')
    save();url='http://127.0.0.1:'+str(args.port)
    environment=dict(os.environ);environment['PYTHONPATH']=str(source)+os.pathsep+str(repo/'test2026/cuda/qwen38_5090')
    command=[sys.executable,str(frozen/'local_coder_batch_http.py'),'--checkpoint',str(args.checkpoint),'--log-root',str(args.output/'server'),'--port',str(args.port),'--',str(args.runtime),'-B',str(frozen/'local_coder_batch_worker.py'),str(args.build),str(repo),str(args.checkpoint),str(frozen/args.profile.name)]
    serverlog=(args.output/'server.log').open('wb');process=subprocess.Popen(command,stdout=serverlog,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
    try:
        deadline=time.monotonic()+600
        while True:
            if process.poll() is not None:raise RuntimeError('Native batch server exited before ready')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as response:report['health']=json.load(response)
                break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>=deadline:raise TimeoutError('Native batch startup deadline')
                time.sleep(.5)
        report['status']='collecting';save()
        with (args.output/'quality-client.log').open('w') as log:
            result=subprocess.run([sys.executable,str(frozen/'run_local_coder_quality.py'),'--manifest',str(frozen/args.manifest.name),
                '--url',url,'--engine-label','native27b-continuous-batch','--output',str(args.output/'responses'),'--parallel-tasks','8'],stdout=log,stderr=subprocess.STDOUT)
        report['client_exit_code']=result.returncode;report['status']='collected_pending_evaluation' if result.returncode==0 else 'client_failure'
    except Exception as error:report['status']='failed';report['issues'].append(repr(error));raise
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                report['issues'].append('Forced owned process shutdown');os.killpg(process.pid,signal.SIGKILL);process.wait()
        report['server_exit_code']=process.returncode;serverlog.close();save()
    return report['client_exit_code']
if __name__=='__main__':sys.exit(main())
