"""Own one exclusive SGLang reference attempt and retain its actual lifecycle.

The HTTP client records timing. This controller's process durations are not
model latency. Never run alongside another model, transfer, or replay job.
"""
from pathlib import Path,PureWindowsPath
import argparse,hashlib,json,os,re,socket,subprocess,sys,time,zipfile
from urllib.request import urlopen

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def job_path(root,name):
    if not re.fullmatch(r'sglang-fixed4k-(ordinary|mtp)-attempt[1-9][0-9]*',name):
        raise ValueError('A fresh named SGLang study attempt is required')
    return root/'results'/name

def linux_path(path):
    path=PureWindowsPath(path)
    if path.drive.upper()!='D:':raise ValueError('Reference files must stay on the fixed D: study')
    return '/mnt/d/'+Path(*path.parts[1:]).as_posix()

def wait_ready(process,url,deadline,*,now=time.monotonic,pause=time.sleep,open_url=urlopen):
    """Observe the owned process and endpoint without creating a second server."""
    while True:
        code=process.poll()
        if code is not None:raise RuntimeError('Owned reference exited before readiness: '+str(code))
        try:
            with open_url(url+'/health',timeout=5) as response:
                if response.status==200:return response.read()
        except OSError:pass
        if now()>=deadline:raise TimeoutError('Readiness observation expired; owned process requires cleanup')
        pause(5)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,required=True);a.add_argument('--job',required=True)
    a.add_argument('--mode',choices=('ordinary','mtp'),required=True)
    a.add_argument('--preparation',type=Path,required=True);a.add_argument('--ordinary',type=Path)
    args=a.parse_args();root=args.root.resolve();out=job_path(root,args.job)
    assert root==Path('D:/CantorAI/5090-study-20261023').resolve()
    methods=Path(__file__).resolve().parent;repo=root/'Garnet'
    if (args.mode=='mtp')!=(args.ordinary is not None):raise ValueError('MTP requires the closed ordinary attempt')
    if not args.job.startswith('sglang-fixed4k-'+args.mode+'-'):raise ValueError('Job name and mode differ')
    parent='fixed-length-http-study-attempt1';proofpath=root/'results'/(parent+'-verified.json')
    if not proofpath.is_file():
        print(json.dumps({'status':'sglang_fixed4k_native_proof_missing_no_start_no_output'}));return 2
    proof=json.loads(proofpath.read_bytes());closed=json.loads((root/'results'/parent/'workflow.json').read_bytes())
    assert proof['status']=='fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed']
    assert sha(root/'results'/(parent+'-evidence.zip'))==proof['archive_sha256']
    assert closed['cleanup_finished'] and closed['exit_code']==0 and closed['library_unchanged']
    p=json.loads(args.preparation.read_bytes());assert p['status']=='sglang_fixed4k_run_prepared_unrun'
    assert p['native_proof_sha256']==sha(proofpath) and p['native_archive_sha256']==proof['archive_sha256']
    for name,digest in p['files'].items():assert sha(root/name)==digest,('Study method/source drift',name)
    for name,digest in p['runtime_dependencies'].items():assert sha(root/'build/bin'/name)==digest
    ordinary_proof=None
    if args.ordinary is not None:
        args.ordinary=args.ordinary.resolve();assert args.ordinary.parent==root/'results'
        ordinary_proofpath=root/'results'/(args.ordinary.name+'-verified.json')
        ordinary_proof=json.loads(ordinary_proofpath.read_bytes())
        assert ordinary_proof['status']=='reference_fixed4k_profile_replayed' and ordinary_proof['engine']=='sglang' and ordinary_proof['mode']=='ordinary'
        assert sha(root/'results'/(args.ordinary.name+'-evidence.zip'))==ordinary_proof['archive_sha256']
        ow=json.loads((args.ordinary/'workflow.json').read_bytes());assert ow['cleanup_finished'] and ow['exit_code']==0
    archive=root/'results'/(args.job+'-evidence.zip');assert not out.exists() and not archive.exists()
    sys.path.insert(0,str(methods));from run_logged import run
    from run_bf16_low_batch_native_build import idle
    idle()
    with socket.socket() as endpoint:endpoint.bind(('127.0.0.1',18081))
    out.mkdir();os.chdir(repo);server=None;code=1
    launcher=methods/('serve_sglang_fixed4k.sh' if args.mode=='ordinary' else 'serve_sglang_mtp3_fixed4k.sh')
    w={'status':'starting','engine':'sglang','mode':args.mode,'preparation_sha256':sha(args.preparation),
       'native_proof_sha256':sha(proofpath),'native_archive_sha256':proof['archive_sha256'],
       'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],
       'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    common=[root/name for name in p['files']]+[args.preparation,proofpath]
    if ordinary_proof is not None:
        w['ordinary_proof_sha256']=sha(ordinary_proofpath);w['ordinary_archive_sha256']=ordinary_proof['archive_sha256'];common.append(ordinary_proofpath)
    def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
    save()
    try:
        code=run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','Reference identity, full-length IDs, cache and stream-accounting controls',common)
        if code:raise RuntimeError('Reference CPU controls failed')
        command=['wsl.exe','-d','Ubuntu-24.04','--','env',
                 'GARNET_REF_PIDFILE='+linux_path(out/'server.pid'),
                 'GARNET_REF_LOGFILE='+linux_path(out/'server.log'),'bash',linux_path(launcher)]
        w['server_command']=command;save()
        with (out/'wsl-wrapper.log').open('xb',buffering=0) as wrapper:
            server=subprocess.Popen(command,stdout=wrapper,stderr=subprocess.STDOUT)
            w['owned_wsl_wrapper_pid']=server.pid;save()
            (out/'health.bin').write_bytes(wait_ready(server,'http://127.0.0.1:18081',time.monotonic()+5400))
            code=run([sys.executable,str(methods/'prepare_sglang_fixed4k_metadata.py'),'--root',str(root),'--job',str(out),'--mode',args.mode],out/'metadata-ledger','correctness','Actual ready-server identity, owned PID and named installed sources',common)
            if code:raise RuntimeError('Actual SGLang identity/source verification failed')
            w['status']='client_running';save()
            command=[sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',sys.executable,str(methods/'probe_reference_fixed4k.py'),'--root',str(root),'--engine','sglang','--mode',args.mode,'--output',str(out/'client'),'--metadata',str(out/'metadata.json'),'--launcher',str(launcher),'--server-log',str(out/'server.log')]
            if args.ordinary is not None:command+=['--ordinary',str(args.ordinary/'client')]
            code=run(command,out/'client-ledger','correctness','Actual untimed controls/805screen then124frozen4K controls and124timedstreams; timings belong to HTTP harness',common)
            if code:raise RuntimeError('Reference controls or fixed4K client failed')
            w['status']='reference_fixed4k_collected_needs_replay'
    except BaseException as error:
        w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w['exit_code']=code;save()
        if server is not None:
            if server.poll() is None:
                pid=int((out/'server.pid').read_text().strip())
                stop=['wsl.exe','-d','Ubuntu-24.04','--','/usr/bin/python3',linux_path(methods/'stop_reference_server.py'),'sglang','--pid',str(pid)]
                stop_code=run(stop,out/'stop-ledger','setup','SIGTERM only after actual owned reference PID/argv verification',common)
                w['stop_exit_code']=stop_code;save()
                if stop_code:raise RuntimeError('Owned reference cleanup signal failed; inspect original server, do not restart')
            # A wait timeout is an observation failure, never a termination claim.
            w['server_exit_code']=server.wait(timeout=600);save()
        idle()
        w['source_after_sha256']={name:sha(root/name) for name in p['files']}
        w['runtime_dependencies_after']={name:sha(root/'build/bin'/name) for name in p['runtime_dependencies']}
        assert w['source_after_sha256']==w['source_sha256'] and w['runtime_dependencies_after']==w['runtime_dependencies']
        w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common
        raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
        manifest={'files':{name:hashlib.sha256(data).hexdigest() for name,data in raw.items()},'native_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,data in raw.items():z.writestr(name,data)
            z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
