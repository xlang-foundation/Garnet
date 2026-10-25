"""Own exclusive SGLang correctness/overlap characterization, never benchmark."""
from pathlib import Path
import argparse,hashlib,json,os,re,socket,subprocess,sys,time,zipfile
if __package__:
 from .run_sglang_fixed4k import wait_ready,linux_path
 from .run_mtp_prefill16_model import closed_native_admission
else:
 from run_sglang_fixed4k import wait_ready,linux_path
 from run_mtp_prefill16_model import closed_native_admission
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def job_path(root,name):
 if not re.fullmatch(r'sglang-output-characterization-attempt[1-9][0-9]*',name):raise ValueError('A fresh named output characterization attempt is required')
 return root/'results'/name

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--job',required=True);a.add_argument('--preparation',type=Path,required=True);a.add_argument('--closed-native-archive-sha256',required=True);args=a.parse_args();root=args.root.resolve();m=Path(__file__).resolve().parent
 assert root==Path('D:/CantorAI/5090-study-20261023').resolve() and m==root/'Garnet/test2026/cuda/qwen38_5090'
 p=json.loads(args.preparation.read_bytes());assert p['status']=='sglang_output_characterization_prepared_unrun'
 for name,digest in p['files'].items():assert sha(root/name)==digest,('Characterization source/input drift',name)
 for name,digest in p['runtime_dependencies'].items():assert sha(root/'build/bin'/name)==digest
 parent='fixed-length-http-study-attempt1';proofpath=root/'results'/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes());assert proof['status']=='fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed'] and sha(proofpath)==p['native_proof_sha256'] and sha(root/'results'/(parent+'-evidence.zip'))==p['native_archive_sha256']==proof['archive_sha256']
 previous='sglang-fixed4k-ordinary-attempt2';prior=json.loads((root/'results'/(previous+'-partial-verified.json')).read_bytes());assert prior['status']=='reference_fixed4k_attempt2_failure_partial_evidence_replayed' and sha(root/'results'/(previous+'-partial-verified.json'))==p['prior_partial_proof_sha256'] and sha(root/'results'/(previous+'-evidence.zip'))==p['prior_archive_sha256']==prior['archive_sha256']
 native_closed=closed_native_admission(root,args.closed_native_archive_sha256)
 out=job_path(root,args.job);archive=root/'results'/(args.job+'-evidence.zip');assert not out.exists() and not archive.exists()
 sys.path.insert(0,str(m));from run_logged import run
 from run_bf16_low_batch_native_build import idle
 idle()
 with socket.socket() as endpoint:endpoint.bind(('127.0.0.1',18081))
 out.mkdir();os.chdir(root/'Garnet');server=None;code=1;launcher=m/'serve_sglang_reserved_fixed4k.sh'
 common=[root/name for name in p['files']]+[args.preparation,proofpath,root/'results/bf16-batch8-model-admission-attempt1/workflow.json']
 w={'status':'starting','engine':'sglang','mode':'ordinary','preparation_sha256':sha(args.preparation),'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'native_proof_sha256':p['native_proof_sha256'],'native_archive_sha256':p['native_archive_sha256'],'prior_partial_proof_sha256':p['prior_partial_proof_sha256'],'prior_archive_sha256':p['prior_archive_sha256'],'closed_native_admission':native_closed,'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
 def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
 save()
 try:
  code=run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','Synthetic protocol/overlap/stream-color replay plus existing reference/native ownership gates',common)
  if code:raise RuntimeError('Characterization CPU controls failed')
  command=['wsl.exe','-d','Ubuntu-24.04','--','env','GARNET_REF_PIDFILE='+linux_path(out/'server.pid'),'GARNET_REF_LOGFILE='+linux_path(out/'server.log'),'bash',linux_path(launcher)];w['server_command']=command;save()
  with (out/'wsl-wrapper.log').open('xb',buffering=0) as wrapper:
   server=subprocess.Popen(command,stdout=wrapper,stderr=subprocess.STDOUT);w['owned_wsl_wrapper_pid']=server.pid;save();(out/'health.bin').write_bytes(wait_ready(server,'http://127.0.0.1:18081',time.monotonic()+5400))
   code=run([sys.executable,str(m/'prepare_sglang_reserved_fixed4k_metadata.py'),'--root',str(root),'--job',str(out),'--mode','ordinary'],out/'metadata-ledger','correctness','Actual ready SGLang config/checkpoint/named installed sources and owned PID',common)
   if code:raise RuntimeError('Actual characterization server identity failed')
   w['status']='client_running';save()
   command=[sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',sys.executable,str(m/'reference_concurrency_characterization.py'),'--root',str(root),'--output',str(out/'client'),'--metadata',str(out/'metadata.json')]
   code=run(command,out/'client-ledger','correctness','Actual288request/135overlap waves, strict4K counts, ID/text variants and streamed/nonstream color isolation; no benchmark',common)
   if code:raise RuntimeError('Output/overlap/color characterization failed')
   w['status']='sglang_output_characterization_collected_needs_replay'
 except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
 finally:
  w['exit_code']=code;save()
  if server is not None:
   if server.poll() is None:
    pid=int((out/'server.pid').read_text().strip());stop=['wsl.exe','-d','Ubuntu-24.04','--','/usr/bin/python3',linux_path(m/'stop_reference_server.py'),'sglang','--pid',str(pid)]
    w['stop_exit_code']=run(stop,out/'stop-ledger','setup','SIGTERM only actual owned reference parent after PID/cmdline verification',common);save()
    if w['stop_exit_code']:raise RuntimeError('Owned characterization stop failed; inspect original process, do not restart')
   w['server_exit_code']=server.wait(timeout=600);save()
  idle();w['source_after_sha256']={n:sha(root/n) for n in p['files']};w['runtime_dependencies_after']={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']};assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'];w['cleanup_finished']=True;save()
  paths=[f for f in out.rglob('*') if f.is_file()]+common;raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for name,data in raw.items():z.writestr(name,data)
   z.writestr('evidence-manifest.json',json.dumps({'files':{name:hashlib.sha256(data).hexdigest() for name,data in raw.items()},'native_archive_sha256':p['native_archive_sha256'],'prior_archive_sha256':p['prior_archive_sha256'],'scope':p['scope']},indent=2))
  print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
 return code
if __name__=='__main__':raise SystemExit(main())
