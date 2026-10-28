"""Full14case parity twice in independent disabled/enabled native processes."""
from pathlib import Path
import base64,hashlib,json,os,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';m=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(m));from run_logged import run
from integrated_top1_model_replay import compare
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();name='integrated-batch-top1-model-lifecycle-attempt1';out=results/name
prep=m/'evidence/integrated-top1-model-preparation.json';p=json.loads(prep.read_bytes())
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
for n,h in p['files'].items():assert sha(root/n)==h,('Source/method/input drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h,('Runtime drift',n)
for parent,entry in p['parents'].items():
 proof=results/(parent+'-verified.json');r=json.loads(proof.read_bytes());assert sha(proof)==entry['proof_sha256'] and r['status']==entry['status'] and sha(results/(parent+'-evidence.zip'))==r['archive_sha256']==entry['archive_sha256']
library=root/'build/bin/garnet.dll';digest=sha(library)
assert digest==p['runtime_dependencies']['garnet.dll']==sha(root/'build/bin/garnet-native-integrated-batch-top1-attempt1.dll')
parent='bf16-padded-lifecycle-scope-fixed-attempt2';parent_archive=results/(parent+'-evidence.zip')
with zipfile.ZipFile(parent_archive) as z:
 prefix='results/'+parent+'/native/';old=json.loads(z.read(prefix+'report.json'));old_vectors={n.removeprefix(prefix):z.read(n) for n in z.namelist() if n.startswith(prefix) and n.endswith('.f32')}
assert sha(root/'build/bin/garnet-native-bf16-float-reference-attempt1.dll')==old['garnet_dll_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
 command=r"$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\CantorAI\5090-study-20261023\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
 live=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip();assert not live,('Study executable live',live)
def retire(before,started):
 resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root);retired=[]
 for target in cache.iterdir():
  if target.name in before or not target.name.startswith('garnet-cuda-graph-'):continue
  st=target.lstat();assert not(getattr(st,'st_file_attributes',0)&0x400) and target.resolve(strict=True).parent==resolved and st.st_ctime>=started-2
  entries=list(target.iterdir());assert all(f.name in ('model.engine','model.engine.tmp') for f in entries)
  for f in entries:
   fs=f.lstat();assert f.is_file() and not(getattr(fs,'st_file_attributes',0)&0x400) and f.resolve(strict=True).parent==target.resolve(strict=True)
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
   retired.append({'path':str(f),'bytes':fs.st_size,'sha256':h.hexdigest()})
  for f in entries:f.unlink()
  target.rmdir()
 return retired
idle();out.mkdir();os.chdir(repo);code=1
w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'library_sha256':digest,'runtime_dependencies':p['runtime_dependencies'],'variants':[],'cleanup_finished':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in p['files']]+[prep,receipt]+[results/(parent+'-verified.json') for parent in p['parents']]
save()
try:
 code=run([sys.executable,str(m/'test_integrated_top1_model_replay.py'),str(parent_archive)],out/'checker-ledger','correctness','Seven synthetic mutation controls over the actual closed parent fixture',common)
 if code:raise RuntimeError('Full-corpus checker mutation controls failed')
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_model_state_witness','test2026.cuda.qwen38_5090.test_bf16_same_precision_gate','test2026.cuda.qwen38_5090.test_bf16_padded_state_gate'],out/'cpu-ledger','correctness','Unchanged complete and selected-row logical state CPU controls',common)
 if code:raise RuntimeError('State witness CPU controls failed')
 for label,mode in [('disabled','0'),('enabled','1')]:
  idle();before={f.name for f in cache.iterdir()};started=time.time();entry={'label':label,'mode':mode,'finished':False};w['variants'].append(entry);save()
  try:
   code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/(label+'-resources')),'--','cmd.exe','/d','/c',str(m/'test_integrated_top1_model.bat'),str(out/label),str(cache),mode],out/(label+'-ledger'),'correctness','Complete14case/102raw-vector parent parity and actual native dispatch mode '+label,common+[library])
   entry.update(exit_code=code,finished=True);save()
   if code:raise RuntimeError('Full model gate failed: '+label)
   native=out/label;new=json.loads((native/'report.json').read_bytes());vectors={f.name:f.read_bytes() for f in native.glob('*.f32')}
   parity=compare(old,new,old_vectors,vectors,digest,mode);(out/(label+'-parent-parity.json')).write_text(json.dumps(parity,indent=2),encoding='utf8');entry['parent_parity_passed']=True;save()
  finally:idle();entry['retired_private_engine_files']=retire(before,started);save()
 w['status']='integrated_top1_full_model_lifecycle_finished_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==digest);save()
 idle();w['cleanup_finished']=True;save()
 paths=[f for f in out.rglob('*') if f.is_file()]+common;raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
 manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'linked_archives':{parent+'-evidence.zip':entry['archive_sha256'] for parent,entry in p['parents'].items()},'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
