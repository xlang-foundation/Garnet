"""One supervised ordinary/MTP HTTP study after the actual model gate closes."""
from pathlib import Path
import hashlib,json,os,shutil,socket,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');REPO=ROOT/'Garnet';M=REPO/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(M));sys.path.insert(0,str(REPO))
from run_logged import run
from run_bf16_low_batch_native_build import idle
from install_mmmu_quality_corpus import install
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();NAME='bf16-low-batch-http-study-attempt1';OUT=ROOT/'results'/NAME
prep=M/'evidence/bf16-low-batch-http-preparation.json';p=json.loads(prep.read_bytes());archive=ROOT/'results'/(NAME+'-evidence.zip')
assert not OUT.exists() and not archive.exists()
for n,h in p['files'].items():assert sha(ROOT/n)==h,('Source/method/input drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(ROOT/'build/bin'/n)==h,('Runtime drift',n)
parent='bf16-low-batch-model-consistency-attempt2';proofpath=ROOT/'results'/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes());closed=json.loads((ROOT/'results'/parent/'workflow.json').read_bytes())
assert sha(proofpath)==p['parent_proof_sha256'] and proof['status']=='bf16_low_batch_model_consistency_replayed' and sha(ROOT/'results'/(parent+'-evidence.zip'))==proof['archive_sha256']==p['parent_archive_sha256']
assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']==p['runtime_dependencies']
library=ROOT/'build/bin/garnet.dll';digest=sha(library);assert digest==proof['library_sha256']==sha(ROOT/'build/bin/garnet-native-bf16-low-batch-attempt1.dll')
receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root']);assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
with socket.socket() as endpoint:endpoint.bind(('127.0.0.1',18082))
def retire(before,started):
 resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT);retired=[]
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
idle();OUT.mkdir();os.chdir(REPO);code=1;corpus=ROOT/'tools/mmmu-quality-corpus'
w={'status':'setup','preparation_sha256':sha(prep),'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'parent_proof_sha256':sha(proofpath),'parent_archive_sha256':proof['archive_sha256'],'library_sha256':digest,'variants':[],'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
def save():(OUT/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[ROOT/n for n in p['files']]+[prep,proofpath];save()
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_low_batch_worker_contract','test2026.cuda.qwen38_5090.test_mmmu_http_quality','test2026.cuda.qwen38_5090.test_mmmu_quality_protocol','test2026.cuda.qwen38_5090.test_fixed_context_native_stream'],OUT/'cpu-ledger','correctness','29CPU worker/request/token notification, raw screening and4Kstream-schema mutation controls',common)
 if code:raise RuntimeError('HTTP CPU controls failed')
 w['corpus_install']=install(ROOT/p['corpus_archive'],corpus);save()
 for mode in ('ordinary','mtp'):
  idle();before={f.name for f in cache.iterdir()};started=time.time();entry={'mode':mode,'finished':False};w['variants'].append(entry);w['status']='serving_'+mode;save()
  try:
   command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(OUT/(mode+'-resources')),'--',sys.executable,str(M/'probe_bf16_low_batch_http.py'),'--root',str(ROOT),'--output',str(OUT/mode),'--corpus',str(corpus),'--mode',mode,'--exploratory-pilot']
   if mode=='mtp':command+=['--ordinary-controls',str(OUT/'ordinary')]
   code=run(command,OUT/(mode+'-ledger'),'correctness','Actual B1 '+mode+' HTTP controls,805screen,4Kcontrols and explicitly exploratory short pilot',common+[library])
   entry.update(exit_code=code,finished=True);save()
   if code:raise RuntimeError('HTTP study failed: '+mode)
  finally:idle();entry['retired_private_engine_files']=retire(before,started);save()
 w['status']='bf16_low_batch_http_study_collected_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==digest);save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in OUT.rglob('*') if f.is_file()]+common;raw={f.relative_to(ROOT).as_posix():f.read_bytes() for f in paths};manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'parent_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
raise SystemExit(code)
