"""Sequential B4 smoke and B1 model consistency, with immutable failure archives."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');REPO=ROOT/'Garnet';M=REPO/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(M));from run_logged import run
from run_bf16_low_batch_native_build import idle
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
NAME='tokenizer-nfc-model-consistency-attempt1';OUT=ROOT/'results'/NAME
prep=M/'evidence/tokenizer-nfc-model-preparation.json';p=json.loads(prep.read_bytes())
assert not OUT.exists() and not (ROOT/'results'/(NAME+'-evidence.zip')).exists()
for n,h in p['files'].items():assert sha(ROOT/n)==h,('Source/input/method drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(ROOT/'build/bin'/n)==h,('Runtime drift',n)
for parent,row in p['parents'].items():
 proof=ROOT/'results'/(parent+'-verified.json');q=json.loads(proof.read_bytes());closed=json.loads((ROOT/'results'/parent/'workflow.json').read_bytes())
 assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(parent+'-evidence.zip'))==row['archive_sha256']==q['archive_sha256']
 assert closed['cleanup_finished'] and closed['exit_code']==0
library=ROOT/'build/bin/garnet.dll';digest=sha(library);assert digest==sha(ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll')
receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
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
idle();OUT.mkdir();os.chdir(REPO);code=1
w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'runtime_dependencies':p['runtime_dependencies'],'parents':p['parents'],'library_sha256':digest,'variants':[],'cleanup_finished':False,'scope':p['scope']}
def save():(OUT/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[ROOT/n for n in p['files']]+[prep,receipt]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']];save()
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_packed_state_gate','test2026.cuda.qwen38_5090.test_bf16_model_state_witness','test2026.cuda.qwen38_5090.test_bf16_same_precision_gate','test2026.cuda.qwen38_5090.test_bf16_padded_state_gate','test2026.cuda.qwen38_5090.test_tokenizer_model_parent'],OUT/'cpu-ledger','correctness','New packed representation controls and unchanged reference/witness guards',common)
 if code:raise RuntimeError('Packed-state checker CPU controls failed')
 for label,batch in [('b4-smoke',4),('b1',1)]:
  idle();before={f.name for f in cache.iterdir()};started=time.time();entry={'label':label,'physical_batch':batch,'finished':False};w['variants'].append(entry);save()
  try:
   code=run([sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(OUT/(label+'-resources')),'--','cmd.exe','/d','/c',str(M/'test_bf16_low_batch_model_fixed.bat'),str(OUT/label),str(cache),str(batch)],OUT/(label+'-ledger'),'correctness','Retained B4 smoke' if batch==4 else 'Actual B1 ordinary/MTP/prefix/reset consistency; not HTTP or speed',common+[library])
   entry.update(exit_code=code,finished=True);save()
   if code:raise RuntimeError('Native model consistency failed: '+label)
   report=json.loads((OUT/label/'report.json').read_bytes());assert report['status']==('bf16_low_batch_b4_smoke_finished' if batch==4 else 'bf16_low_batch_model_consistency_finished')
   from compare_tokenizer_model_parent import compare_reports
   parent_folder=ROOT/'results/bf16-low-batch-model-consistency-attempt2'/label
   comparison=compare_reports(report,json.loads((parent_folder/'report.json').read_bytes()),lambda n:(OUT/label/n).read_bytes(),lambda n:(parent_folder/n).read_bytes())
   (OUT/label/'parent8ae-comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf8');entry['prior_8ae_comparison']=comparison;save()
  finally:idle();entry['retired_private_engine_files']=retire(before,started);save()
 w['status']='tokenizer_nfc_model_consistency_collected_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==digest);save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in OUT.rglob('*') if f.is_file()]+common;raw={f.relative_to(ROOT).as_posix():f.read_bytes() for f in paths}
 manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'scope':p['scope']};archive=ROOT/'results'/(NAME+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
raise SystemExit(code)
