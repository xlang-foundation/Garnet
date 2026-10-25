"""Guarded BF16 lifecycle candidate; parent corpus must be closed and replayed."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-padded-lifecycle-scope-fixed-attempt2';out=results/name;assert not out.exists()
prep_path=methods/'evidence/bf16-padded-lifecycle-scope-fixed-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in p['source_sha256'].items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
for n,h in p['methods'].items():assert sha((methods/n).read_bytes())==h,('Method drift',n)
for n,h in p['pinned_inputs'].items():assert sha((root/n).read_bytes())==h,('Input drift',n)
parent='bf16-same-precision-mtp-attempt1';proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert proof['status']=='bf16_same_precision_mtp_corpus_replayed' and len(proof['cases'])==4 and proof['raw_hidden_and_normalized_arrays']==20 and not proof['historical_fp32_gate_reclassified']
assert sha(proofpath.read_bytes())==p['parent_proof_sha256'] and sha((methods/'verify_bf16_same_precision_mtp.py').read_bytes())==proof['replay_verifier_sha256']
for n,h in p['linked_archives'].items():assert sha((results/n).read_bytes())==h
assert p['linked_archives'][parent+'-evidence.zip']==proof['archive_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished'] and previous['source_sha256']==previous['source_after_sha256']==p['source_sha256']
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']==p['library_sha256']
fixture=results/'bf16-kv-metadata-fixture.json';assert sha(fixture.read_bytes())==p['fixture_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
prior=json.loads((results/'bf16-padded-lifecycle-attempt1/workflow.json').read_bytes())
assert prior['exit_code']!=0 and prior['cleanup_finished'] and prior['library_unchanged']
priorproof=results/'bf16-padded-lifecycle-attempt1-verified.json'
assert sha(priorproof.read_bytes())==p['preserved_failure_proof_sha256']
assert json.loads(priorproof.read_bytes())['status']=='bf16_padded_lifecycle_failure_preserved'
idle();out.mkdir();before={f.name for f in cache.iterdir()};started=time.time();os.chdir(repo)
w={'status':'running','source_sha256':p['source_sha256'],'library_sha256':digest,'preparation_sha256':sha(prep_path.read_bytes()),
   'parent_proof_sha256':sha(proofpath.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],'fixture_sha256':sha(fixture.read_bytes()),'cleanup_finished':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();code=1
try:
 code=run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_bf16_model_state_witness','test2026.cuda.qwen38_5090.test_bf16_same_precision_gate','test2026.cuda.qwen38_5090.test_bf16_padded_state_gate'],out/'cpu-ledger','correctness','Complete and explicitly selected-row state witness CPU controls',[prep_path]+[methods/n for n in p['methods']])
 if code:raise RuntimeError('Same-precision state CPU controls failed')
 code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(methods/'test_bf16_padded_lifecycle_scope_fixed.bat'),str(out/'native'),str(cache)],out/'ledger','correctness','BF16 private padding, unequal prefix commit, rollback, reset and early retirement',
  [library,prep_path,proofpath,receipt,fixture,Path(__file__)]+[repo/n for n in p['source_sha256']]+[methods/n for n in p['methods']]+[root/n for n in p['pinned_inputs']])
 if code:raise RuntimeError('Packed same-precision ordinary/MTP gate failed')
 w['status']='bf16_padded_lifecycle_finished_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,library_unchanged=sha(library.read_bytes())==digest,source_after_sha256={n:sha((repo/n).read_bytes()) for n in p['source_sha256']});save()
 idle();retired=[];resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root)
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
 w.update(retired_private_engine_files=retired,cleanup_finished=True);save()
 files=[f for f in out.rglob('*') if f.is_file()]+[repo/n for n in p['source_sha256']]+[methods/n for n in p['methods']]+[root/n for n in p['pinned_inputs']]+[prep_path,proofpath,receipt,fixture,priorproof]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files};manifest={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':p['linked_archives'],'scope':p['scope']};archive=results/(name+'-evidence.zip');assert not archive.exists()
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,v in raw.items():z.writestr(n,v)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
