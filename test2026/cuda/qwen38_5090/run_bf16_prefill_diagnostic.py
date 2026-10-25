"""Three sequential teacher-forced controls after the preserved BF16 mismatch."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-prefill-diagnostic-attempt1';out=results/name;assert not out.exists()
proofpath=results/'bf16-model-b4-t32-attempt1-verified.json';proof=json.loads(proofpath.read_bytes())
assert proof['status']=='bf16_model_b4_t32_failure_preserved' and proof['ordinary_mtp_raw_hidden_exact']
assert sha((results/'bf16-model-b4-t32-attempt1-evidence.zip').read_bytes())==proof['archive_sha256']
assert sha((methods/'verify_bf16_model_b4_t32.py').read_bytes())==proof['replay_verifier_sha256']
parent=results/'bf16-model-b4-t32-attempt1';previous=json.loads((parent/'workflow.json').read_bytes());assert previous['exit_code']==1 and previous['cleanup_finished']
events=[json.loads(l) for l in (parent/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==1
sources=previous['source_sha256'];prep_path=methods/'evidence/bf16-prefill-diagnostic-preparation.json';prep=json.loads(prep_path.read_bytes())
sources={**sources,**prep['additional_source_sha256']}
for n,h in sources.items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
for n,h in prep['methods'].items():assert sha((methods/n).read_bytes())==h,('Method drift',n)
for n,h in prep['pinned_inputs'].items():assert sha((root/n).read_bytes())==h,('Input drift',n)
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']==previous['library_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time();os.chdir(repo)
w={'status':'running','source_sha256':sources,'library_sha256':digest,'preparation_sha256':sha(prep_path.read_bytes()),
   'parent_proof_sha256':sha(proofpath.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],'profiles':[],
   'cleanup_finished':False,'scope':prep['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();code=1
try:
 for mode in ('fp32-t4','bf16-t4','bf16-t32'):
  idle();folder=out/mode;folder.mkdir()
  code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(folder/'resources'),'--','cmd.exe','/d','/c',str(methods/'test_bf16_prefill_diagnostic.bat'),mode,str(folder/'native'),str(cache)],folder/'ledger','correctness','teacher-forced BF16 precision/prefill diagnostic '+mode,
   [library,prep_path,proofpath,receipt,Path(__file__)]+[repo/n for n in sources]+[methods/n for n in prep['methods']]+[root/n for n in prep['pinned_inputs']])
  w['profiles'].append({'mode':mode,'exit_code':code});save();idle()
  if code:raise RuntimeError('Diagnostic control failed: '+mode)
 w['status']='bf16_prefill_diagnostic_finished_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,library_unchanged=sha(library.read_bytes())==digest,source_after_sha256={n:sha((repo/n).read_bytes()) for n in sources});save()
 idle();retired=[];resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root)
 for target in cache.iterdir():
  if target.name in before or not target.name.startswith('garnet-cuda-graph-'):continue
  st=target.lstat();assert not(getattr(st,'st_file_attributes',0)&0x400) and target.resolve(strict=True).parent==resolved and st.st_ctime>=started-2
  entries=list(target.iterdir());assert all(p.name in ('model.engine','model.engine.tmp') for p in entries)
  for p in entries:
   ps=p.lstat();assert p.is_file() and not(getattr(ps,'st_file_attributes',0)&0x400) and p.resolve(strict=True).parent==target.resolve(strict=True)
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8<<20),b''):h.update(block)
   retired.append({'path':str(p),'bytes':ps.st_size,'sha256':h.hexdigest()})
  for p in entries:p.unlink()
  target.rmdir()
 w.update(retired_private_engine_files=retired,cleanup_finished=True);save()
print(json.dumps(w,indent=2));raise SystemExit(code)
