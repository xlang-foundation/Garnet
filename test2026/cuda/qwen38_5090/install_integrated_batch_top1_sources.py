"""Guarded source installation only, after both full lifecycle replays.

No compiler, CUDA initialization, SDK replacement or native runtime build/load.
The runtime preparation must bind the actual independently replayed parent proof;
it is intentionally absent while the parent worker is still running.
"""
from pathlib import Path
import base64,hashlib,json,subprocess,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';m=repo/'test2026/cuda/qwen38_5090'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
name='integrated-batch-top1-source-install-attempt1';out=results/name;assert not out.exists()
prep_path=m/'evidence/integrated-batch-top1-source-install-preparation.json';p=json.loads(prep_path.read_bytes())
parent='bf16-padded-lifecycle-scope-fixed-attempt2';proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert sha(proofpath)==p['parent_proof_sha256'] and proof['status']=='bf16_padded_lifecycle_replayed'
assert len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102 and not proof['performance_admitted']
assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==p['parent_archive_sha256']
assert sha(m/'verify_bf16_padded_lifecycle_scope_fixed.py')==proof['replay_verifier_sha256']
w_parent=json.loads((results/parent/'workflow.json').read_bytes())
assert w_parent['exit_code']==0 and w_parent['cleanup_finished'] and w_parent['library_unchanged']
assert w_parent['source_sha256']==w_parent['source_after_sha256']==p['qualified_parent_source_sha256']
for n,h in p['base_source_sha256'].items():assert sha(repo/n)==h,('Source drift',n)
for n,h in p['methods'].items():assert sha(m/n)==h,('Method drift',n)
dependencies={n:sha(root/'build/bin'/n) for n in ('garnet.dll','xlang3.exe','xlang3_runtime.dll','nvinfer_10.dll','cudart64_13.dll','cublasLt64_13.dll')}
assert dependencies==p['runtime_dependencies'] and dependencies['garnet.dll']==proof['library_sha256']
retained=root/'build/bin/garnet-native-bf16-float-reference-attempt1.dll';assert sha(retained)==dependencies['garnet.dll']
for flag in (b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF'):
 assert flag in (root/'build/CMakeCache.txt').read_bytes()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
# Also reject a study process that happens to be idle at the instant GPU use is read.
command="$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\\CantorAI\\5090-study-20261023\\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
observed=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip()
assert not observed,('Study executable remains live',observed)
bundle=results/'integrated-batch-top1-source.zip';assert sha(bundle)==p['source_zip_sha256']
expected={'CMakeLists.txt','src/backends/tensorrt/cuda_tensor_io.cpp','src/cuda/logits_sampler_kernel.cu','src/cuda/logits_top1_batch_dispatch.h','test2026/native/logits_top1_batch_policy_test.cpp'}
new_expected={'src/cuda/logits_top1_batch_dispatch.h','test2026/native/logits_top1_batch_policy_test.cpp'}
assert set(p['candidate_source_sha256'])==expected and set(p['allowed_new_source'])==new_expected
old={};candidate={}
with zipfile.ZipFile(bundle) as z:
 assert z.testzip() is None and len(z.namelist())==len(expected) and set(z.namelist())==expected
 for n,h in p['candidate_source_sha256'].items():
  target=repo/n;assert target.resolve().is_relative_to(repo.resolve())
  candidate[n]=z.read(n);assert hashlib.sha256(candidate[n]).hexdigest()==h
  if n in new_expected:assert not target.exists()
  else:assert n in p['base_source_sha256'];old[n]=target.read_bytes()
out.mkdir();record={'status':'installing_unbuilt_sources','preparation_sha256':sha(prep_path),'source_zip_sha256':sha(bundle),
 'parent_proof_sha256':sha(proofpath),'parent_archive_sha256':proof['archive_sha256'],
 'runtime_dependencies_before':dependencies,'base_source_sha256':p['base_source_sha256'],
 'candidate_source_sha256':p['candidate_source_sha256'],'scope':p['scope']}
with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n,b in old.items():z.writestr(n,b)
def save():(out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf8')
save()
try:
 for n,b in candidate.items():target=repo/n;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 assert {n:sha(repo/n) for n in candidate}==p['candidate_source_sha256']
 assert all(sha(repo/n)==h for n,h in p['base_source_sha256'].items() if n not in candidate)
 assert {n:sha(root/'build/bin'/n) for n in dependencies}==dependencies
 record.update(status='sources_installed_native_library_unbuilt',exit_code=0)
except BaseException as error:
 skipped=[]
 for n,b in candidate.items():
  target=repo/n
  if target.exists() and sha(target)!=hashlib.sha256(b).hexdigest():skipped.append(n);continue
  if n in old:target.write_bytes(old[n])
  elif target.exists():assert target.resolve().is_relative_to(repo.resolve());target.unlink()
 record.update(status='source_install_failed_restore_attempted',exit_code=1,error=repr(error),restore_skipped_source_drift=skipped);raise
finally:
 record.update(runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in dependencies},
  final_source_sha256={n:sha(repo/n) if (repo/n).exists() else None for n in candidate})
 save();files=[f for f in out.iterdir() if f.is_file()]+[prep_path,bundle,proofpath]+[m/n for n in p['methods']]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
 manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'parent_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':record['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size}))
