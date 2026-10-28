"""Install the pinned opt-in B1 route and build against the retained native SDK.

No Garnet loading or GPU model execution. Preserve the last qualified library
and every overwritten source byte before changing the existing study checkout.
"""
from pathlib import Path
import base64,hashlib,json,os,shutil,subprocess,sys,zipfile

ROOT=Path('D:/CantorAI/5090-study-20261023');REPO=ROOT/'Garnet';M=REPO/'test2026/cuda/qwen38_5090'
NAME='bf16-low-batch-native-build-attempt1';OUT=ROOT/'results'/NAME
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
 ps=r"$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\CantorAI\5090-study-20261023\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
 live=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],text=True).strip()
 assert not live,('Study process remains live',live)

def main():
 sys.path.insert(0,str(M));from run_logged import run
 prep=M/'evidence/bf16-low-batch-native-build-preparation.json';p=json.loads(prep.read_bytes())
 archive=ROOT/'results'/(NAME+'-evidence.zip');assert not OUT.exists() and not archive.exists()
 for n,h in p['base_files'].items():assert sha(ROOT/n)==h,('Base source drift',n)
 for n,h in p['methods'].items():assert sha(ROOT/n)==h,('Method drift',n)
 for n,h in p['environment_files'].items():assert sha(Path(n))==h,('SDK/build environment drift',n)
 for n,h in p['runtime_dependencies'].items():assert sha(ROOT/'build/bin'/n)==h,('Runtime drift',n)
 for n,row in p['parents'].items():
  proof=ROOT/'results'/(n+'-verified.json');q=json.loads(proof.read_bytes());closed=json.loads((ROOT/'results'/n/'workflow.json').read_bytes())
  assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==sha(ROOT/'results'/(n+'-evidence.zip'))==row['archive_sha256']
  assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
 library=ROOT/'build/bin/garnet.dll';retained=ROOT/'build/bin/garnet-native-integrated-batch-top1-attempt1.dll'
 saved=ROOT/'build/bin/garnet-native-bf16-low-batch-attempt1.dll'
 assert sha(library)==sha(retained)==p['runtime_dependencies']['garnet.dll'] and not saved.exists()
 cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cmake_cache_before_sha256']
 for flag in (b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF',b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF'):assert flag in cache.read_bytes()
 bundle=ROOT/p['source_zip'];assert sha(bundle)==p['source_zip_sha256']
 old={};candidate={}
 with zipfile.ZipFile(bundle) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(p['candidate_files']) and set(z.namelist())==set(p['candidate_files'])
  for n,h in p['candidate_files'].items():
   target=REPO/n;assert target.resolve().is_relative_to(REPO.resolve()) and target.parent.is_dir()
   candidate[n]=z.read(n);assert hashlib.sha256(candidate[n]).hexdigest()==h
   if p['candidate_base'][n] is None:assert not target.exists(),('Preserve unrecorded target',n)
   else:assert sha(target)==p['candidate_base'][n];old[n]=target.read_bytes()
 idle();assert shutil.disk_usage(ROOT).free>max(2<<30,3*library.stat().st_size)
 OUT.mkdir();os.chdir(REPO);code=1
 with zipfile.ZipFile(OUT/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n,b in old.items():z.writestr(n,b)
 (OUT/'CMakeCache-before.txt').write_bytes(cache.read_bytes())
 w={'status':'installing_b1_optin_sources','preparation_sha256':sha(prep),'source_zip_sha256':sha(bundle),'source_before':p['base_files'],'candidate_files':p['candidate_files'],'candidate_base':p['candidate_base'],'methods':p['methods'],'environment_before':p['environment_files'],'runtime_before':p['runtime_dependencies'],'parents':p['parents'],'runtime_loaded':False,'cleanup_finished':False,'scope':p['scope']}
 def save():(OUT/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
 common=[ROOT/n for n in p['methods']]+[prep,bundle]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
 save()
 try:
  for n,b in candidate.items():(REPO/n).write_bytes(b)
  expected=dict(p['base_files']);expected.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()})
  assert all(sha(ROOT/n)==h for n,h in expected.items());w['installed_source']=expected;save()
  code=run([sys.executable,'-m','unittest']+p['test_modules'],OUT/'cpu-ledger','correctness','Actual remote B1 construction/profile and unchanged B4/B8/cache transaction regressions',common+[ROOT/n for n in expected])
  if code:raise RuntimeError('B1 native CPU contracts failed')
  code=run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],OUT/'build-ledger','build','Reused native Windows build tree, preserved SDK/MT/sm120 and default-off B1 extension',common+[ROOT/n for n in expected]+[Path(n) for n in p['environment_files']])
  if code:raise RuntimeError('B1 native DLL build failed')
  assert sha(library)!=p['runtime_dependencies']['garnet.dll']
  for n,h in expected.items():assert sha(ROOT/n)==h
  for n,h in p['runtime_dependencies'].items():
   if n!='garnet.dll':assert sha(ROOT/'build/bin'/n)==h
  assert sha(retained)==p['runtime_dependencies']['garnet.dll']
  saved.write_bytes(library.read_bytes());assert sha(saved)==sha(library)
  w.update(status='bf16_low_batch_native_built_model_unqualified',library_sha256=sha(library),saved_library=str(saved),installed_source=expected)
 except BaseException as error:
  w.update(status='bf16_low_batch_native_build_failed_restored',error=repr(error));code=code or 1
  if library.exists() and sha(library)!=p['runtime_dependencies']['garnet.dll']:(OUT/'failed-garnet.dll').write_bytes(library.read_bytes())
  library.write_bytes(retained.read_bytes());assert sha(library)==p['runtime_dependencies']['garnet.dll']
  skipped=[]
  for n,b in candidate.items():
   target=REPO/n
   if not target.exists() or sha(target)!=p['candidate_files'][n]:skipped.append(n);continue
   if n in old:target.write_bytes(old[n])
   else:assert target.resolve().is_relative_to(REPO.resolve());target.unlink()
  w['restore_skipped_source_drift']=skipped;raise
 finally:
  w.update(exit_code=code,source_after={n:sha(ROOT/n) if (ROOT/n).is_file() else None for n in set(p['base_files'])|{'Garnet/'+n for n in candidate}},methods_after={n:sha(ROOT/n) for n in p['methods']},environment_after={n:sha(Path(n)) for n in p['environment_files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime_dependencies']},retained_library_sha256=sha(retained))
  save();idle();w['cleanup_finished']=True;save()
  paths=[f for f in OUT.rglob('*') if f.is_file()]+common+[ROOT/n for n in w['source_after'] if (ROOT/n).is_file()]+[cache,ROOT/'build/build.ninja',ROOT/'build/CMakeFiles/rules.ninja']
  raw={f.relative_to(ROOT).as_posix():f.read_bytes() for f in paths};manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'scope':p['scope']}
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for n,b in raw.items():z.writestr(n,b)
   z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
  print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'files':len(raw),'bytes':archive.stat().st_size}),flush=True)
 return code

if __name__=='__main__':raise SystemExit(main())
