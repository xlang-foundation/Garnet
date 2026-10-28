"""Preserve, install and build the default-off full-cache reference; no GPU run."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-float-reference-model-build-attempt1'
prep_path=methods/'evidence/bf16-float-reference-model-build-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in p['methods'].items():assert sha((methods/n).read_bytes())==h,('Method drift',n)
for n,h in p['base_source_sha256'].items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
for parent,wanted in [('bf16-kv-float-reference-native-attempt1','bf16_float_reference_native_fp64_replayed'),('bf16-float-reference-plugin-cpu-attempt1','bf16_float_reference_plugin_cpu_replayed')]:
 proof=json.loads((results/(parent+'-verified.json')).read_bytes());assert proof['status']==wanted
 assert sha((results/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==p['parent_archives'][parent+'-evidence.zip']
previous=json.loads((results/'bf16-kv-float-reference-native-attempt1/workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
library=root/'build/bin/garnet.dll';retained=root/'build/bin/garnet-native-bf16-model-attempt1.dll'
assert sha(library.read_bytes())==sha(retained.read_bytes())==p['original_library_sha256']
bundle=results/'bf16-float-reference-model-source.zip';assert sha(bundle.read_bytes())==p['source_zip_sha256']
new={};old={}
with zipfile.ZipFile(bundle) as z:
 assert z.testzip() is None and set(z.namelist())==set(p['candidate_source_sha256'])
 for n,h in p['candidate_source_sha256'].items():
  target=repo/n;assert target.resolve().is_relative_to(repo.resolve());new[n]=z.read(n);assert sha(new[n])==h
  if target.exists():assert n in p['base_source_sha256'];old[n]=target.read_bytes()
  else:assert n in p['allowed_new_source']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
for token in (b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF'):assert token in (root/'build/CMakeCache.txt').read_bytes()
out=results/name;assert not out.exists();out.mkdir();archive=results/(name+'-evidence.zip');assert not archive.exists()
dependencies={f.name:sha(f.read_bytes()) for f in (root/'build/bin').iterdir() if f.name in ('xlang3.exe','xlang3_runtime.dll','nvinfer_10.dll','cudart64_13.dll','cublasLt64_13.dll')}
assert len(dependencies)==5
w={'status':'installing_unbuilt_reference','preparation_sha256':sha(prep_path.read_bytes()),'original_library_sha256':p['original_library_sha256'],
   'source_zip_sha256':p['source_zip_sha256'],'candidate_source_sha256':p['candidate_source_sha256'],
   'original_source_sha256':{n:sha(v) for n,v in old.items()},'runtime_dependencies_before_sha256':dependencies,'scope':p['scope']}
with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n,v in old.items():z.writestr(n,v)
def save():(out/'manifest.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();code=1;os.chdir(repo)
try:
 for n,v in new.items():(repo/n).parent.mkdir(parents=True,exist_ok=True);(repo/n).write_bytes(v)
 code=run([sys.executable,'-m','unittest']+p['cpu_tests'],out/'cpu-ledger','correctness','Default-off and full FP32 BF16-value reference contracts',[prep_path]+[repo/n for n in new])
 if code:raise RuntimeError('Reference CPU contracts failed')
 code=run(['cmd.exe','/d','/c',str(methods/'build_bf16_model_integration.bat')],out/'build-ledger','build','Full Garnet reference plugin/lowering build',[library,prep_path,Path(__file__)]+[repo/n for n in new])
 if code:raise RuntimeError('Reference native build failed')
 assert {n:sha((repo/n).read_bytes()) for n in new}==p['candidate_source_sha256']
 after={n:sha((root/'build/bin'/n).read_bytes()) for n in dependencies};assert after==dependencies
 assert sha(library.read_bytes())!=p['original_library_sha256']
 saved=root/'build/bin/garnet-native-bf16-float-reference-attempt1.dll';assert not saved.exists();saved.write_bytes(library.read_bytes())
 w.update(status='bf16_float_reference_native_build_finished_unqualified',exit_code=0,library_sha256=sha(library.read_bytes()),retained_candidate_library=str(saved),runtime_dependencies_after_sha256=after)
except BaseException as error:
 w.update(status='bf16_float_reference_native_build_failed_restored',exit_code=code or 1,error=repr(error))
 if library.exists() and sha(library.read_bytes())!=p['original_library_sha256']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
 library.write_bytes(retained.read_bytes());assert sha(library.read_bytes())==p['original_library_sha256']
 skipped=[]
 for n,v in new.items():
  target=repo/n
  if target.exists() and sha(target.read_bytes())!=sha(v):skipped.append(n);continue
  if n in old:target.write_bytes(old[n])
  elif target.exists():assert target.resolve().is_relative_to(repo.resolve());target.unlink()
 w['restore_skipped_source_drift']=skipped;raise
finally:
 save();files=[f for f in out.rglob('*') if f.is_file()]+[prep_path,bundle]+[methods/n for n in p['methods']]+[repo/n for n in new if (repo/n).exists()]+[root/'build/CMakeCache.txt',root/'build/build.ninja',root/'build/CMakeFiles/rules.ninja']
 files += [results/(n.removesuffix('-evidence.zip')+'-verified.json') for n in p['parent_archives']]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
 m={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':p['parent_archives'],'scope':p['scope']}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,v in raw.items():z.writestr(n,v)
  z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
