"""Preserving native tokenizer rebuild and actual C API gate; no GPU model."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');REPO=ROOT/'Garnet';M=REPO/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(M));from run_logged import run;from run_bf16_low_batch_native_build import idle
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();NAME='tokenizer-nfc-native-build-attempt1';OUT=ROOT/'results'/NAME;archive=ROOT/'results'/(NAME+'-evidence.zip');prep=M/'evidence/tokenizer-nfc-native-build-preparation.json';p=json.loads(prep.read_bytes());library=ROOT/'build/bin/garnet.dll';retained=ROOT/'build/bin/garnet-native-bf16-low-batch-attempt1.dll';saved=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll'
assert not OUT.exists() and not archive.exists() and not saved.exists();assert sha(retained)==sha(library)==p['runtime']['garnet.dll']
for n,h in {**p['base_files'],**p['methods'],**p['inputs']}.items():assert sha(ROOT/n)==h,('Input/source drift',n)
for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
for n,h in p['environment'].items():assert sha(Path(n))==h
assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
for n,row in p['parents'].items():
 f=ROOT/'results'/(n+'-verified.json');q=json.loads(f.read_bytes());closed=json.loads((ROOT/'results'/n/'workflow.json').read_bytes());assert sha(f)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(n+'-evidence.zip'))==row['archive_sha256'] and closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
old={};candidate={}
with zipfile.ZipFile(ROOT/p['source_zip']) as z:
 assert z.testzip() is None and set(z.namelist())==set(p['candidate_files'])
 for n,h in p['candidate_files'].items():
  f=REPO/n;assert f.resolve().is_relative_to(REPO.resolve()) and f.parent.is_dir();assert (sha(f) if f.is_file() else None)==p['candidate_base'][n];candidate[n]=z.read(n);assert hashlib.sha256(candidate[n]).hexdigest()==h
  if f.exists():old[n]=f.read_bytes()
idle();assert shutil.disk_usage(ROOT).free>max(2<<30,3*library.stat().st_size);OUT.mkdir();os.chdir(REPO);code=1
with zipfile.ZipFile(OUT/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n,b in old.items():z.writestr(n,b)
(OUT/'CMakeCache-before.txt').write_bytes((ROOT/'build/CMakeCache.txt').read_bytes());w={'status':'installing_tokenizer','preparation_sha256':sha(prep),'runtime_before':p['runtime'],'environment_before':p['environment'],'parents':p['parents'],'cuda_objects_before':p['cuda_objects_before'],'candidate_base':p['candidate_base'],'cleanup_finished':False,'scope':p['scope'],'performance_admitted':False};common=[prep]+[ROOT/n for n in {**p['base_files'],**p['methods'],**p['inputs']}]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
def save():(OUT/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save()
try:
 for n,b in candidate.items():(REPO/n).write_bytes(b)
 expected={**p['base_files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}};assert all(sha(ROOT/n)==h for n,h in expected.items());w['installed_source']=expected;save()
 code=run([sys.executable,'-m','unittest']+p['test_modules'],OUT/'cpu-ledger','correctness','49unchanged B1/legacy model/cache construction regressions',common)
 if code:raise RuntimeError('CPU profile regression')
 code=run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],OUT/'build-ledger','build','Preserved native tree/SDK; CPU tokenizer source and headers only',common)
 if code:raise RuntimeError('Native tokenizer build')
 assert sha(library)!=p['runtime']['garnet.dll'];w['candidate_library_sha256']=sha(library);save()
 code=run([sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(OUT/'native-tokenizer-cpu')],OUT/'native-cpu-ledger','correctness','955actual rebuilt native C API encodes against frozen Rust oracles',common+[library])
 if code:raise RuntimeError('Rebuilt native tokenizer CPU mismatch')
 for n,h in expected.items():assert sha(ROOT/n)==h
 for n,h in p['runtime'].items():
  if n!='garnet.dll':assert sha(ROOT/'build/bin'/n)==h
 saved.write_bytes(library.read_bytes());assert sha(saved)==sha(library);w.update(status='tokenizer_native_built_cpu_qualified_model_unqualified',library_sha256=sha(library))
except BaseException as error:
 w.update(status='tokenizer_native_build_failed_restored',error=repr(error));code=code or 1
 if library.exists() and sha(library)!=p['runtime']['garnet.dll']:(OUT/'failed-garnet.dll').write_bytes(library.read_bytes())
 library.write_bytes(retained.read_bytes());skipped=[]
 for n,h in p['candidate_files'].items():
  f=REPO/n
  if not f.exists() or sha(f)!=h:skipped.append(n);continue
  if n in old:f.write_bytes(old[n])
  else:assert f.resolve().is_relative_to(REPO.resolve());f.unlink()
 w['restore_skipped_source_drift']=skipped;raise
finally:
 source_names=set(p['base_files'])|{'Garnet/'+n for n in candidate};w.update(exit_code=code,source_after={n:sha(ROOT/n) if (ROOT/n).is_file() else None for n in source_names},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},methods_after={n:sha(ROOT/n) for n in p['methods']},retained_library_sha256=sha(retained),cuda_objects_after={f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')});save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in OUT.rglob('*') if f.is_file()]+common+[ROOT/n for n in source_names if (ROOT/n).is_file()]+[ROOT/'build/CMakeCache.txt',ROOT/'build/build.ninja',ROOT/'build/CMakeFiles/rules.ninja'];raw={f.relative_to(ROOT).as_posix():f.read_bytes() for f in paths}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'scope':p['scope']},indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}),flush=True)
raise SystemExit(code)
