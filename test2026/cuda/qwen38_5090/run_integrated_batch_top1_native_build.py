"""Build installed sampler sources on the actual SDK, without loading Garnet."""
from pathlib import Path
import base64,hashlib,json,os,shutil,subprocess,sys,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
m=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(m));from run_logged import run
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
name='integrated-batch-top1-native-build-attempt1';out=results/name
prep=m/'evidence/integrated-batch-top1-native-build-preparation.json';p=json.loads(prep.read_bytes())
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
for n,h in p['files'].items():assert sha(root/n)==h,('Source/method/input drift',n)
for n,h in p['environment_files'].items():assert sha(Path(n))==h,('Build environment drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h,('Runtime drift',n)
parent='integrated-batch-top1-correctness-attempt1';proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert sha(proofpath)==p['component_proof_sha256'] and proof['status']=='integrated_batch_top1_correctness_replayed'
assert len(proof['variants'])==4 and not proof['performance_admitted'] and not proof['native_runtime_built']
assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==p['component_archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
assert closed['source_sha256']==closed['source_after_sha256']==p['component_sources']
library=root/'build/bin/garnet.dll';retained=root/'build/bin/garnet-native-bf16-float-reference-attempt1.dll'
assert sha(library)==sha(retained)==p['runtime_dependencies']['garnet.dll']
saved=root/'build/bin/garnet-native-integrated-batch-top1-attempt1.dll';assert not saved.exists()
policy=root/'build/bin/garnet_batch_top1_policy_test.exe';assert not policy.exists()
def idle():
 used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<1024
 command=r"$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\CantorAI\5090-study-20261023\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
 live=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip();assert not live,('Study executable live',live)
idle();assert shutil.disk_usage(root).free>2*library.stat().st_size
cache=root/'build/CMakeCache.txt'
assert sha(cache)==p['cmake_cache_before_sha256']
for token in (b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF',b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF'):assert token in cache.read_bytes()
out.mkdir();os.chdir(repo);code=1
w={'status':'building_native_sampler_unqualified','preparation_sha256':sha(prep),'source_sha256':p['files'],
 'environment_files':p['environment_files'],'runtime_dependencies_before':p['runtime_dependencies'],
 'component_proof_sha256':sha(proofpath),'component_archive_sha256':proof['archive_sha256'],
 'runtime_loaded':False,'scope':p['scope']}
common=[root/n for n in p['files']]+[prep,proofpath]
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();(out/'CMakeCache-before.txt').write_bytes(cache.read_bytes())
try:
 code=run([sys.executable,'-m','unittest']+p['test_ids'],out/'cpu-ledger','correctness','Explicit inherited model/profile and sampler fixture test inventory',common)
 if code:raise RuntimeError('Native build CPU contracts failed')
 code=run(['cmd.exe','/d','/c',str(m/'build_integrated_batch_top1_native.bat'),p['vcvars']],out/'build-ledger','build','Actual preserved Windows SDK native DLL and CPU policy target',common+[library]+[Path(n) for n in p['environment_files']])
 if code:raise RuntimeError('Native sampler build/link failed')
 assert policy.is_file() and sha(library)!=p['runtime_dependencies']['garnet.dll']
 assert {n:sha(root/n) for n in p['files']}==p['files']
 assert all(sha(root/'build/bin'/n)==h for n,h in p['runtime_dependencies'].items() if n!='garnet.dll')
 assert sha(retained)==p['runtime_dependencies']['garnet.dll']
 # Import checking precedes executing this CPU-only target. CUDA is not loaded.
 buildlog=(out/'build-ledger/commands.log').read_bytes().lower();assert b'dump of file' in buildlog
 importlog=buildlog.split(b'dump of file')[-1]
 assert not any(n in importlog for n in (b'cudart',b'nvcuda',b'cublas',b'nvinfer',b'garnet.dll'))
 code=run([str(policy)],out/'policy-ledger','correctness','Native build CPU dispatch policy, nine settings and112shape boundaries',common+[policy])
 if code:raise RuntimeError('Built CPU policy failed')
 saved.write_bytes(library.read_bytes());assert sha(saved)==sha(library)
 w.update(status='integrated_batch_top1_native_built_model_unqualified',library_sha256=sha(library),retained_candidate_library=str(saved),policy_executable_sha256=sha(policy))
except BaseException as error:
 w.update(status='integrated_batch_top1_native_build_failed_library_restored',error=repr(error));code=code or 1
 if library.exists() and sha(library)!=p['runtime_dependencies']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
 library.write_bytes(retained.read_bytes());assert sha(library)==p['runtime_dependencies']['garnet.dll']
 raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},environment_files_after={n:sha(Path(n)) for n in p['environment_files']},
  runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},retained_original_library_sha256=sha(retained))
 if policy.exists():w['policy_executable_after_sha256']=sha(policy)
 save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in out.rglob('*') if f.is_file()]+common+[cache,root/'build/build.ninja',root/'build/CMakeFiles/rules.ninja']
 paths += [Path(n) for n in p['environment_files'] if Path(n).suffix.lower() in ('.h','.lib') and Path(n).is_relative_to(root)]
 if policy.exists():paths.append(policy)
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths}
 manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'component_archive_sha256':proof['archive_sha256'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
