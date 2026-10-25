"""Standalone B1 cache-update collection after the model worker is closed/replayed.

A replayed failed model parent is preserved as failed; this isolated operator
does not load/replace Garnet or promote the model or any serving profile.
"""
from pathlib import Path
import base64,hashlib,json,os,subprocess,sys,zipfile

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';m=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(m));from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();name='bf16-low-batch-commit-native-attempt1';out=results/name
prep=m/'evidence/bf16-low-batch-commit-preparation.json';p=json.loads(prep.read_bytes())
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
for n,h in p['files'].items():assert sha(root/n)==h,('Source/method/input drift',n)
for n,h in p['inherited_model_sources'].items():assert sha(root/n)==h,('Closed model source drift',n)
for n,h in p['tools'].items():assert sha(Path(n))==h,('Compiler drift',n)
for parent,entry in p['parents'].items():
 raw=(results/(parent+'-verified.json')).read_bytes();proof=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==entry['proof_sha256'] and proof['status']==entry['status']
 assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==entry['archive_sha256']
parent='integrated-batch-top1-model-lifecycle-attempt1';closed=json.loads((results/parent/'workflow.json').read_bytes())
assert closed['cleanup_finished'] and closed['library_unchanged'] and closed['source_sha256']==closed['source_after_sha256']==p['inherited_model_sources']
assert closed['runtime_dependencies']==closed['runtime_dependencies_after']==p['runtime_dependencies']
assert (closed['exit_code']==0)==(p['parents'][parent]['status']=='integrated_top1_full_model_lifecycle_replayed')
for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h
library=root/'build/bin/garnet.dll';digest=sha(library);scratch=root/'tools/bf16-kv-native'
assert scratch.resolve(strict=True).parent==(root/'tools').resolve(strict=True)
assert sha(scratch/'bf16_kv_fixture_layout.h')==p['fixture_header_sha256']
exe=scratch/'bf16-low-batch-commit.exe';policy=scratch/'low-batch-commit-policy.exe';owner=scratch/'low-batch-commit-owner.json'
assert not exe.exists() and not policy.exists() and not owner.exists()
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
 command=r"$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\CantorAI\5090-study-20261023\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
 live=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip();assert not live,('Study executable live',live)
idle();owner.write_text(json.dumps({'study':str(root),'component':'bf16-low-batch-commit','scope':'New standalone compiler outputs only; no Garnet source/runtime installation.'},indent=2),encoding='utf8')
out.mkdir();os.chdir(repo);os.environ['CUDA_VISIBLE_DEVICES']='0';code=1
w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'inherited_model_sources':p['inherited_model_sources'],
 'parents':p['parents'],'tools':p['tools'],'runtime_dependencies':p['runtime_dependencies'],'observed_unused_library_sha256':digest,
 'runtime_loaded':False,'cleanup_finished':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in p['files']]+[root/n for n in p['inherited_model_sources']]+[prep]+[results/(n+'-verified.json') for n in p['parents']]
save()
try:
 code=run([sys.executable,'-m','unittest']+p['test_ids'],out/'cpu-ledger','correctness','Separate B1 integer oracle, byte-stream mutation and unchanged B4/B8 layout controls',common)
 if code:raise RuntimeError('B1 commit CPU controls failed')
 code=run(['cmd.exe','/d','/c',str(m/'compile_bf16_low_batch_commit.bat'),p['vcvars'],p['nvcc'],str(scratch),str(Path(p['host_compiler']).parent)],out/'build-ledger','build','Actual compiler CPU policy/import guard and private sm120 B1 commit collector',common+[Path(n) for n in p['tools']])
 if code:raise RuntimeError('B1 commit build or CPU policy failed')
 assert Path((scratch/'low-batch-commit-host-compiler.txt').read_text().splitlines()[0]).samefile(Path(p['host_compiler']))
 w.update(executable_sha256=sha(exe),policy_executable_sha256=sha(policy));save()
 code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(exe),str(out/'witnesses.bin')],out/'gpu-ledger','correctness','20B1 cases, full host/GPU canonical bytes,3graph profiles and66rejected calls; independent exact-byte replay mandatory',common+[exe])
 if code:raise RuntimeError('B1 commit collection failed')
 w['status']='bf16_low_batch_commit_collected_needs_byte_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},inherited_model_sources_after={n:sha(root/n) for n in p['inherited_model_sources']},
  tools_after={n:sha(Path(n)) for n in p['tools']},runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==digest)
 for key,path in [('executable',exe),('policy_executable',policy)]:
  if path.exists():w[key+'_after_sha256']=sha(path)
 save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in out.rglob('*') if f.is_file()]+common+[owner,scratch/'bf16_kv_fixture_layout.h']
 paths += [f for f in (exe,policy,scratch/'low-batch-commit-host-compiler.txt',scratch/'low-batch-commit-policy-imports.txt') if f.exists()]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths};manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'parents':p['parents'],'scope':p['scope']}
 archive=results/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
