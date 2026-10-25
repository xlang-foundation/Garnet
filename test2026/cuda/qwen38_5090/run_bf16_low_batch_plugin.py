"""One isolated CPU/forwarding collector after actual closed operator/model proofs."""
from pathlib import Path
import base64,hashlib,json,os,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';m=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(m));from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();name='bf16-low-batch-plugin-native-attempt1';out=results/name
prep=m/'evidence/bf16-low-batch-plugin-preparation.json';p=json.loads(prep.read_bytes());scratch=root/'tools/bf16-kv-native'
assert not out.exists() and not (results/(name+'-evidence.zip')).exists()
for field in ('files','inherited_model_sources','sdk_sha256'):
 for n,h in p[field].items():assert sha(root/n)==h,('Source/SDK drift',n)
for n,h in p['tools'].items():assert sha(Path(n))==h,('Tool drift',n)
for n,h in p['runtime_dependencies'].items():assert sha(root/'build/bin'/n)==h,('Runtime drift',n)
for parent,entry in p['parents'].items():
 raw=(results/(parent+'-verified.json')).read_bytes();proof=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==entry['proof_sha256'] and proof['status']==entry['status']
 assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']==entry['archive_sha256']
for parent in ('integrated-batch-top1-model-lifecycle-attempt1','bf16-low-batch-commit-native-attempt1'):
 closed=json.loads((results/parent/'workflow.json').read_bytes());assert closed['cleanup_finished'] and closed['library_unchanged']
assert scratch.resolve(strict=True).parent==(root/'tools').resolve(strict=True)
assert sha(scratch/'bf16_kv_fixture_layout.h')==p['fixture_header_sha256']
outputs=['bf16-low-batch-plugin-cpu.exe','bf16-low-batch-plugin-forwarding.exe','low-batch-plugin-owner.json','low-batch-plugin-host-compiler.txt','low-batch-plugin-cpu-imports.txt','low-batch-plugin-forwarding-imports.txt']
outputs += ['low-batch-plugin-'+n+'.obj' for n in ('wrapper','legacy-wrapper','cpu-operators','cpu-probe','kernel','gpu-probe')]
assert all(not (scratch/n).exists() for n in outputs),'Preserve existing fixed outputs'
def idle():
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
 command=r"$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\CantorAI\5090-study-20261023\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
 live=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip();assert not live,('Study executable live',live)
idle();owner=scratch/'low-batch-plugin-owner.json';owner.write_text(json.dumps({'study':str(root),'component':'bf16-low-batch-plugin','scope':'Isolated compiler outputs; no Garnet installation.'},indent=2),encoding='utf8')
library=root/'build/bin/garnet.dll';digest=sha(library);out.mkdir();os.chdir(repo);code=1
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PATH']=str(root/'tools/TensorRT/bin')+';'+os.environ['PATH']
w={'status':'running','preparation_sha256':sha(prep),'source_sha256':p['files'],'inherited_model_sources':p['inherited_model_sources'],'sdk_sha256':p['sdk_sha256'],
 'tools':p['tools'],'runtime_dependencies':p['runtime_dependencies'],'parents':p['parents'],'observed_unused_library_sha256':digest,'garnet_runtime_loaded':False,'cleanup_finished':False,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in dict.fromkeys(list(p['files'])+list(p['inherited_model_sources']))]+[prep]+[results/(n+'-verified.json') for n in p['parents']]
save()
try:
 code=run([sys.executable,str(m/'test_bf16_low_batch_plugin_attention_replay.py'),str(results/'bf16-low-batch-attention-native-attempt1-evidence.zip')],out/'checker-ledger','correctness','Six synthetic wrapper-forwarding mutation controls over actual qualified parent fixture',common)
 if code:raise RuntimeError('Wrapper checker controls failed')
 code=run(['cmd.exe','/d','/c',str(m/'compile_bf16_low_batch_plugin.bat'),p['vcvars'],p['nvcc'],str(scratch),str(Path(p['host_compiler']).parent),str(root)],out/'build-ledger','build','Pinned actual compiler: isolated B1 CPU contract and sm120 forwarding collector',common+[Path(n) for n in p['tools']]+[root/n for n in p['sdk_sha256']])
 if code:raise RuntimeError('Wrapper collector build failed')
 assert Path((scratch/'low-batch-plugin-host-compiler.txt').read_text().splitlines()[0]).samefile(Path(p['host_compiler']))
 w['executable_sha256']={n:sha(scratch/n) for n in outputs if n.endswith('.exe')};save();idle()
 code=run([str(scratch/'bf16-low-batch-plugin-cpu.exe')],out/'contract-ledger','correctness','32B1 CPU profiles, distinct registry/wire/clone/shape/format and B4/B8 cross-admission guards; no enqueue',common+[scratch/'bf16-low-batch-plugin-cpu.exe']+[root/n for n in p['sdk_sha256']])
 if code:raise RuntimeError('CPU registry/wire contract failed')
 idle()
 code=run([sys.executable,str(m/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(scratch/'bf16-low-batch-plugin-forwarding.exe'),str(out/'witnesses.f32')],out/'gpu-ledger','correctness','44case exact-parent wrapper forwarding,87host enqueues,3graphs,40cache equivalence and104retained direct-kernel negatives',common+[scratch/'bf16-low-batch-plugin-forwarding.exe'])
 if code:raise RuntimeError('GPU wrapper forwarding collection failed')
 w['status']='bf16_low_batch_plugin_collected_needs_exact_parent_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},inherited_model_sources_after={n:sha(root/n) for n in p['inherited_model_sources']},sdk_after_sha256={n:sha(root/n) for n in p['sdk_sha256']},tools_after={n:sha(Path(n)) for n in p['tools']},runtime_dependencies_after={n:sha(root/'build/bin'/n) for n in p['runtime_dependencies']},library_unchanged=sha(library)==digest)
 w['executables_after_sha256']={n:sha(scratch/n) for n in outputs if n.endswith('.exe') and (scratch/n).exists()};save();idle();w['cleanup_finished']=True;save()
 paths=[f for f in out.rglob('*') if f.is_file()]+common+[owner,scratch/'bf16_kv_fixture_layout.h']+[scratch/n for n in outputs if not n.endswith('.obj') and (scratch/n).exists()]
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in paths};manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'parents':p['parents'],'scope':p['scope']}
 with zipfile.ZipFile(results/(name+'-evidence.zip'),'x',zipfile.ZIP_DEFLATED) as z:
  for n,b in raw.items():z.writestr(n,b)
  z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
 archive=results/(name+'-evidence.zip');print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
