"""CPU plugin contract check after closed/replayed independent reference GPU test."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prep_path=methods/'evidence/bf16-float-reference-plugin-cpu-preparation.json';p=json.loads(prep_path.read_bytes())
for n,h in {**p['files'],**p['sdk_sha256']}.items():assert sha(root/n)==h,('Source/SDK drift',n)
parent='bf16-kv-float-reference-native-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='bf16_float_reference_native_fp64_replayed' and len(proof['cases'])==42 and proof['fp64_witness_scalars']==442880 and proof['maximum_tolerance_fraction']<=1
assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']
assert sha(methods/'verify_bf16_kv_float_reference_native.py')==proof['replay_verifier_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
library=root/'build/bin/garnet.dll';digest=sha(library);assert digest==previous['library_sha256']==p['expected_unused_library_sha256']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
name='bf16-float-reference-plugin-cpu-attempt1';out=results/name;assert not out.exists();out.mkdir();archive=results/(name+'-evidence.zip');assert not archive.exists()
os.environ['PATH']=str(root/'tools/TensorRT/bin')+';'+str(root/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+os.environ['PATH'];os.chdir(root/'tools/bf16-kv-native')
w={'status':'running','source_sha256':p['files'],'sdk_sha256':p['sdk_sha256'],'preparation_sha256':sha(prep_path),'parent_archive_sha256':proof['archive_sha256'],
   'library_sha256':digest,'scope':p['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();common=[root/n for n in p['files']]+[prep_path,proof_path];code=1
try:
 code=run([str(root/'tools/bf16-kv-native/probe_bf16_float_reference_plugin_cpu.exe')],out/'cpu','correctness','Reference plugin /MT CPU wire/shape/format/negative-enqueue contracts',common+[root/n for n in p['sdk_sha256']])
 if code:raise RuntimeError('Reference plugin CPU contract failed')
 w['status']='bf16_float_reference_plugin_cpu_finished_needs_replay'
except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
 w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in p['files']},sdk_after_sha256={n:sha(root/n) for n in p['sdk_sha256']},library_unchanged=sha(library)==digest);save()
 raw={f.relative_to(root).as_posix():f.read_bytes() for f in [q for q in out.rglob('*') if q.is_file()]+common}
 m={'files':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},'scope':p['scope']}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,v in raw.items():z.writestr(n,v)
  z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
 print(json.dumps({'status':w['status'],'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
