"""Archive CPU TensorRT plugin contracts, after independent operator replay."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prep_path=methods/'evidence/bf16-plugin-cpu-contract-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in {**prep['files'],**prep['sdk_sha256']}.items():assert sha(root/n)==h,('Source/SDK drift',n)
observed=results/'bf16-plugin-sdk-runtime-observation.json'
assert sha(observed)==prep['files']['Garnet/test2026/cuda/qwen38_5090/evidence/bf16-plugin-sdk-runtime-observation.json']
assert json.loads(observed.read_bytes())['sha256']==prep['sdk_sha256']['tools/TensorRT/bin/nvinfer_10.dll']
parent='bf16-kv-overlay-native-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='bf16_kv_overlay_native_fp64_replayed' and len(proof['cases'])==70 and proof['fp64_witness_scalars']==1172480 and proof['maximum_tolerance_fraction']<=1
assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']
assert sha(methods/'verify_bf16_kv_overlay_native.py')==proof['replay_verifier_sha256'] and sha(methods/'bf16_kv_overlay_oracle.py')==proof['oracle_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
library=root/'build/bin/garnet.dll';digest=sha(library);assert digest==previous['library_sha256']
name='bf16-plugin-cpu-contract-attempt1';out=results/name;assert not out.exists();out.mkdir()
archive=results/(name+'-evidence.zip');assert not archive.exists()
os.environ['PATH']=str(root/'tools/TensorRT/bin')+';'+str(root/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+os.environ['PATH']
os.chdir(root/'tools/bf16-kv-native')
w={'status':'running','source_sha256':prep['files'],'sdk_sha256':prep['sdk_sha256'],'preparation_sha256':sha(prep_path),
   'parent_overlay_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':digest,'scope':prep['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();common=[root/n for n in prep['files']]+[prep_path,proof_path];code=1
try:
    code=run([str(root/'tools/bf16-kv-native/probe_bf16_kv_plugin_contract.exe')],out/'cpu','correctness','TensorRT BF16 plugin clone/serialization/format/shape contracts',common+[root/n for n in prep['sdk_sha256']])
    if code:raise RuntimeError('CPU plugin contracts failed')
    w['status']='bf16_plugin_cpu_contract_finished_needs_independent_replay'
except BaseException as error:
    w.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    w.update(exit_code=code,source_after_sha256={n:sha(root/n) for n in prep['files']},sdk_after_sha256={n:sha(root/n) for n in prep['sdk_sha256']},library_unchanged=sha(library)==digest);save()
    files=[p for p in out.rglob('*') if p.is_file()]+common
    raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},'scope':prep['scope']}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in raw.items():z.writestr(n,v)
        z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'status':w['status'],'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
