"""Collect untimed BF16 attention witnesses after the closed commit-operator gate."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
methods=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prep_path=methods/'evidence/bf16-kv-overlay-native-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['files'].items():assert sha(root/n)==h,('Source drift',n)
parent='bf16-kv-commit-native-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='bf16_kv_commit_native_operator_replayed' and proof['full_state_byte_cases']==34 and proof['rejected_calls']==62
parent_archive=results/(parent+'-evidence.zip');assert sha(parent_archive)==proof['archive_sha256']
assert sha(methods/'verify_bf16_kv_commit_native.py')==proof['replay_verifier_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
library=root/'build/bin/garnet.dll';digest=sha(library);assert digest==previous['library_sha256']==prep['observed_unused_library_sha256']
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();name='bf16-kv-overlay-native-attempt1';out=results/name;assert not out.exists();out.mkdir()
target=results/(name+'-evidence.zip');assert not target.exists()
w={'status':'running','source_sha256':prep['files'],'preparation_sha256':sha(prep_path),
   'parent_commit_archive_sha256':proof['archive_sha256'],'library_sha256':digest,'cleanup_finished':False,'scope':prep['scope']};code=1
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
common=[root/n for n in prep['files']]+[prep_path,proof_path,library];save()
try:
    code=run([sys.executable,str(methods/'test_bf16_kv_overlay_oracle.py')],out/'cpu-oracle','correctness','Independent FP64 oracle analytical controls',common)
    if code:raise RuntimeError('FP64 oracle CPU controls failed')
    code=run([sys.executable,str(methods/'prepare_bf16_kv_operator_fixture.py'),str(out/'regenerated-layout.h')],out/'cpu-fixture','correctness','Independent Python layout fixture',common)
    if code:raise RuntimeError('CPU fixture regeneration failed')
    assert sha(out/'regenerated-layout.h')==prep['fixture_header_sha256']
    executable=root/'tools/bf16-kv-native/probe_bf16_kv_overlay_native.exe'
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(executable),str(out/'witnesses.bin')],
             out/'native','correctness','BF16 dense attention witnesses and readonly/cache/graph controls',common)
    if code:raise RuntimeError('BF16 attention native control failed')
    resource=json.loads((out/'resources/resource-monitor.json').read_bytes());assert resource['gpu']['uuid']==prep['gpu_uuid']
    w['status']='bf16_overlay_collected_needs_independent_fp64_replay'
except BaseException as error:
    w.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    w['exit_code']=code;w['library_unchanged']=sha(library)==digest
    w['source_after_sha256']={n:sha(root/n) for n in prep['files']};save()
    idle();w['cleanup_finished']=True;save()
    files=[p for p in out.rglob('*') if p.is_file()]+[p for p in common if p!=library]
    raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},'scope':prep['scope'],
              'parent_commit_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':digest}
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in raw.items():z.writestr(n,v)
        z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'status':w['status'],'archive_sha256':sha(target),'bytes':target.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
