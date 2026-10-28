"""Standalone finite-value BF16 commit operator gate, after closed HTTP replay.

No DLL build/replacement, model loading or serving timing. Preserve failed runs.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
methods=repo/'test2026/cuda/qwen38_5090';sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prep_path=methods/'evidence/bf16-kv-commit-native-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['files'].items():assert sha(root/n)==h,('Source drift',n)
parent='register128-padded-http-pilot-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='register128_padded_http_pilot_replayed' and sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']
assert sha(methods/'verify_register128_padded_http.py')==proof['replay_verifier_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
events=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_bytes().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library);assert digest==proof['library_sha256']
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();name='bf16-kv-commit-native-attempt1';out=results/name;assert not out.exists();out.mkdir()
target=results/(name+'-evidence.zip');assert not target.exists()
w={'status':'running','source_sha256':prep['files'],'preparation_sha256':sha(prep_path),
   'parent_http_archive_sha256':proof['archive_sha256'],'library_sha256':digest,'cleanup_finished':False,
   'scope':prep['scope']};code=1
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save()
try:
    common=[root/n for n in prep['files']]+[prep_path,proof_path,library]
    code=run([sys.executable,str(methods/'test_kv_bf16_study_layout.py')],out/'cpu-layout','correctness','BF16 layout CPU boundaries',common)
    if code:raise RuntimeError('BF16 layout CPU test failed')
    code=run([sys.executable,str(methods/'prepare_bf16_kv_operator_fixture.py'),str(out/'regenerated-layout.h')],out/'cpu-fixture','correctness','Independent Python layout fixture',common)
    if code:raise RuntimeError('BF16 CPU fixture regeneration failed')
    assert sha(out/'regenerated-layout.h')==prep['fixture_header_sha256']
    executable=root/'tools/bf16-kv-native/probe_bf16_kv_commit_native.exe'
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(executable)],
             out/'native','correctness','BF16 commit native all-byte controls',common)
    if code:raise RuntimeError('BF16 native commit operator failed')
    resource=json.loads((out/'resources/resource-monitor.json').read_bytes())
    assert resource['gpu']['uuid']==prep['gpu_uuid']
    w['status']='bf16_commit_operator_finished_needs_independent_replay'
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
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},
              'scope':prep['scope'],'parent_http_archive_sha256':proof['archive_sha256'],'observed_unused_library_sha256':digest}
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in raw.items():z.writestr(n,v)
        z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'status':w['status'],'archive_sha256':sha(target),'bytes':target.stat().st_size,'files':len(raw)}))
raise SystemExit(code)
