"""Preserve terminal compact-model success/failure without changing source."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='kv-delta-full-b4-t4-attempt1';out=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
w=json.loads((out/'workflow.json').read_bytes());assert w['status'] in ('failed','compact_model_finished_needs_independent_replay') and w['cleanup_finished']
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
files=[p for p in out.rglob('*') if p.is_file()]
for group in ('candidate_source_sha256','unchanged_native_source_sha256','unchanged_parent_source_sha256'):
    for n,h in w[group].items():
        p=root/'Garnet'/n;assert sha(p.read_bytes())==h;files.append(p)
files += [results/'kv-delta-model-source.zip',root/'test-kv-delta-full-b4-t4.bat']
files += [methods/n for n in ('run_kv_delta_full_b4_t4.py','archive_kv_delta_full_b4_t4.py','verify_kv_delta_full_b4_t4.py','run_logged.py','monitor_native_probe.py','sample_vram.py','evidence/kv-delta-model-source-handoff.json')]
parent='native-kv-delta-default-off-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='native_kv_delta_default_off_replayed' and proof['archive_sha256']==w['parent_archive_sha256']
assert sha((results/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256'];files.append(proof_path)
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'library_sha256':w['library_sha256'],'linked_archives':{parent+'-evidence.zip':proof['archive_sha256']},'scope':w['scope']}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
