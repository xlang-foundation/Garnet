"""Preserve closed native build/default-off regression, including failure."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='native-kv-delta-default-off-attempt1';out=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
w=json.loads((out/'workflow.json').read_bytes());assert w['status'] in ('failed','default_off_finished_needs_independent_replay') and w['cleanup_finished']
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
build=results/'native-kv-delta-build-attempt1';b=json.loads((build/'manifest.json').read_bytes())
assert b['status']=='native_kv_delta_build_finished_unqualified' and b['exit_code']==0 and b['library_sha256']==w['library_sha256']
files=[p for folder in (out,build) for p in folder.rglob('*') if p.is_file()]
files += [root/'Garnet'/n for n in w['source_sha256']]
for n,h in w['source_sha256'].items():assert sha((root/'Garnet'/n).read_bytes())==h
files += [Path(b['retained_candidate_library']),results/'native-kv-delta-source.zip',root/'test-b4-host-joined-mtp.bat']
files += [methods/n for n in ('run_native_kv_delta_build.py','run_native_kv_delta_default_off.py','archive_native_kv_delta_default_off.py','verify_native_kv_delta_default_off.py','probe_b4_host_joined_mtp.py','native_worker_io.py','run_logged.py','monitor_native_probe.py','sample_vram.py','evidence/native-kv-delta-source-handoff.json')]
files += [results/'unicode-transport-repair2-verified.json',results/'study-ssd-engine-cache/receipt.json']
links=['shared-mtp3-full-b4-t4-host-joined-attempt1-evidence.zip','canonical-kv-overlay-plugin-attempt1-evidence.zip']
files += [results/n.replace('-evidence.zip','-verified.json') for n in links]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'library_sha256':w['library_sha256'],'linked_archives':{n:sha((results/n).read_bytes()) for n in links},'scope':w['scope']}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
