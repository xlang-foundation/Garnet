"""Preserve terminal private-padding streaming evidence and state/teacher lineage."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='padded-mtp-stream-attempt1';out=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
w=json.loads((out/'workflow.json').read_bytes());assert w['status'] in ('failed','padded_stream_finished_needs_independent_replay') and w['cleanup_finished']
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
files=[p for p in out.rglob('*') if p.is_file()]
for n,h in w['source_sha256'].items():
    p=root/'Garnet'/n;assert sha(p.read_bytes())==h;files.append(p)
for n,h in w['stream_source_sha256'].items():assert sha((methods/n).read_bytes())==h;files.append(methods/n)
files += [methods/n for n in ('run_padded_mtp_stream.py','archive_padded_mtp_stream.py','verify_padded_mtp_stream.py','verify_padded_mtp_state.py','native_worker_io.py','run_logged.py','monitor_native_probe.py','sample_vram.py','evidence/padded-mtp-stream-preparation.json')]
files += [root/n for n in ('test-padded-mtp-stream.bat','serve-native-kv-delta-padded-mtp-worker.bat')]
files += [results/'unicode-transport-repair2-verified.json',results/'padded-mtp-state-attempt1/workflow.json',root/'models/Qwen3.8-27B-NVFP4/config.json']
links=['padded-mtp-state-attempt1-evidence.zip','kv-delta-stream-attempt1-evidence.zip']
for n,h in zip(links,(w['padded_state_archive_sha256'],w['retained_stream_archive_sha256'])):assert sha((results/n).read_bytes())==h
files += [results/n.replace('-evidence.zip','-verified.json') for n in links]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'library_sha256':w['library_sha256'],'linked_archives':{n:sha((results/n).read_bytes()) for n in links},'scope':w['scope']}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
