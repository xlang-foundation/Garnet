"""Preserve a closed B4 diagnostic, including raw failure output if unsuccessful."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='b4-mtp-state-wall-trace-attempt1';folder=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
workflow=json.loads((folder/'workflow.json').read_bytes());events=[json.loads(line) for line in (folder/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_bytes())
digest=sha((root/'build/bin/garnet.dll').read_bytes());assert digest==workflow['library_sha256']==build['library_sha256']
files=[f for f in folder.rglob('*') if f.is_file()]
files += [root/'Garnet'/n for group in ('candidate_source_sha256','unchanged_policy_source_sha256') for n in build[group]]
files += [root/'Garnet'/n for n in workflow['workspace_source_sha256']]
files += [methods/n for n in ('probe_b4_mtp_state_wall_trace.py','run_b4_mtp_state_wall_trace.py','archive_b4_mtp_state_wall_trace.py','verify_b4_mtp_state_wall_trace.py','probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py','run_logged.py','monitor_native_probe.py','sample_vram.py')]
files += [root/'test-b4-mtp-state-wall-trace.bat',build_path]
links=['shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-evidence.zip','batch4-t4-mtp-host-embedding-http-pilot-attempt1-evidence.zip']
files += [results/n.replace('-evidence.zip','-verified.json') for n in links]
raw={f.relative_to(root).as_posix():f.read_bytes() for f in files}
manifest={'library_sha256':digest,'files':{n:sha(b) for n,b in raw.items()},'linked_archives':{n:sha((results/n).read_bytes()) for n in links},'scope':'Closed synchronized nested wall diagnostics and correctness rerun. Not uninstrumented latency, kernel timing or serving speed.'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
