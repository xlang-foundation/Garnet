"""Archive closed host-embedding B2 stream protocol and source/proof lineage."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='batch-mtp-host-embedding-stream-protocol-attempt1';folder=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
workflow=json.loads((folder/'workflow.json').read_text());events=[json.loads(line) for line in (folder/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
digest=sha((root/'build/bin/garnet.dll').read_bytes());assert digest==build['library_sha256']==workflow['library_sha256']
files=[p for p in folder.rglob('*') if p.is_file()]
files += [root/'Garnet'/path for key in ('candidate_source_sha256','unchanged_policy_source_sha256') for path in build[key]]
files += [root/'Garnet'/path for path in workflow['workspace_source_sha256']]
files += [methods/path for path in ('native_batch_mtp_host_embedding_stream_worker.py','probe_batch_mtp_host_embedding_stream_protocol.py','run_batch_mtp_host_embedding_stream_protocol.py','archive_batch_mtp_host_embedding_stream_protocol.py','verify_batch_mtp_host_embedding_stream_protocol.py','native_worker_io.py','run_logged.py','monitor_native_probe.py','sample_vram.py')]
files += [root/'serve-native-batch2-mtp-host-embedding-worker.bat',root/'test-batch-mtp-host-embedding-stream-protocol.bat',build_path,results/'shared-mtp3-full-b2-host-embedding-host-attempt1-verified.json',results/'shared-mtp3-full-b2-host-embedding-host-attempt1/workflow.json',results/'unicode-transport-repair2-verified.json',Path(__file__)]
files += [results/'native-mtp3-http'/(key+suffix) for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green') for suffix in ('-request.json','-response.json')]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'library_sha256':digest,'files':{path:sha(data) for path,data in raw.items()},'linked_archives':{path:sha((results/path).read_bytes()) for path in ('shared-mtp3-full-b2-host-embedding-host-attempt1-evidence.zip','batch-mtp-deferred-stream-protocol-attempt1-evidence.zip','unicode-transport-repair2-evidence.zip')},'scope':'Closed B2 host embedding MTP3 stream protocol, source and full-model proof lineage. No loaded-module receipt, HTTP, sanitizer or performance admission.'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for path,data in raw.items():z.writestr(path,data)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
