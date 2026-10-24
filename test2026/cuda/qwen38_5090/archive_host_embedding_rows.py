"""Retain host-embedding native build and selected-row gate, including failures."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
folder=results/'host-embedding-rows-attempt1';build_folder=results/'native-host-embedding-build-attempt1'
target=results/'host-embedding-rows-attempt1-evidence.zip';assert not target.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
for ledger in (folder/'fixture-ledger/commands.jsonl',folder/'native-ledger/commands.jsonl',build_folder/'build-ledger/commands.jsonl'):
    events=[json.loads(line) for line in ledger.read_text().splitlines()];assert events[-1]['event']=='finished'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
build=json.loads((build_folder/'manifest.json').read_text());library=root/'build/bin/garnet.dll'
assert sha(library.read_bytes())==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
assert sha(Path(build['retained_library']).read_bytes())==build['old_library_sha256']
files=[p for directory in (folder,build_folder) for p in directory.rglob('*') if p.is_file()]
files += [root/'Garnet'/name for key in ('candidate_source_sha256','unchanged_policy_source_sha256') for name in build[key]]
files += [methods/name for name in ('host_embedding_fixture.py','probe_host_embedding_rows.py','run_host_embedding_rows.py','verify_host_embedding_rows.py','verify_host_embedding_rows_archive.py','run_native_host_embedding_build.py','run_logged.py','monitor_native_probe.py','sample_vram.py','evidence/host-embedding-source-handoff.json')]
files += [root/'test-host-embedding-rows.bat',root/'build-layer0-stages.bat',results/'batch-mtp-deferred-http-pilot-attempt1-verified.json',results/'native-full-attention-fp8-build-attempt1/manifest.json',Path(__file__)]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'library_sha256':build['library_sha256'],'original_library_sha256':build['old_library_sha256'],
    'files':{name:sha(data) for name,data in raw.items()},
    'linked_archives':{'batch-mtp-deferred-http-pilot-attempt1-evidence.zip':sha((results/'batch-mtp-deferred-http-pilot-attempt1-evidence.zip').read_bytes())},
    'scope':'Closed incremental native build plus selected embedding raw rows/layout/lifecycle. DLLs retained remotely by hash, not embedded. No independently loaded-module receipt, sanitizer/full-model/HTTP/performance admission.'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in raw.items():z.writestr(name,data)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
