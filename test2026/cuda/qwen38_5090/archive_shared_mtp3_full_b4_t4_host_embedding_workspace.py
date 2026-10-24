"""Archive closed B2 new-DLL device/host model regression, success or failure."""
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
mode='host';name='shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1'
folder=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
workflow=json.loads((folder/'workflow.json').read_text())
events=[json.loads(line) for line in (folder/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
digest=sha((root/'build/bin/garnet.dll').read_bytes());assert digest==workflow['library_sha256']==build['library_sha256']
files=[p for p in folder.rglob('*') if p.is_file()]
files += [root/'Garnet'/path for key in ('candidate_source_sha256','unchanged_policy_source_sha256') for path in build[key]]
files += [root/'Garnet'/path for path in workflow['workspace_source_sha256']]
files += [methods/path for path in ('probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py','run_shared_mtp3_full_b4_t4_host_embedding_workspace.py','archive_shared_mtp3_full_b4_t4_host_embedding_workspace.py','verify_shared_mtp3_full_b4_t4_host_embedding_workspace.py','run_logged.py','monitor_native_probe.py','sample_vram.py')]
files += [root/'test-shared-mtp3-full-b4-t4-host-embedding-workspace.bat',build_path,results/'host-embedding-rows-attempt1-verified.json',results/'shared-mtp3-full-b2-deferred-attempt1-verified.json',results/'shared-mtp3-full-b2-deferred-attempt1/native/report.json',results/'batch-mtp-deferred-http-pilot-attempt1-verified.json',results/'study-ssd-engine-cache/receipt.json',Path(__file__)]
files += [results/'native-mtp3-http'/(key+suffix) for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green') for suffix in ('-request.json','-response.json')]
links=['host-embedding-rows-attempt1-evidence.zip','shared-mtp3-full-b2-deferred-attempt1-evidence.zip','batch-mtp-deferred-http-pilot-attempt1-evidence.zip']
if mode=='host':
    files += [results/'shared-mtp3-full-b2-host-embedding-device-attempt1-verified.json']
    links += ['shared-mtp3-full-b2-host-embedding-device-attempt1-evidence.zip']
files += [results/'shared-mtp3-full-b2-host-embedding-host-attempt1-verified.json',results/'batch-mtp-host-embedding-http-pilot-attempt1-verified.json',results/'shared-full-b4-t4-attempt1-verified.json',methods/'evidence/b4-t4-host-embedding-mtp3-workspace-preparation.json']
links += ['shared-mtp3-full-b2-host-embedding-host-attempt1-evidence.zip','batch-mtp-host-embedding-http-pilot-attempt1-evidence.zip','shared-full-b4-t4-attempt1-evidence.zip']
files += [methods/'install_b4_builder256_policy.py',methods/'test_mtp_builder_workspace.py',methods/'test_mtp_session_workspace_policy.py',results/'mtp-b4-builder256-source-receipt.json',results/'shared-mtp3-full-b4-t4-host-embedding-attempt1-verified.json',methods/'evidence/mtp-b4-t4-builder256-source-policy.json',methods/'evidence/mtp-b4-t4-builder256-cpu-tests.json']
links += ['shared-mtp3-full-b4-t4-host-embedding-attempt1-evidence.zip']
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'mode':mode,'library_sha256':digest,'files':{path:sha(data) for path,data in raw.items()},'linked_archives':{path:sha((results/path).read_bytes()) for path in links},'scope':'Closed B4 T4 host MTP3 retained-ID and same-profile state witness, raw last-hidden and legacy cross-profile diagnostics. DLL identity is disk/source receipt, not independent loaded-module attestation. Raw state arrays not retained; no performance admission.'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for path,data in raw.items():z.writestr(path,data)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
