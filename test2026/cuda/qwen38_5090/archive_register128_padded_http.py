"""Preserve the terminal compact HTTP pilot, raw errors and all timing cohorts."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='register128-padded-http-pilot-attempt1';out=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
w=json.loads((out/'workflow.json').read_bytes());assert w['status'] in ('failed','register128_http_finished_needs_independent_replay') and w['cleanup_finished']
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
files=[p for p in out.rglob('*') if p.is_file()]
for n,h in w['source_sha256'].items():
    p=root/'Garnet'/n;assert sha(p.read_bytes())==h;files.append(p)
for group in ('stream_source_sha256','http_helper_sha256'):
    for n,h in w[group].items():assert sha((methods/n).read_bytes())==h;files.append(methods/n)
files += [methods/n for n in ('probe_register128_padded_http.py','test_register128_padded_http.bat','run_register128_padded_http.py','archive_register128_padded_http.py','verify_register128_padded_http.py','run_logged.py','monitor_native_probe.py','sample_vram.py','evidence/register128-padded-http-preparation.json')]
files += [root/n for n in ('test-register128-padded-http.bat','serve-native-kv-delta-register128-padded-mtp-worker.bat')]
files += [results/n for n in ('register128-padded-stream-attempt1-verified.json','register128-padded-stream-attempt1/workflow.json','kv-delta-full-b4-t4-attempt1-verified.json','padded-mtp-http-pilot-attempt1-verified.json','unicode-transport-repair2-verified.json','batch-http-serialization-regression.json','batch-http-pilot-attempt1-failure-verified.json')]
files += [results/'native-mtp3-http'/(key+suffix) for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green') for suffix in ('-request.json','-response.json')]
files += [root/'models/Qwen3.8-27B-NVFP4/config.json',results/'register128-padded-state-attempt1-verified.json',methods/'verify_register128_padded_stream.py',methods/'verify_register128_padded_state.py']
links={'register128-padded-state-attempt1-evidence.zip':w['padded_state_archive_sha256'],'register128-padded-stream-attempt1-evidence.zip':w['stream_archive_sha256'],'kv-delta-full-b4-t4-attempt1-evidence.zip':w['full_model_archive_sha256'],'padded-mtp-http-pilot-attempt1-evidence.zip':w['parent_http_archive_sha256'],**w['reference_archives']}
for n,h in links.items():assert sha((results/n).read_bytes())==h
prep=json.loads((methods/'evidence/register128-padded-http-preparation.json').read_bytes())
files += [methods/n for n in prep['cpu_sources']] + [results/n for n in prep['cpu_results']]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'library_sha256':w['library_sha256'],'linked_archives':links,'scope':w['scope']}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
