"""Close and retain all compact warm diagnostic data, including failed attempts."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
name='kv-delta-decode-cuda-trace-attempt1';out=results/name;target=results/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
w=json.loads((out/'workflow.json').read_bytes());assert w['cleanup_finished'] and w['status'] in ('failed','decode_cuda_trace_finished_needs_independent_replay')
ledger=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()]
assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
files=[p for p in out.rglob('*') if p.is_file()]
for key,prefix in (('source_sha256',root/'Garnet'),('helper_sha256',methods),('method_sha256',methods)):
    for n,h in w[key].items():
        p=prefix/n;assert sha(p.read_bytes())==h;files.append(p)
files += [methods/n for n in ('evidence/kv-delta-decode-cuda-trace-preparation.json','run_logged.py','monitor_native_probe.py','sample_vram.py')]
files += [root/'test-kv-delta-decode-cuda-trace.bat',root/'models/Qwen3.8-27B-NVFP4/config.json',results/'nsight-2026-cab-extraction/setup.json']
files += [results/n for n in ('padded-mtp-http-pilot-attempt1-verified.json','padded-mtp-http-pilot-attempt1/workflow.json','kv-delta-full-b4-t4-attempt1-verified.json')]
links={'nsight-cuda-smoke-2026-attempt1-evidence.zip':w['nsight_smoke_archive_sha256'],'padded-mtp-http-pilot-attempt1-evidence.zip':w['parent_http_archive_sha256'],'kv-delta-full-b4-t4-attempt1-evidence.zip':w['full_model_archive_sha256']}
for n,h in links.items():assert sha((results/n).read_bytes())==h
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'library_sha256':w['library_sha256'],'linked_archives':links,'scope':w['scope']}
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
