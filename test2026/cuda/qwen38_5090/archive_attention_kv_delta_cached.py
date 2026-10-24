"""Archive a closed cached KV-delta component, including failures."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');name='attention-kv-delta-cached-attempt1'
out=root/'results'/name;target=root/'results'/(name+'-evidence.zip');assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
report=json.loads((out/'manifest.json').read_bytes())
assert report['status'] in ('failed','cached_component_finished_needs_independent_replay')
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==report['library_sha256']
for folder in out.glob('*-ledger'):
    events=[json.loads(l) for l in (folder/'commands.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='finished'
raw={p.relative_to(root).as_posix():p.read_bytes() for p in out.rglob('*') if p.is_file()}
for n,h in report['source_sha256'].items():
    if n=='build/bin/garnet.dll':continue
    b=(root/n).read_bytes();assert sha(b)==h;raw[n]=b
methods=root/'Garnet/test2026/cuda/qwen38_5090'
for n in ('archive_attention_kv_delta_cached.py','verify_attention_kv_delta_cached.py','run_logged.py','monitor_native_probe.py','sample_vram.py'):
    p=methods/n;raw[p.relative_to(root).as_posix()]=p.read_bytes()
if 'executable_sha256' in report:
    p=root/'build/bin/garnet_attention_kv_delta_cached_test.exe'
    assert sha(p.read_bytes())==report['executable_sha256'];raw[p.relative_to(root).as_posix()]=p.read_bytes()
link='attention-kv-delta-timing-attempt1-evidence.zip'
assert sha((root/'results'/link).read_bytes())==report['parent_archive_sha256']
manifest={'files':{n:sha(b) for n,b in raw.items()},'linked_archives':{link:report['parent_archive_sha256']},'library_sha256':report['library_sha256'],'scope':report['scope']}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in raw.items():z.writestr(n,b)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
