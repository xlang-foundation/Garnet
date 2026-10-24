"""Archive a terminal standalone attention component study, including failures."""
from pathlib import Path
import hashlib, json, subprocess, zipfile

root=Path('D:/CantorAI/5090-study-20261023'); name='attention-b4-t4-component-attempt1'
out=root/'results'/name; target=root/'results'/(name+'-evidence.zip')
assert not target.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
report=json.loads((out/'manifest.json').read_bytes())
assert report['status'] in ('failed','component_finished_needs_independent_replay')
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==report['library_sha256']
for folder in out.glob('*-ledger'):
    events=[json.loads(l) for l in (folder/'commands.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='finished'
files={p.relative_to(root).as_posix():p.read_bytes() for p in out.rglob('*') if p.is_file()}
for n,want in report['source_sha256'].items():
    if n=='build/bin/garnet.dll': continue
    raw=(root/n).read_bytes();assert sha(raw)==want;files[n]=raw
methods=root/'Garnet/test2026/cuda/qwen38_5090'
for n in ('archive_attention_b4_t4_component.py','verify_attention_b4_t4_component.py','run_logged.py','monitor_native_probe.py','sample_vram.py'):
    p=methods/n;files[p.relative_to(root).as_posix()]=p.read_bytes()
exe=root/'build/bin/garnet_attention_b4_t4_test.exe'
if 'executable_sha256' in report:
    assert sha(exe.read_bytes())==report['executable_sha256']
    files[exe.relative_to(root).as_posix()]=exe.read_bytes()
parent='b4-joined-layer-profile-attempt1-evidence.zip'
assert sha((root/'results'/parent).read_bytes())==report['parent_archive_sha256']
manifest={'files':{n:sha(b) for n,b in files.items()},'linked_archives':{parent:report['parent_archive_sha256']},
          'library_sha256':report['library_sha256'],'scope':report['scope']}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in files.items():z.writestr(n,b)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(files)}))
