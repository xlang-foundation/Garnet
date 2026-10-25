"""Freeze terminal diagnostic outputs; no timing or model admission."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-prefill-diagnostic-attempt1';out=results/name
w=json.loads((out/'workflow.json').read_bytes());assert w['cleanup_finished'] and w['status'] in ('failed','bf16_prefill_diagnostic_finished_needs_replay')
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
for c in w['profiles']:
 events=[json.loads(l) for l in (out/c['mode']/'ledger/commands.jsonl').read_text().splitlines()]
 assert events[-1]['event']=='finished' and events[-1]['exit_code']==c['exit_code']
prep_path=methods/'evidence/bf16-prefill-diagnostic-preparation.json';prep=json.loads(prep_path.read_bytes());assert sha(prep_path.read_bytes())==w['preparation_sha256']
files=[p for p in out.rglob('*') if p.is_file()]+[repo/n for n in w['source_sha256']]+[methods/n for n in prep['methods']]+[root/n for n in prep['pinned_inputs']]
files += [prep_path,methods/'archive_bf16_prefill_diagnostic.py',methods/'verify_bf16_prefill_diagnostic.py',results/'bf16-model-b4-t32-attempt1-verified.json',results/'study-ssd-engine-cache/receipt.json']
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
parent='bf16-model-b4-t32-attempt1-evidence.zip';m={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':{parent:sha((results/parent).read_bytes())},'scope':w['scope']}
archive=results/(name+'-evidence.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,v in raw.items():z.writestr(n,v)
 z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
print(json.dumps({'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
