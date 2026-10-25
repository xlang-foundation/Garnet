"""Archive the terminal BF16 model integration gate, including a failure."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-model-b4-t32-attempt1';out=results/name
w=json.loads((out/'workflow.json').read_bytes());assert w['cleanup_finished'] and w['status'] in ('failed','bf16_b4_t32_finished_needs_replay')
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
prep_path=methods/'evidence/bf16-model-b4-t32-preparation.json';prep=json.loads(prep_path.read_bytes())
assert sha(prep_path.read_bytes())==w['preparation_sha256']
files=[p for p in out.rglob('*') if p.is_file()]+[repo/n for n in w['source_sha256']]+[methods/n for n in prep['methods']]
files += [root/n for n in prep['pinned_inputs']]+[prep_path,results/'bf16-kv-metadata-fixture.json',results/'study-ssd-engine-cache/receipt.json',root/'models/Qwen3.8-27B-NVFP4/config.json']
links=['bf16-model-integration-build-attempt1-evidence.zip','bf16-model-default-off-attempt1-evidence.zip','native-kv-delta-default-off-attempt1-evidence.zip']
files += [results/n.replace('-evidence.zip','-verified.json') for n in links]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
m={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':{n:sha((results/n).read_bytes()) for n in links},'scope':w['scope']}
archive=results/(name+'-evidence.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
print(json.dumps({'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
