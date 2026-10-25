"""Preserve the terminal default-off model result, including failures."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sha=lambda b:hashlib.sha256(b).hexdigest();name='bf16-model-default-off-attempt1';out=results/name
w=json.loads((out/'workflow.json').read_bytes());assert w['cleanup_finished'] and w['status'] in ('failed','bf16_default_off_finished_needs_replay')
events=[json.loads(l) for l in (out/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert sha((root/'build/bin/garnet.dll').read_bytes())==w['library_sha256']
for n,h in w['source_sha256'].items():assert sha((repo/n).read_bytes())==h
files=[p for p in out.rglob('*') if p.is_file()]+[repo/n for n in w['source_sha256']]
files += [root/'test-b4-host-joined-mtp.bat',results/'study-ssd-engine-cache/receipt.json']
files += [methods/n for n in ('run_bf16_model_default_off.py','archive_bf16_model_default_off.py','verify_bf16_model_default_off.py','probe_b4_host_joined_mtp.py','run_logged.py','monitor_native_probe.py','sample_vram.py')]
links=['bf16-model-integration-build-attempt1-evidence.zip','native-kv-delta-default-off-attempt1-evidence.zip']
files += [results/n.replace('-evidence.zip','-verified.json') for n in links]
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
with zipfile.ZipFile(results/'native-kv-delta-default-off-attempt1-evidence.zip') as previous:
    prior=json.loads(previous.read('results/native-kv-delta-default-off-attempt1/workflow.json'))
unchanged={n:h for n,h in prior['source_sha256'].items() if n not in w['source_sha256']}
observed={}
for n,h in unchanged.items():
    data=(repo/n).read_bytes();raw['Garnet/'+n]=data;observed[n]=sha(data)
m={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':{n:sha((results/n).read_bytes()) for n in links},
   'expected_unchanged_source_sha256':unchanged,'observed_unchanged_source_sha256':observed,
   'unchanged_source_observation_scope':'Post-process readback against retained parent. Only candidate sources were pinned by this run at launch; this extra observation is not a retroactive prelaunch claim.',
   'scope':w['scope']}
archive=results/(name+'-evidence.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
print(json.dumps({'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
