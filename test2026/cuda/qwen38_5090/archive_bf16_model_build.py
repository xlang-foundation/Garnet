"""Freeze completed native build evidence without changing deployed artifacts."""
from pathlib import Path
import hashlib,json,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
results=root/'results';name='bf16-model-integration-build-attempt1';out=results/name
sha=lambda b:hashlib.sha256(b).hexdigest()
b=json.loads((out/'manifest.json').read_bytes())
assert b['status']=='bf16_model_native_build_finished_unqualified' and b['exit_code']==0
assert sha((root/'build/bin/garnet.dll').read_bytes())==sha(Path(b['retained_candidate_library']).read_bytes())==b['library_sha256']
assert sha((root/'build/bin/garnet-native-kv-delta-attempt1.dll').read_bytes())==b['old_library_sha256']
for n,h in b['candidate_source_sha256'].items():assert sha((repo/n).read_bytes())==h
for n,h in b['runtime_dependencies_after_sha256'].items():assert sha((root/'build/bin'/n).read_bytes())==h
files=[p for p in out.rglob('*') if p.is_file()]
files += [results/'bf16-model-integration-source.zip',results/'bf16-plugin-cpu-contract-attempt1-verified.json']
files += [methods/n for n in ('evidence/bf16-model-integration-build-preparation.json','evidence/bf16-production-crt-scope-clarification.json','evidence/bf16-model-integration-local-cpu.json','run_bf16_model_integration_build.py','build_bf16_model_integration.bat','archive_bf16_model_build.py','verify_bf16_model_build.py','run_logged.py')]
files += [repo/n for n in b['candidate_source_sha256']]
files += [root/'build/CMakeCache.txt',root/'build/build.ninja',root/'build/CMakeFiles/rules.ninja']
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{n:sha(v) for n,v in raw.items()},'linked_archives':{'bf16-plugin-cpu-contract-attempt1-evidence.zip':sha((results/'bf16-plugin-cpu-contract-attempt1-evidence.zip').read_bytes())},'library_sha256':b['library_sha256'],'scope':b['scope']}
archive=results/(name+'-evidence.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for n,v in raw.items():z.writestr(n,v)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(raw)}))
