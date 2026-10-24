"""Install exactly two validation-only MTP budget changes after failed-run closure."""
from pathlib import Path
import hashlib,json,subprocess
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
plan_path=methods/'evidence/mtp-b4-t4-builder256-source-policy.json';plan=json.loads(plan_path.read_text())
proof_path=results/'shared-mtp3-full-b4-t4-host-embedding-attempt1-verified.json';proof=json.loads(proof_path.read_text())
assert proof['status']=='host_embedding_mtp3_full_b4_t4_failed_evidence_verified' and not proof['model_gate_passed']
assert not proof['comparisons'] and not proof['state_witness_checks'] and proof['raw_last_hidden_arrays_verified']==0
archive=results/'shared-mtp3-full-b4-t4-host-embedding-attempt1-evidence.zip'
assert sha(archive.read_bytes())==proof['archive_sha256']==plan['prior_failed_archive_sha256']
closed=[json.loads(line) for line in (results/'shared-mtp3-full-b4-t4-host-embedding-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert closed[-1]['event']=='finished' and closed[-1]['exit_code']==1
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==plan['native_library_sha256']==proof['library_sha256']
names={'xModel/qwen35/vl_0_8b/cuda_mtp.py','xModel/qwen35/vl_0_8b/cuda_speculative.py'};assert set(plan['files'])==names
old={path:(root/'Garnet'/path).read_bytes() for path in names};new={path:(results/'mtp-b4-builder256-source'/Path(path).name).read_bytes() for path in names}
for path,policy in plan['files'].items():
    assert sha(old[path])==policy['old_sha256'] and sha(new[path])==policy['new_sha256']
    previous=old[path].decode().replace('\r\n','\n');candidate=new[path].decode().replace('\r\n','\n')
    assert previous.count(policy['old_validation'])==1 and previous.replace(policy['old_validation'],policy['new_validation'],1)==candidate
cpu_path=methods/'evidence/mtp-b4-t4-builder256-cpu-tests.json';cpu=json.loads(cpu_path.read_text())
assert cpu['status']=='bounded_b4_t4_builder256_cpu_policy_tests_passed' and cpu['source_policy_sha256']==sha(plan_path.read_bytes())
assert len(cpu['tests'])==2 and all(t['exit_code']==0 for t in cpu['tests'])
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
receipt_path=results/'mtp-b4-builder256-source-receipt.json';assert not receipt_path.exists()
backup=results/'mtp-b4-builder256-original';backup.mkdir(exist_ok=False)
for path,raw in old.items():(backup/Path(path).name).write_bytes(raw)
receipt={'status':'installing_validation_only','library_sha256':digest,'source_policy_sha256':sha(plan_path.read_bytes()),'cpu_tests_sha256':sha(cpu_path.read_bytes()),'prior_failed_proof_sha256':sha(proof_path.read_bytes()),'old_source_sha256':{p:sha(b) for p,b in old.items()},'new_source_sha256':{p:sha(b) for p,b in new.items()}}
try:
    for path,raw in new.items():(root/'Garnet'/path).write_bytes(raw)
    for path,raw in new.items():assert (root/'Garnet'/path).read_bytes()==raw
    assert sha(library.read_bytes())==digest
    receipt['status']='validation_only_policy_installed_unqualified'
except BaseException as error:
    for path,raw in old.items():(root/'Garnet'/path).write_bytes(raw)
    receipt.update(status='validation_only_install_failed_restored',error=repr(error));raise
finally:
    receipt['scope']='Two validation blocks only, no native DLL/model math change. B4/T4/cap4096 explicit256MiB, defaults0/128 preserved. Fresh GPU correctness required.'
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
