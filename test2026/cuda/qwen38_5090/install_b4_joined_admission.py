"""Install bounded B4 joined admission only after previous profiling closure."""
from pathlib import Path
import hashlib,json,subprocess
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090';sha=lambda b:hashlib.sha256(b).hexdigest()
policy_path=methods/'evidence/b4-joined-mlp-source-policy.json';policy=json.loads(policy_path.read_bytes())
assert policy['status']=='experimental_b4_joined_mlp_cpu_policy_only' and policy['cpu_tests_passed']==3 and not policy['default_enabled']
trace='b4-mtp-state-wall-trace-attempt1';proof=json.loads((results/(trace+'-verified.json')).read_bytes())
assert proof['correctness_passed'] and proof['raw_last_hidden_arrays_verified']==8
assert sha((results/(trace+'-evidence.zip')).read_bytes())==proof['archive_sha256']==policy['decision_trace_archive_sha256']
events=[json.loads(l) for l in (results/trace/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==policy['native_library_sha256']
source=root/'Garnet'/policy['file'];old=source.read_bytes();candidate=results/'b4-joined-mlp-staged.py';new=candidate.read_bytes()
assert sha(old)==policy['old_sha256'] and sha(new)==policy['new_sha256']
assert old.decode()==policy['old_source'] and new.decode()==policy['new_source']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
backup=results/'b4-joined-mlp-original.py';receipt=results/'b4-joined-mlp-install-receipt.json';assert not backup.exists() and not receipt.exists()
backup.write_bytes(old)
try:
 source.write_bytes(new);assert source.read_bytes()==new and sha(library.read_bytes())==policy['native_library_sha256']
except BaseException:
 source.write_bytes(old);raise
result={'status':'experimental_b4_joined_admission_installed_unqualified','source_policy_sha256':sha(policy_path.read_bytes()),'old_sha256':sha(old),'new_sha256':sha(new),'native_library_sha256':policy['native_library_sha256'],'trace_archive_sha256':proof['archive_sha256'],'gpu_correctness_admitted':False}
receipt.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
