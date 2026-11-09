"""Combine existing exact FP32 four-row fusion with qualified shared-KV verification profiles."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;Q=M/'mistral-causal-batch4-helper1';N=M/'mistral-gqa16-value-native1'
assert not (R/'source1').exists();subprocess.run(['/bin/cp','-cR',str(Q/'source1'),str(R/'source1')],check=True)
for p in [R/'source1',*(R/'source1').rglob('*')]:
 if not p.is_symlink():p.chmod(p.stat().st_mode|0o200|(0o100 if p.is_dir() else 0))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
files={str(p.relative_to(R/'source1')):sha(p) for p in sorted((R/'source1').rglob('*')) if p.is_file()};assert files==read(Q/'source-bindings1.json')['files_sha256']
(R/'source-bindings1.json').write_text(json.dumps(dict(passed=True,files_sha256=files,unchanged_from_base=True,base_bindings_sha256=sha(Q/'source-bindings1.json')),indent=2)+'\n')
s=(N/'b1_worker1.py').read_text();assert s.count('assert not fused_enabled')==1;s=s.replace('assert not fused_enabled',"assert 'native_mistral_verification_gqa16_scores32_optional_v1' in garnet.backend_info() and 'native_mistral_verification_gqa16_values64_32_optional_v1' in garnet.backend_info()")
(R/'b1_worker1.py').write_text(s);compile(s,'b1_worker1.py','exec')
(R/'input1.json').write_bytes((M/'mistral-bounded-draft-long1/input1.json').read_bytes())
plan=dict(scope='Existing FP32 rows4 fused gate/up path plus current GQA16 QK/value verification; real causal helper, B1T4. No comparator reruns.',base_native=str(Q),base_native_provenance_sha256=sha(Q/'provenance1.json'),base_worker=str(N/'b1_worker1.py'),base_worker_sha256=sha(N/'b1_worker1.py'),changed_source_files=[],compiled_option='GARNET_METAL_FUSED_Q4_DECODE',control_library=str(Q/'build-on1'),candidate_library=str(R/'build-on1'),input_sha256=sha(R/'input1.json'),new_justification='Earlier decision1 in mistral-fused-q4-decode-native1 left real learned-draft performance unmeasured. Current source explicitly admits B1/T4 and prepares fused profiles in PrepareVerification. Combine with subsequently qualified shared-QK/value kernels.',prior_decision_sha256=sha(M/'mistral-fused-q4-decode-native1/decision1.json'),prefill_unchanged=True,xmodel_operator_changes=False,no_comparator_execution=True,goal_complete=False)
(R/'plan1.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared unchanged source with one compile-profile option and a causal worker.')
