"""Freeze both original-model warmed NVTX diagnostics before any model load."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve();repo = root/'Garnet'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
stem = 'qwen35-nvtx-stage-v1';output=root/(stem+'-before-run-identity.json')
assert not output.exists()
assert read(root/'qwen35-trt-layer-v2-controller-exit.json')['exit_code'] == 1
rejected = read(root/'qwen35-trt-layer-v2-rejected-analysis.json')
assert rejected['all56_input_output_work_and_layer_metadata_checks_passed'] and not rejected['diagnostic_accepted']
assert read(root/'qwen35-nvtx-cpu-selftest-v1-summary.json')['actual_nvtx_names_nesting_and_single_thread_passed']
assert read(root/'qwen35-nvtx-attribution-v1-cpu-controls.json')['all_six_attribution_controls_passed']
build = read(root/'qwen35-nvtx-bridge-v1-build-inputs.json')
assert build['source_sha256'].lower() == digest(repo/'test2026/cuda/qwen35_nvtx_bridge.cpp')
for name, expected in build['nvtx_headers_sha256'].items():assert digest(Path(name)) == expected.lower()
binary = root/'qwen35_nvtx_bridge_v1.dll'
assert digest(binary) == read(root/'qwen35-nvtx-bridge-v1-exit.json')['dll_sha256'].lower()
assert digest(binary) == read(root/'qwen35-nvtx-cpu-selftest-v1-summary.json')['sources_sha256'][binary.name]
prior = root/'qwen35-trt-layer-v2-before-run-identity.json'
subprocess.run([sys.executable, str(repo/'test2026/cuda/validate_qwen35_execution_identity.py'), str(prior)], check=True)
identity = read(prior)
identity['capture_provenance'] = {'parent_manifest': prior.name,'parent_manifest_sha256':digest(prior),
    'parent_captured_at_utc':identity['captured_at_utc'],'checkpoint_full_hashes':'Inherited full digests with size/mtime guard revalidation, no new payload scan.'}
identity['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('qwen35_nvtx_bridge.cpp','profile_qwen35_nvtx_stages.x','profile_qwen35_nvtx_stages.py',
             'probe_qwen35_nvtx_bridge.py','audit_qwen35_nvtx_cpu.py','qwen35_nvtx_attribution.py',
             'test_qwen35_nvtx_attribution.py','summarize_qwen35_nvtx_trace.py','summarize_qwen35_nvtx_stages.py','freeze_qwen35_nvtx_stages.py'):
    identity['source_sha256']['test2026/cuda/'+name]=digest(repo/'test2026/cuda'/name)
for name in ('run-qwen35-nvtx-stage-v1.ps1','prepare-qwen35-nvtx-stage-v1.py','build-qwen35-nvtx-bridge-v1.ps1',
             stem+'-prospective-design.json',stem+'-nsys-version.log','qwen35_nvtx_bridge_v1.dll',
             'qwen35-nvtx-bridge-v1-build-inputs.json','qwen35-nvtx-bridge-v1-exit.json',
             'qwen35-nvtx-cpu-selftest-v1-summary.json','qwen35-nvtx-attribution-v1-cpu-controls.json',
             'qwen35-trt-layer-v2-controller-exit.json','qwen35-trt-layer-v2-rejected-analysis.json'):
    identity['orchestration_sha256'][name]=digest(root/name)
identity['external_input_sha256'].update({name: expected.lower() for name,expected in build['nvtx_headers_sha256'].items()})
identity['nvtx_scope']='CPU-only annotations in warmed request8; original native policy fixed, no layer profiler flag. Both models sequential; full28 workload/work/owner gates, marker protocol and >=99percent kernel count/time API correlation required. No primary latency/memory promotion.'
output.write_text(json.dumps(identity,indent=2)+'\n')
print('QWEN35_NVTX_STAGE_SOURCE_EPOCH_READY')
