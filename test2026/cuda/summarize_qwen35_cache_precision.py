"""Consolidate actual cache-view diagnostics and retained weight-cast warnings."""
import argparse
import hashlib
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'vllm-cache-precision-diagnostic-summary.json'
assert not output.exists(), 'Preserve consolidated cache evidence'
cases = []
sources = []
tags = ('diagnostic-vllm-cache-2b-b1-mtp3-fp32conv-v1',
        'diagnostic-vllm-cache-2b-b1-mtp3-bf16conv-v1',
        'diagnostic-vllm-cache-4b-b8-serial-fp32conv-v1')
pattern = re.compile(r'Attempted to load weight (\S+) with dtype (torch\.\w+) into parameter with dtype (torch\.\w+)')
for tag in tags:
    path = a.root / (tag + '-summary.json')
    blob = path.read_bytes()
    data = json.loads(blob)
    assert data['all_passed'] and data['all_12_prior_input_prompt_output_stop_cache_controls_exact']
    assert data['diagnostic_only'] and not data['primary_performance_eligible']
    for source in data['sources']:
        assert hashlib.sha256((a.root / source['file']).read_bytes()).hexdigest() == source['sha256']
    sources.append({'file': path.name, 'sha256': hashlib.sha256(blob).hexdigest()})
    log_path = a.root / (tag + '.log')
    log_blob = log_path.read_bytes()
    sources.append({'file': log_path.name, 'sha256': hashlib.sha256(log_blob).hexdigest()})
    casts = sorted(set(pattern.findall(log_blob.decode('utf-8-sig'))))
    stages = data['stages']
    assert [stage['event'] for stage in stages] == ['loaded', 'after_warmup', 'complete']
    final = stages[-1]
    cases.append({
        'tag': tag, 'model': data['model'], 'batch': data['batch'],
        'same_gpu_uuid': data['same_gpu_uuid_as_saved_lmstudio'],
        'conv_dtype': final['conv_dtype'], 'recurrent_dtype': final['recurrent_dtype'],
        'bound_role_dtype_counts': final['typed_role_dtype_counts'],
        'raw_runner_dtype_counts': final['raw_runner_dtype_counts'],
        'unique_referenced_storages': final['unique_referenced_storages'],
        'referenced_cache_storage_bytes': final['unique_referenced_storage_bytes'],
        'logical_view_bytes_not_additive': final['logical_view_bytes_not_additive'],
        'worker_allocated_bytes': final['worker_memory']['allocated_bytes'],
        'worker_reserved_bytes': final['worker_memory']['reserved_bytes'],
        'stage_allocator_counters': [{'event': s['event'], 'worker_memory': s['worker_memory']} for s in stages],
        'loader_dtype_warnings': [{'name_as_reported': name, 'checkpoint_dtype_reported': old,
                                   'parameter_dtype_reported': new} for name, old, new in casts],
        'unsupported_entries': final['unsupported_entries'],
    })
result = {
    'all_three_diagnostics_passed': True, 'diagnostic_only': True,
    'primary_performance_eligible': False, 'cases': cases, 'sources': sources,
    'scope': 'Three sequential CUDA diagnostic processes with exact saved configuration except the worker '
             'extension. Actual bound typed views and backing-storage deduplication, three snapshots and12 '
             'prior input/prompt/output/stop/cache controls each. Logical alias views are not additive memory. '
             'Worker allocator allocated/reserved counters do not establish Windows process totals or physical '
             'residency. Loader warnings are retained precision provenance, not a complete parameter census or '
             'independent quality proof. Equal checkpoint and nominal BF16 settings do not establish identical '
             'norm-weight or intermediate precision. Historical latency cohorts remain unchanged.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_three_diagnostics_passed': True, 'cases': [{
    key: case[key] for key in ('model', 'batch', 'conv_dtype', 'recurrent_dtype',
                               'referenced_cache_storage_bytes', 'worker_allocated_bytes',
                               'worker_reserved_bytes')
} for case in cases]}, indent=2))
