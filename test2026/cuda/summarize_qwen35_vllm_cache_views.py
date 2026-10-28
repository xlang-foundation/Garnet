"""Validate diagnostic cache tensors against the exact saved vLLM profile.

No RPC timing is accepted as a benchmark. Worker allocator counters and referenced
tensor storage extents are distinct from Windows process counters and residency.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('tag')
p.add_argument('prior')
a = p.parse_args()
output = a.root / (a.tag + '-summary.json')
assert not output.exists(), 'Preserve cache diagnostic evidence'
sources = []


def read(name):
    path = a.root / name
    blob = path.read_bytes()
    sources.append({'file': name, 'sha256': hashlib.sha256(blob).hexdigest()})
    return json.loads(blob)


terminal = read(a.tag + '-exit.json')
assert terminal['exit_code'] == 0 and terminal['diagnostic_only'] is True
assert terminal['primary_performance_eligible'] is False
config = read(a.tag + '.config.json')
prior_config = read(a.prior + '.config.json')
if a.prior == 'matched512-vllm-4b-b8':
    completion = read('vllm-4b-b8-saved-completion-admission.json')
    assert completion['all_passed']
    for source in completion['sources']:
        assert hashlib.sha256((a.root / source['file']).read_bytes()).hexdigest() == source['sha256']
else:
    assert read(a.prior + '-exit.json')['exit_code'] == 0
assert config['vllm'] == prior_config['vllm'] == '0.32.0'
assert config['torch'] == prior_config['torch'] == '2.13.0+cu130'
current = dict(config['options'])
assert current.pop('worker_extension_cls') == 'qwen35_vllm_memory.MemoryExtension'
assert current == prior_config['options'], 'Diagnostic changed other model/cache settings'
assert config['environment'] == prior_config['environment']
rows = read(a.tag + '.json')
prior = {(row['trial'], row['workload_group']): row for row in read(a.prior + '.json')}
assert len(rows) == 12 and len(prior) == 28
prior_spec = prior_config['options'].get('speculative_config')
assert not prior_spec or prior_spec['method'] == 'mtp'
configured_speculation = {'enabled': bool(prior_spec),
                         'method': 'native checkpoint MTP' if prior_spec else None,
                         'num_speculative_tokens': prior_spec['num_speculative_tokens'] if prior_spec else 0}
missing_speculation_rows = 0
for row in rows:
    reference = prior[(row['trial'], row['workload_group'])]
    assert row['diagnostic_only'] is True
    for key in ('input_sha256', 'batch', 'model', 'warmup', 'request_output_limit', 'cache_policy'):
        assert row[key] == reference[key], (key, row['trial'], row['workload_group'])
    if 'speculative' not in reference:
        missing_speculation_rows += 1
    assert row['speculative'] == reference.get('speculative', configured_speculation) == configured_speculation
    assert len(row['records']) == len(reference['records']) == row['batch']
    for actual, expected in zip(row['records'], reference['records']):
        for key in ('prompt_token_ids', 'token_ids', 'finish_reason', 'num_cached_tokens'):
            assert actual[key] == expected[key], (key, row['trial'], row['workload_group'])
        assert actual['num_cached_tokens'] == 0
snapshots = read(a.tag + '-cache-views.json')
assert [row['event'] for row in snapshots] == ['loaded', 'after_warmup', 'complete']
size = '2b' if rows[0]['model'] == 'Qwen3.5-2B' else '4b'
batch = rows[0]['batch']
lm = read(f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')


def uuid(value):
    return str(value).lower().removeprefix('gpu-').replace('-', '')


expected_uuid = uuid(lm['adapter']['uuid'])
expected_conv = {'float32': 'torch.float32', 'bfloat16': 'torch.bfloat16'}[current['mamba_cache_dtype']]
expected_gdn = 18 if size == '2b' else 24
stages = []
for snapshot in snapshots:
    assert len(snapshot['workers']) == 1, 'Expected tensor parallel size1'
    worker = snapshot['workers'][0]
    assert worker['static_forward_context_available']
    assert worker['typed_layer_view_count'] > 0
    device = worker['cuda_device']
    assert device.get('uuid') and uuid(device['uuid']) == expected_uuid, 'Different or unverified physical GPU'
    gdn = [layer for layer in worker['layers'] if layer['is_gated_delta_layer']]
    assert len(gdn) >= expected_gdn, 'Target GDN layers missing from inventory'
    assert all(layer['bound_dtypes_match_resolved_state_dtypes'] for layer in gdn)
    assert all(layer['bound_view_dtypes'] == [expected_conv, 'torch.float32'] for layer in gdn)
    views = worker['views']
    conv = [v for v in views if v['role'] == 'gated_delta_conv_state']
    recurrent = [v for v in views if v['role'] == 'gated_delta_recurrent_state']
    assert len(conv) == len(recurrent) == len(gdn)
    assert all(v['dtype'] == expected_conv for v in conv)
    assert all(v['dtype'] == 'torch.float32' for v in recurrent)
    storages = worker['unique_referenced_storages']
    ids = {storage['id'] for storage in storages}
    assert len(ids) == len(storages)
    assert len({(s['device'], s['base_pointer_hex'], s['bytes']) for s in storages}) == len(storages)
    assert sum(s['bytes'] for s in storages) == worker['unique_referenced_storage_bytes']
    for view in views:
        assert view['storage_id'] in ids
        assert view['logical_bytes'] == math.prod(view['shape']) * view['element_size_bytes']
        assert view['device'].startswith('cuda')
    stages.append({
        'event': snapshot['event'], 'cuda_device': device, 'gated_delta_layers': len(gdn),
        'conv_dtype': expected_conv, 'recurrent_dtype': 'torch.float32',
        'typed_layer_view_count': worker['typed_layer_view_count'],
        'typed_role_dtype_counts': dict(collections.Counter(v['role'] + ':' + v['dtype'] for v in views if v['role'] != 'raw_runner_cache')),
        'raw_runner_dtype_counts': dict(collections.Counter(v['dtype'] for v in views if v['role'] == 'raw_runner_cache')),
        'unique_referenced_storages': len(storages),
        'unique_referenced_storage_bytes': worker['unique_referenced_storage_bytes'],
        'logical_view_bytes_not_additive': sum(v['logical_bytes'] for v in views),
        'worker_memory': worker['worker_memory'],
        'unsupported_entries': worker['unsupported_entries'],
    })
result = {
    'all_passed': True, 'diagnostic_only': True, 'primary_performance_eligible': False,
    'model': rows[0]['model'], 'batch': batch, 'prior_profile': a.prior,
    'all_12_prior_input_prompt_output_stop_cache_controls_exact': True,
    'same_options_except_worker_extension': True,
    'prior_speculation_rows_without_metadata': missing_speculation_rows,
    'prior_configured_speculation': configured_speculation,
    'same_gpu_uuid_as_saved_lmstudio': lm['adapter']['uuid'], 'stages': stages, 'sources': sources,
    'scope': 'Actual CUDA-bound tensor dtype/shape/stride/storage metadata, including GDN conv/recurrent roles. '
             'Raw byte allocations do not establish state precision. Shared storage is deduplicated per worker, '
             'not across processes. Worker torch allocated/reserved/peaks, logical views and referenced storage '
             'extents are separate measures; none establishes complete process VRAM or physical residency. '
             'Diagnostic RPCs/holds invalidate primary latency acceptance. Twelve saved workload groups match; '
             'this is not independent quality or a complete baseline rerun.',
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('all_passed', 'model', 'batch', 'prior_profile')}, indent=2))
