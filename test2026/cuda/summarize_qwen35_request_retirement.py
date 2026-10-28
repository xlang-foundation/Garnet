"""Summarize instrumented API/retirement boundaries without causal speed claims."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('result', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists(), 'Preserve retirement diagnostics'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


rows, reference = read(a.result), read(a.reference)
assert len(rows) == len(reference) == 28
signature = ('generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids')
for index, (row, old) in enumerate(zip(rows, reference)):
    assert row['input_sha256'] == old['input_sha256']
    assert row['prompt_token_ids'] == old['prompt_token_ids']
    assert row['profile'] == old['profile']
    assert row['cache_policy'] == old['cache_policy']
    assert row['speculative_metrics'] == old['speculative_metrics']
    assert all(row[key] == old[key] for key in signature)
    values = row['retirement_diagnostic']
    assert values['request_index'] == index and values['primary_performance_eligible'] is False
    assert abs(values['fully_retired_wall_seconds'] - values['api_return_wall_seconds'] -
               values['post_return_gpu_retirement_seconds']) < 1e-8
warm = [row['retirement_diagnostic'] for row in rows if not row['warmup']]
assert len(warm) == 20
keys = ('prior_request_barrier_seconds', 'api_return_wall_seconds',
        'post_return_gpu_retirement_seconds', 'fully_retired_wall_seconds')
summary = {'sources_sha256': sources, 'all28_input_output_stop_work_profile_controls_passed': True,
           'warm_requests': len(warm), 'warm_milliseconds': {
               key: {'median': statistics.median(row[key] * 1000 for row in warm),
                     'minimum': min(row[key] * 1000 for row in warm),
                     'maximum': max(row[key] * 1000 for row in warm)} for key in keys},
           'primary_performance_eligible': False,
           'scope': 'All20 instrumented warm groups retained, with unchanged all28 inputs/outputs/stops/'
                    'discrete MTP work and profile. Tail includes CUDA barrier/host overhead and any '
                    'pending context work. No correction-only causality, historical bias magnitude, '
                    'model speed promotion or direct comparison to unsynchronized HTTP baselines.'}
a.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary['warm_milliseconds'], indent=2))
