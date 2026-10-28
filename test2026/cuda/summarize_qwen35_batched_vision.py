"""Preserve batch-vision component passes and rejections; no model admission."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--attempt', required=True)
a = p.parse_args()
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


cases = []
for size in ('2b', '4b'):
    for batch in (2, 4, 8):
        tag = f'batched-vision-{size}-b{batch}-plain-{a.attempt}'
        result = read(a.root / (tag + '.json'))
        terminal = read(a.root / (tag + '-exit.json'))
        assert result['vision_batch'] == batch and result['model'] == f'Qwen3.5-{size.upper()}'
        assert result['absolute_tolerance'] == 2e-5 and result['relative_tolerance'] == 2e-4
        assert result['serial_engine_closed_before_batched_load']
        assert result['batched_builder_workspace_limit_bytes'] == terminal['builder_workspace_limit_bytes']
        serial = [row for row in result['records'] if row['mode'] == 'serial']
        assert len(serial) == 16 and {row['image_index'] for row in serial} == set(range(8))
        for index in range(8):
            assert len({row['visual_sha256'] for row in serial if row['image_index'] == index}) == 1
        compared = [row for row in result['records'] if 'within_original_bound' in row]
        accepted = terminal['native_exit_code'] == 0 and result.get('all_numeric_bounds_passed') is True
        assert accepted == terminal['method_accepted']
        sanitizer = None
        if accepted:
            assert result['status'] == 'COMPONENT_PASSED_FULL_REQUEST_STILL_REQUIRED'
            assert len(result['records']) == 56 and all(row['within_original_bound'] for row in compared)
            graph = result['batched_graph_stats']
            assert graph['contexts'] == graph['graph_captures'] == 1
            assert graph['graph_replays'] > 0 and graph['graph_capture_failures'] == 0
            memtag = tag.replace('-plain-', '-memcheck-')
            mem = read(a.root / (memtag + '.json'))
            memexit = read(a.root / (memtag + '-exit.json'))
            assert memexit['execution_identity_sha256'] == terminal['execution_identity_sha256']
            assert memexit['native_exit_code'] == 0 and mem['all_numeric_bounds_passed']
            raw = (a.root / (memtag + '.log')).read_bytes()
            sources[memtag + '.log'] = hashlib.sha256(raw).hexdigest()
            assert b'ERROR SUMMARY: 0 errors' in raw
            sanitizer = 'unfiltered_memcheck_zero_errors'
        else:
            assert terminal['original_bound_rejected']
            assert terminal['native_exit_code'] != 0 and result['status'] == 'REJECTED_ORIGINAL_NUMERICAL_BOUND'
            assert any(not row['within_original_bound'] for row in compared)
        cases.append({'model': size, 'batch': batch, 'component_accepted': accepted,
                      'status': result['status'], 'native_exit_code': terminal['native_exit_code'],
                      'builder_workspace_limit_bytes': result['batched_builder_workspace_limit_bytes'],
                      'compared_rows': len(compared),
                      'rows_outside_original_bound': sum(not row['within_original_bound'] for row in compared),
                      'maximum_absolute_error': max(row['max_absolute_error'] for row in compared),
                      'maximum_bound_fraction': max(row['max_bound_fraction'] for row in compared),
                      'sanitizer': sanitizer,
                      'execution_identity_sha256': terminal['execution_identity_sha256']})

output = a.root / f'batched-vision-{a.attempt}-summary.json'
assert not output.exists(), 'Preserve component evidence'
summary = {'sources_sha256': sources, 'cases': cases,
           'all_six_profiles_screened': len(cases) == 6,
           'accepted_component_profiles': sum(row['component_accepted'] for row in cases),
           'full_request_accepted': False, 'independent_vision_quality_accepted': False,
           'primary_performance_eligible': False,
           'scope': 'Native serial versus existing batch-aware vision, eight distinct images, original '
                    '2e-5 absolute + 2e-4 relative bounds. Row-order/replay checks and sanitizer gate remain '
                    'prospective. Rejections are retained; no production routing or default promotion. '
                    'Component evidence does not establish independent quality, full requests, memory, '
                    'latency, or the 2x goal.'}
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(cases, indent=2))
