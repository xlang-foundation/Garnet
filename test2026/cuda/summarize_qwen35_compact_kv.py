"""Standalone compact-KV copy corpus, without serving or memory admission."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-component-v1-summary.json'
assert not output.exists(), 'Preserve component proof'
sources = {}


def blob(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return raw


def read(path):
    return json.loads(blob(path).decode('utf-8-sig'))


identity = read(a.identity)
epoch = sources[a.identity.name]
repo = Path(__file__).resolve().parents[2]
for name in ('src/cuda/qwen35_compact_kv.cuh', 'src/cuda/qwen35_compact_kv.cu',
             'test2026/cuda/probe_qwen35_compact_kv.cu'):
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == identity['source_sha256'][name]
assert read(a.root / 'compact-kv-component-v1-controller-exit.json')['exit_code'] == 0
assert read(a.root / 'compact-kv-component-v1-build-exit.json')['exit_code'] == 0
binary_sha = hashlib.sha256((a.root / 'compact-kv-probe.exe').read_bytes()).hexdigest()
expected = set(itertools.product((1, 2, 4, 8), (2, 4), (384, 512), (4, 64), range(6)))
cases = []
for mode in ('plain', 'memcheck', 'racecheck', 'synccheck'):
    tag = f'compact-kv-component-v1-{mode}'
    terminal = read(a.root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['binary_sha256'].lower() == binary_sha
    assert terminal['execution_identity_sha256'].lower() == epoch
    raw = blob(a.root / (tag + '.log')).decode('utf-8-sig')
    rows = [json.loads(line) for line in raw.splitlines() if line.startswith('{')]
    checks = [row for row in rows if row['event'] == 'case']
    assert len(checks) == 192 and {(row['batch'], row['kv'], row['capacity'], row['tokens'], row['pattern'])
                                   for row in checks} == expected
    assert all(row['all_bytes_and_guards_exact'] for row in checks)
    assert rows[-1]['event'] == 'complete' and rows[-1]['all_passed']
    assert rows[-1]['cases'] == 192 and rows[-1]['negative_checks'] == 160
    if mode != 'plain':
        pattern = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                   if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
        assert re.search(pattern, raw), 'Missing unfiltered zero-error sanitizer report'
    cases.append({'mode': mode, 'whole_buffer_cases': 192, 'negative_checks': 160,
                  'all_exact_and_guards_preserved': True,
                  'sanitizer_zero_errors_or_hazards': None if mode == 'plain' else True})
layout = read(a.root / 'compact-kv-output-layout-plan.json')
summary = {'sources_sha256': sources, 'binary_sha256': binary_sha,
           'garnet_runtime_unchanged_sha256': identity['garnet_binary_sha256'],
           'all_four_modes_passed': True, 'cases': cases,
           'prospective_output_pool_bytes_only': [
               {key: row[key] for key in ('model', 'batch', 'capacity', 'logical_output_pool_difference_bytes_only')}
               for row in layout['cases'] if row['current_layout_matches_measured_profile']],
           'serving_integrated': False, 'measured_allocation_reduction': False,
           'independent_model_quality_accepted': False, 'primary_performance_eligible': False,
           'scope': 'Unwired copy-only pack/scatter corpus; exact complete compact outputs and canonical caches, '
                    'source/slot/active preservation and poisoned guards. Negative metadata/invalid-slot '
                    'checks remain bounded. TensorRT output layout, tensor-owner/alias policy, error-flag '
                    'consumption, accepted-prefix transaction/rollback/poisoning, real models, full requests, '
                    'memory and latency are unqualified. The standalone pass does not admit production '
                    'integration or establish the 2x goal.'}
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'all_four_modes_passed': True, 'whole_buffer_cases_per_mode': 192,
                  'negative_checks_per_mode': 160, 'serving_integrated': False}))
