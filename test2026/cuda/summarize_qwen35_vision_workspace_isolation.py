"""Same-batch workspace isolation, distinct from failed batched vision."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
a = p.parse_args()
output = a.root / 'vision-workspace-isolation-v1-summary.json'
assert not output.exists(), 'Preserve isolation evidence'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


identity = read(a.identity)
epoch_sha = sources[a.identity.name]
controller = read(a.root / 'vision-workspace-isolation-v1-controller-exit.json')
assert controller['exit_code'] == 0
cases = []
for size in ('2b', '4b'):
    for mode in ('plain', 'memcheck'):
        tag = f'batched-vision-{size}-b1-{mode}-workspace-isolation-v1'
        terminal = read(a.root / (tag + '-exit.json'))
        result = read(a.root / (tag + '.json'))
        assert terminal['native_exit_code'] == 0 and terminal['method_accepted']
        assert terminal['execution_identity_sha256'].lower() == epoch_sha
        assert result['vision_batch'] == 1 and result['serial_engine_closed_before_batched_load']
        assert result['serial_builder_workspace_limit_bytes'] == 64 << 20
        assert result['batched_builder_workspace_limit_bytes'] == 128 << 20
        assert result['absolute_tolerance'] == 2e-5 and result['relative_tolerance'] == 2e-4
        assert result['all_numeric_bounds_passed'] and len(result['records']) == 56
        assert len({row['path'] for row in result['inputs']}) == 8
        compared = [row for row in result['records'] if 'within_original_bound' in row]
        assert len(compared) == 40 and all(row['bitwise_equal'] for row in compared)
        graph = result['batched_graph_stats']
        assert graph['contexts'] == graph['graph_captures'] == 1
        assert graph['graph_replays'] > 0 and graph['graph_capture_failures'] == 0
        if mode == 'memcheck':
            raw = (a.root / (tag + '.log')).read_bytes()
            sources[tag + '.log'] = hashlib.sha256(raw).hexdigest()
            assert b'ERROR SUMMARY: 0 errors' in raw
        cases.append({'model': size, 'mode': mode, 'all_40_comparisons_bitwise_equal': True,
                      'serial_workspace_limit_bytes': 64 << 20, 'candidate_workspace_limit_bytes': 128 << 20,
                      'records': 56, 'graph_stats': graph})
summary = {'sources_sha256': sources, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
           'cases': cases, 'all_passed': True, 'independent_vision_quality_accepted': False,
           'batched_vision_root_cause_identified': False,
           'primary_performance_eligible': False,
           'scope': 'Same eight inputs within each model at batch1: serial64MiB versus serial128MiB, '
                    'including repeated ordinary calls and graph replay, plus unfiltered memcheck. '
                    'Workspace alone does not reproduce the failed batch2/4/8 parity on these controls. '
                    'Batch-specific tactics, projection reductions or other causes remain unisolated. '
                    'No batched vision, independent quality, full-request or speed promotion.'}
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'all_passed': True, 'cases': len(cases), 'batching_root_cause_identified': False}))
