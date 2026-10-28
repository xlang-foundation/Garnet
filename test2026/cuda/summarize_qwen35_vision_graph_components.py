"""Admit completed plain/memcheck vision replay controls, without speed claims."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'vision-graph-component-summary.json'
assert not output.exists(), 'Preserve vision component summary'
repo = Path('D:/CantorAI/Garnet')
epoch_path = a.root / 'vision-graph-component-v1-before-run-identity.json'
identity = json.loads(epoch_path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for name in ('runtime/cuda_serving.py', 'xModel/qwen35/vl_0_8b/cuda_decoder.py',
             'test2026/cuda/probe_qwen35_vision_graph.py'):
    assert digest(repo / name) == identity['source_sha256'][name], 'Consumed vision source changed'
native = digest(a.root / 'build-vs18/bin/garnet.dll')
assert native == identity['garnet_binary_sha256']
sources = [{'file': epoch_path.name, 'sha256': digest(epoch_path)}]
cases = []
for mode in ('plain', 'memcheck'):
    for size in ('2b', '4b'):
        tag = f'vision-graph-component-v1-{size}-{mode}'
        data = json.loads((a.root / (tag + '.json')).read_text(encoding='utf-8'))
        terminal = json.loads((a.root / (tag + '-exit.json')).read_text(encoding='utf-8'))
        log = (a.root / (tag + '.log')).read_text(encoding='utf-8-sig')
        assert terminal['exit_code'] == 0 and data['all_passed']
        assert len(data['records']) == data['cases'] == 32 and data['unique_images'] == 8
        assert data['same_compiled_engine_for_both_modes'] and data['additional_pending_close_execution']
        assert data['output_survives_close'] and data['closed_and_double_enable_rejected']
        assert data['extra_owned_staging_bytes'] == 1474560
        stats = data['graph_stats']
        assert stats['contexts'] == stats['graph_captures'] == 1
        assert stats['graph_replays'] > 0 and stats['graph_capture_failures'] == 0
        if mode == 'memcheck':
            assert 'ERROR SUMMARY: 0 errors' in log
        for ending in ('.json', '.log', '-exit.json'):
            path = a.root / (tag + ending)
            sources.append({'file': path.name, 'sha256': digest(path)})
        cases.append({'model': data['model'], 'mode': mode, 'cases': 32,
                      'additional_pending_close_execution': True,
                      'captures': 1, 'contexts': 1, 'replays': stats['graph_replays'],
                      'extra_owned_staging_bytes': 1474560})
result = {'all_passed': True, 'garnet_dll_sha256': native, 'cases': cases, 'sources': sources,
          'scope': 'Both actual models plain and unfiltered memcheck; same-engine16 ordinary/16 replay '
                   'finite visual hashes over8 unique images, source/staging/one-context/one-capture and '
                   'additional pending-close output lifetime controls. Does not establish independent '
                   'vision quality, complete request state preservation, latency or process memory.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_passed': True, 'models_modes': len(cases)}, indent=2))
