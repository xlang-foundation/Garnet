"""Audit standalone BF16 codec/projection evidence; not inference acceptance."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
stem = 'lossless-bf16-projection-' + (sys.argv[2] if len(sys.argv) > 2 else 'v1')
output = root / (stem + '-summary.json')
assert not output.exists(), 'Preserve component summary'
sources = {}


def read(name, lines=False):
    raw = (root / name).read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8-sig')
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


identity = read(stem + '-before-build-identity.json')
assert read(stem + '-build-exit.json')['exit_code'] == 0
design = read('lossless-bf16-projection-v1-prospective-design.json')
fixture = read('lossless-bf16-projection-v1-fixtures.json')
assert read('lossless-bf16-codec-v1-tests.json')['all_passed']
fixture_names = {r['name'] for r in fixture['fixtures']}
matrix_names = {r['name'] for r in fixture['fixtures'] if r['inner']}
binary, cases, plain = set(), [], None
for mode in ('plain', 'memcheck', 'racecheck', 'synccheck'):
    tag = stem + '-' + mode
    terminal = read(tag + '-exit.json')
    assert terminal['exit_code'] == 0
    assert terminal['source_identity_sha256'].lower() == sources[stem + '-before-build-identity.json']
    binary.add(terminal['binary_sha256'].lower())
    data = read(tag + '.jsonl', lines=True)
    assert data[0]['event'] == 'metadata' and data[-1]['event'] == 'complete' and data[-1]['all_passed']
    assert data[0]['gpu_uuid'] == identity['gpu_uuid']
    assert data[0]['uploads_ordered_on_test_stream']
    codec = [r for r in data if r['event'] == 'codec']
    assert {(r['fixture'], r['streaming_reads']) for r in codec} == {(n, s) for n in fixture_names for s in (False, True)}
    assert len(codec) == len(fixture_names) * 2 and all(r['different_words'] == 0 and r['guards_passed'] for r in codec)
    projections = [r for r in data if r['event'] == 'projection']
    shapes = (4, 8, 16, 32, 64, 128, 256, 512) if mode == 'plain' else (4, 32, 512) if mode == 'memcheck' else ()
    expected = {(name, rows, variant) for name in matrix_names for rows in shapes for variant in ('ordinary', 'stage', 'stage_streaming')}
    assert len(projections) == len(expected) and {(r['fixture'], r['rows'], r['variant']) for r in projections} == expected
    assert data[-1]['codec_cases'] == len(codec) and data[-1]['projection_profiles'] == len(matrix_names) * len(shapes)
    assert all(r['all_output_words_bit_exact'] and r['guards_passed'] for r in projections)
    for record in projections:
        assert len(record['cold_samples_us']) == len(record['warm_samples_us']) == (31 if mode == 'plain' else 3)
        assert abs(statistics.median(record['cold_samples_us']) - record['cold_median_us']) < 1e-6
    log = (root / (tag + '.log')).read_bytes()
    sources[tag + '.log'] = hashlib.sha256(log).hexdigest()
    if mode == 'racecheck':
        assert b'RACECHECK SUMMARY: 0 hazards' in log
    elif mode != 'plain':
        assert b'ERROR SUMMARY: 0 errors' in log
    cases.append({'mode': mode, 'codec_cases': len(codec), 'projection_profiles': len(matrix_names) * len(shapes), 'passed': True})
    if mode == 'plain':
        plain = {(r['fixture'], r['rows'], r['variant']): r for r in projections}
assert len(binary) == 1
scores = []
for variant in ('stage', 'stage_streaming'):
    models = []
    for size, config in design['decode_screen_weights'].items():
        ordinary = candidate = 0.0
        no_regressions = True
        for family, count in config['operator_counts'].items():
            left = plain[(size + '-' + family, config['rows'], 'ordinary')]
            right = plain[(size + '-' + family, config['rows'], variant)]
            ordinary += count * left['cold_median_us']
            candidate += count * right['cold_median_us']
            no_regressions &= right['cold_median_us'] <= left['cold_median_us']
        ratio = ordinary / candidate
        models.append({'model': size, 'rows': config['rows'], 'ordinary_weighted_us': ordinary,
                       'candidate_weighted_us': candidate, 'ratio': ratio, 'no_selected_operator_regression': no_regressions,
                       'passes_screen': ratio >= design['minimum_cold_decode_ratio'] and no_regressions})
    scores.append({'variant': variant, 'models': models, 'eligible_for_broader_component_screen': all(m['passes_screen'] for m in models)})
result = {'sources_sha256': sources, 'binary_sha256': next(iter(binary)), 'all_correctness_cases_passed': True,
    'cases': cases, 'decode_subset_scores': scores,
    'all_plain_profiles': list(plain.values()), 'production_integration_accepted': False,
    'model_memory_measured': False, 'inference_speed_measured': False,
    'scope': 'CPU golden/all-bit/malformed codec checks and canonical-fixture GPU bit/guard tests, '
        'production GEMM source reused with exact BF16 outputs. Plain32 matrix/row profiles and '
        'memcheck12; race/sync cover only all10 decoder variants. Cold timings exclude artificial '
        'eviction, whose cost/total are also retained; warm timings are separate. Selected joined '
        'and down operators only; joined wider-row cases are component shapes, not production '
        'prefill execution. Passing this subset only permits broader screening, not model '
        'integration or2x acceptance. Original and encoded weights coexist for validation; '
        'a single staged matrix needs additional full-size temporary storage. No actual '
        'whole-model ownership/cache/VRAM/residency/performance claim.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_correctness_cases_passed': True, 'decode_subset_scores': scores}, indent=2))
