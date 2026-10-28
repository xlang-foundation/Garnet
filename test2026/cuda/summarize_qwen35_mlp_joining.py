"""Gate an unwired gate/up projection screen using predeclared conditions."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--preliminary', action='store_true')
args = parser.parse_args()
root = args.root
stem = 'mlp-joined-projection'
source = root / f'{stem}-screen.jsonl'
output = root / f'{stem}-screen-{"preliminary" if args.preliminary else "summary"}.json'
assert not output.exists(), 'Preserve MLP screen summary'
blob = source.read_bytes()
rows = [json.loads(line) for line in blob.decode().splitlines() if line.startswith('{')]
cases = [row for row in rows if row.get('event') == 'joined_projection_case']
assert len(cases) == 16
assert {(row['model'], row['m']) for row in cases} == {
    (size, count) for size in ('2b', '4b') for count in (4, 8, 16, 32, 64, 128, 256, 512)}
assert json.loads((root / f'{stem}-screen-exit.json').read_text())['exit_code'] == 0
results = []
for row in cases:
    intermediate = 6144 if row['model'] == '2b' else 9216
    assert row['projection_group'] == 'mlp' and row['group_widths'] == [intermediate, intermediate]
    assert row['input_bytes_unchanged'] and row['output_guards_intact'] and row['output_splits_bit_exact']
    assert all(part['max_original_fixed_bound_fraction'] <= 1 for part in row['math'])
    times = row['eviction_sweep_graph_median_us']
    results.append({'model': row['model'], 'm': row['m'],
                    'bf16_differences': sum(part['bf16_values_different'] for part in row['math']),
                    'separate_swept_graph_us': times[0], 'joined_only_swept_graph_us': times[1],
                    'joined_split_swept_graph_us': times[2],
                    'joined_split_swept_speed_ratio': times[0] / times[2],
                    'joined_split_warm_speed_ratio': row['warm_graph_median_us'][0] / row['warm_graph_median_us'][2],
                    'max_fixed_bound_fraction': max(part['max_original_fixed_bound_fraction'] for part in row['math'])})
decode = [row for row in results if row['m'] <= 32]
anchors = [row for row in decode if row['m'] in (4, 32)]
assert len(decode) == 8 and len(anchors) == 4
exact = all(row['bf16_differences'] == 0 for row in decode)
benefit = all(row['joined_split_swept_graph_us'] <= .95 * row['separate_swept_graph_us'] for row in anchors)
eligible = exact and benefit
sanitizer_passed = False
if not args.preliminary:
    gate = json.loads((root / f'{stem}-memcheck-exit.json').read_text())
    sanitizer = (root / f'{stem}-memcheck.log').read_text()
    assert gate['exit_code'] == 0 and 'ERROR SUMMARY: 0 errors' in sanitizer and 'LEAK SUMMARY: 0 bytes leaked' in sanitizer
    sanitizer_passed = True
result = {'source': source.name, 'source_sha256': hashlib.sha256(blob).hexdigest(), 'profiles': results,
          'all16_original_fixed_fp64_bounds_guards_input_and_split_checks_passed': True,
          'all8_decode_fixture_profiles_bf16_identical': exact,
          'all4_deployment_anchors_at_least_5_percent_swept_latency_reduction_including_splits': benefit,
          'eligible_for_further_production_validation': eligible and sanitizer_passed,
          'preliminary_component_gate_passed': eligible, 'unfiltered_memcheck_passed': sanitizer_passed,
          'prefill_profiles_changed': [row for row in results if row['m'] > 32 and row['bf16_differences']],
          'scope': 'Synthetic same-production-GEMM fixture, nine rotated graph samples; eviction sweep is not verified L2 '
                   'residency. Decode-width4 candidate only; prefill remains unchanged. Component selection is not model '
                   'correctness, latency, memory, independent quality or2xacceptance. No instrumented timings used.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('all8_decode_fixture_profiles_bf16_identical',
      'all4_deployment_anchors_at_least_5_percent_swept_latency_reduction_including_splits',
      'preliminary_component_gate_passed', 'unfiltered_memcheck_passed')}, indent=2))
