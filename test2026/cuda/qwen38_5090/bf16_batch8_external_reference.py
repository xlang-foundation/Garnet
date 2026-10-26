"""Strict same-input B8-vs-fresh-B4 full-cache reference comparison; no GPU imports."""
import hashlib
import re
from .bf16_batch8_reference_gate import same_reference_state, same_reference_vectors
from .verify_bf16_batch8_admission import replay_cases


def capture_exact(actual_b4, original_b8, rows, read_b4, read_b8):
    assert same_reference_state(original_b8['logical'], actual_b4['logical'], rows)
    assert set(actual_b4['vectors']) == set(original_b8['vectors']) == {'hidden', 'normalized'}
    for kind in ('hidden', 'normalized'):
        raw = []
        for entry, read, size in ((actual_b4['vectors'][kind], read_b4, 81920),
                                  (original_b8['vectors'][kind], read_b8, 163840)):
            assert isinstance(entry['file'], str) and re.fullmatch(r'[A-Za-z0-9_.-]+\.f32', entry['file'])
            data = read(entry['file'])
            assert type(entry['bytes']) is int and entry['bytes'] == len(data) == size
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            raw.append(data)
        assert same_reference_vectors(raw[1], raw[0], rows)


def case_exact(actual, original, case_index, half, read_b4, read_b8):
    assert type(case_index) is int and 0 <= case_index < 9 and type(half) is int and half in (0, 1)
    rows = list(range(half*4, half*4+4))
    assert type(actual['baseline_case_index']) is type(actual['half']) is int
    assert isinstance(actual['physical_rows'], list) and all(type(v) is int for v in actual['physical_rows'])
    assert actual['baseline_case_index'] == case_index and actual['half'] == half and actual['physical_rows'] == rows
    assert type(actual['repeat']) is int and isinstance(actual['order'], list) and all(type(v) is int for v in actual['order'])
    for name in ('group', 'repeat', 'order', 'actual_limit'):
        assert actual[name] == original[name], name
    for name in ('inputs', 'generated_ids', 'text'):
        assert actual[name] == [original[name][i] for i in rows], name
    for name in ('reset_state', 'prefill_state', 'final_state'):
        capture_exact(actual[name], original[name], rows, read_b4, read_b8)
    stats = actual['prompt_cache']
    assert all(type(stats[key]) is int for key in ('physical_batch', 'tokens', 'page_size', 'pack_calls', 'peak_cached_packed_logical_bytes'))
    lengths = [len(row['prompt_ids']) for row in actual['inputs']]
    count = max(lengths)
    assert stats['physical_batch'] == 4 and stats['tokens'] == count and stats['lengths'] == lengths
    assert stats['page_size'] == 128 and stats['pack_calls'] == (count+127)//128
    assert len(stats['cached_pages']) <= 2 and 0 < stats['peak_cached_packed_logical_bytes'] <= 20971520


def replay_reference(report, read_b4, baseline, read_b8):
    assert report['status'] == 'bf16_batch8_external_reference_collected_needs_replay'
    assert tuple(report[key] for key in ('physical_batch', 'capacity', 'prefill_chunk', 'decode_chunk')) == (4, 4096, 32, 4)
    assert report['bf16_kv_float_reference'] is True and report['library_sha256'] == baseline['library_sha256']
    assert report['compact_commit_used'] is report['mtp_executed'] is report['http_executed'] is report['performance_admitted'] is False
    assert len(baseline['cases']) == 9 and len(report['cases']) == 18
    groups = [(label, baseline['cases'][i*3]['inputs'], limit) for i, (label, limit) in enumerate(
        (('short-mixed', 8), ('full4k-text', 128), ('full4k-image', 128)))]
    replay_cases(baseline, read_b8, groups)
    for label in ('empty_state', 'final_reset'):
        assert set(report[label]) == {'0', '1'}
        for half in (0, 1):
            capture_exact(report[label][str(half)], baseline[label], list(range(half*4, half*4+4)), read_b4, read_b8)
    for index, actual in enumerate(report['cases']):
        case_exact(actual, baseline['cases'][index//2], index//2, index%2, read_b4, read_b8)
        assert actual['all_reference_rows_exact'] is True and actual['performance_admitted'] is False
    return {'status': 'bf16_batch8_external_b4_cache_reference_exact', 'baseline_cases': 9, 'fresh_reference_cases': 18,
            'reference_complete_state_captures': 58, 'reference_raw_vectors': 116, 'all_output_ids_states_and_raw_vectors_exact': True,
            'b8_vs_b4_same_input_cache_arithmetic_cases_qualified': True, 'b1_math_qualified': False,
            'broad_quality_qualified': False, 'mtp_qualified': False, 'http_qualified': False,
            'performance_admitted': False, 'two_times_target_achieved': False,
            'scope': 'Only these actual same-input B8 packed-cache vs B4 full-FP32-cache/BF16-value ordinary cases. Shared target projections/linear operators remain shared; not an independent framework/official quality or all-model mathematical proof.'}
