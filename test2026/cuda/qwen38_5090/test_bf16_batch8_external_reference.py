"""Synthetic fresh-reference comparison controls, without model execution."""
from pathlib import Path
import ast
import copy
import hashlib
import struct
import unittest
from .test_bf16_batch8_replay import fixture
from .bf16_batch8_external_reference import case_exact, replay_reference


def synthetic_reference(baseline, original_files):
    """Only fixture conversion; these F32 metadata fields are not a GPU result."""
    files = {}
    def state(original, rows, stem):
        logical = copy.deepcopy(original['logical'])
        for name in ('slots', 'physical_row_sha256', 'logical_row_sha256'):
            logical[name] = [logical[name][row] for row in rows]
        for component in logical['components']:
            component['row_sha256'] = [component['row_sha256'][row] for row in rows]
        logical.update(physical_words_per_row=173932544, reference_bf16_roundtrip_checked=True,
                       physical_row_sha256=list(logical['logical_row_sha256']))
        vectors = {}
        for kind in ('hidden', 'normalized'):
            raw = original_files[original['vectors'][kind]['file']]
            selected = b''.join(raw[row*20480:(row+1)*20480] for row in rows)
            name = stem+'-'+kind+'.f32';files[name] = selected
            vectors[kind] = {'file': name, 'bytes': len(selected), 'sha256': hashlib.sha256(selected).hexdigest()}
        return {'logical': logical, 'vectors': vectors}
    report = {'status': 'bf16_batch8_external_reference_collected_needs_replay', 'physical_batch': 4, 'capacity': 4096,
              'prefill_chunk': 32, 'decode_chunk': 4, 'bf16_kv_float_reference': True,
              'compact_commit_used': False, 'mtp_executed': False, 'http_executed': False, 'performance_admitted': False,
              'library_sha256': baseline['library_sha256'], 'empty_state': {}, 'final_reset': {}, 'cases': []}
    for half in (0, 1):
        rows = list(range(half*4, half*4+4))
        for name in ('empty_state', 'final_reset'):
            report[name][str(half)] = state(baseline[name], rows, name+'-half'+str(half))
    for index, original in enumerate(baseline['cases']):
        for half in (0, 1):
            rows = list(range(half*4, half*4+4));stem = 'case'+str(index)+'-half'+str(half)
            case = {'baseline_case_index': index, 'half': half, 'physical_rows': rows, 'all_reference_rows_exact': True,
                    'performance_admitted': False, **{name: copy.deepcopy(original[name]) for name in ('group', 'repeat', 'order', 'actual_limit')}}
            for name in ('inputs', 'generated_ids', 'text'):
                case[name] = [copy.deepcopy(original[name][i]) for i in rows]
            for name in ('reset_state', 'prefill_state', 'final_state'):
                case[name] = state(original[name], rows, stem+'-'+name)
            lengths = [len(row['prompt_ids']) for row in case['inputs']];n = max(lengths)
            case['prompt_cache'] = {'physical_batch': 4, 'tokens': n, 'lengths': lengths, 'page_size': 128,
                                   'pack_calls': (n+127)//128, 'cached_pages': [0],
                                   'peak_cached_packed_logical_bytes': min(n, 256)*4*20480}
            report['cases'].append(case)
    return report, files


def candidate():
    baseline, raw, _ = fixture();baseline['library_sha256'] = 'synthetic_same_library'
    report, files = synthetic_reference(baseline, raw)
    return report, files, baseline, raw


class Controls(unittest.TestCase):
    def test_all18_same_input_reference_cases_and58captures_are_required(self):
        report, files, baseline, raw = candidate()
        result = replay_reference(report, files.__getitem__, baseline, raw.__getitem__)
        self.assertEqual((result['fresh_reference_cases'], result['reference_complete_state_captures'], result['reference_raw_vectors']), (18, 58, 116))
        self.assertFalse(result['b1_math_qualified']);self.assertFalse(result['performance_admitted'])

    def test_changed_input_ids_outputs_component_global_and_raw_vector_rejected(self):
        for fault in ('input', 'output', 'component', 'global', 'vector', 'missing'):
            report, files, baseline, raw = candidate();case = report['cases'][-1]
            if fault == 'input': case['inputs'][3]['prompt_ids'][4] += 1
            elif fault == 'output': case['generated_ids'][3][-1] += 1
            elif fault == 'component': case['final_state']['logical']['components'][-1]['row_sha256'][3] = 'f'*64
            elif fault == 'global':
                case['final_state']['logical']['logical_row_sha256'][3] = 'f'*64
                case['final_state']['logical']['physical_row_sha256'][3] = 'f'*64
            elif fault == 'vector':
                entry = case['final_state']['vectors']['normalized'];changed = files[entry['file']][:-4]+struct.pack('<f', 99.)
                files[entry['file']] = changed;entry['sha256'] = hashlib.sha256(changed).hexdigest()
            else: report['cases'].pop()
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                replay_reference(report, files.__getitem__, baseline, raw.__getitem__)

    def test_changed_row_map_boolean_indices_and_compact_reference_cannot_pass(self):
        for fault in ('row_map', 'bool_case', 'bool_row', 'compact', 'roundtrip', 'bool_cache'):
            report, files, baseline, raw = candidate();case = report['cases'][0]
            if fault == 'row_map': case['physical_rows'] = [4, 5, 6, 7]
            elif fault == 'bool_case': case['baseline_case_index'] = False
            elif fault == 'bool_row': case['physical_rows'][0] = False
            elif fault == 'compact': case['prefill_state']['logical']['physical_words_per_row'] = 106823680
            elif fault == 'roundtrip': case['prefill_state']['logical']['reference_bf16_roundtrip_checked'] = False
            else: case['prompt_cache']['pack_calls'] = True
            with self.subTest(fault=fault), self.assertRaises((ValueError, AssertionError)):
                case_exact(case, baseline['cases'][0], 0, 0, files.__getitem__, raw.__getitem__)

    def test_probe_selects_existing_full_cache_b4_guard_and_no_compact_commit(self):
        source = Path(__file__).with_name('probe_bf16_batch8_external_reference.py').read_bytes()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'CUDAPackedDecoder']
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual([ast.literal_eval(n) for n in call.args[2:]], [4, 4096, 32, 4])
        options = {n.arg: ast.literal_eval(n.value) for n in call.keywords}
        self.assertIs(options['bf16_kv_float_reference'], True)
        self.assertNotIn('bf16_kv_state', options)
        self.assertTrue(all(options[k] is True for k in ('retain_linear_inputs', 'deferred_state_commit', 'host_embedding_lookup', 'shared_output_storage')))
        witnesses = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'logical_state_witness']
        self.assertEqual(len(witnesses), 1)
        self.assertIs(next(ast.literal_eval(n.value) for n in witnesses[0].keywords if n.arg == 'packed_bf16'), False)
        self.assertIn(b'decoder.delta_commit is decoder.bf16_prefill_commit is None', source)


if __name__ == '__main__':
    unittest.main()
