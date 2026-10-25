"""CPU-only faults preserve the full ordinary B8 oracle and closed-parent gate."""
from pathlib import Path
import copy
import hashlib
import json
import struct
import unittest
from .test_paged_batch8_model import candidate
from .paged_batch8_mtp import BASELINE, replay_mtp, validate_parent

M = Path(__file__).resolve().parent


def synthetic_mtp():
    actual, reference, files = candidate()
    actual['library_sha256'] = reference['library_sha256'] = BASELINE
    actual.update(status='paged_batch8_mtp_collected_needs_replay', mtp_executed=True, mtp_depth=3, draft_chunk=4,
                  draft_builder_workspace_bytes=128 << 20, mtp_stop_ids=[], prefix_replay=True)
    for case in actual['cases']:
        limit = case['actual_limit']
        remaining = limit-1
        processed = 0
        steps = []
        while remaining:
            count = min(4, remaining)
            processed += count
            remaining -= count
            steps.append({'processed_target_counts': [count]*8, 'sampled_so_far': [processed]*8})
        case['mtp_result'] = {'generated_token_ids': copy.deepcopy(case['generated_ids']), 'sampled_tokens_including_stop': [limit]*8,
                              'finish_reasons': ['length']*8, 'stop_token_ids': [None]*8, 'decode_steps': limit-1,
                              'speculative_metrics': {'iterations': steps, 'target_verification_calls': len(steps)}}
        case['mtp_memory'] = {'draft_builder_workspace_bytes': 128 << 20, 'draft_chunk': 4, 'target_prefill_chunk': 32}
    return actual, reference, files


class Controls(unittest.TestCase):
    def test_all_nine_oracle_cases_and29states58vectors_are_required(self):
        actual, reference, files = synthetic_mtp()
        result = replay_mtp(actual, files.__getitem__, reference, files.__getitem__)
        self.assertEqual((result['cases'], result['complete_states'], result['raw_vectors']), (9, 29, 58))
        self.assertFalse(result['performance_admitted'])

    def test_final_state_vector_ids_and_counts_drift_rejected(self):
        for fault in ('component', 'finite_vector', 'ids', 'counts', 'builder', 'stop', 'short_context'):
            actual, reference, files = synthetic_mtp()
            changed = dict(files)
            case = actual['cases'][-1]
            if fault == 'component': case['final_state']['logical']['components'][-1]['row_sha256'][7] = 'f'*64
            elif fault == 'finite_vector':
                entry = case['final_state']['vectors']['hidden'];raw = struct.pack('<f', 42.)*40960
                changed[entry['file']] = raw;entry['sha256'] = hashlib.sha256(raw).hexdigest()
            elif fault == 'ids': case['generated_ids'][7][-1] += 1
            elif fault == 'counts': case['mtp_result']['speculative_metrics']['iterations'][-1]['processed_target_counts'][7] = 0
            elif fault == 'builder': actual['draft_builder_workspace_bytes'] = 256 << 20
            elif fault == 'stop': actual['mtp_stop_ids'] = [248044]
            else: case['inputs'] = copy.deepcopy(case['inputs']);case['inputs'][7]['prompt_ids'].pop()
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                replay_mtp(actual, changed.__getitem__, reference, files.__getitem__)

    def test_closed_parent_cannot_be_live_changed_or_promoted(self):
        for fault in ('valid', 'live', 'source', 'library', 'sdk', 'missing_states', 'mtp_claim'):
            p = {'library_sha256': BASELINE, 'status': 'paged_batch8_model_preservation_replayed', 'archive_sha256': 'synthetic_digest',
                 'preserved_model': {'complete_states': 29, 'raw_vectors': 58, 'all_retained_states_vectors_tokens_exact': True},
                 'external_b1_b4_math_qualified': False, 'mtp_qualified': False, 'http_qualified': False, 'performance_admitted': False, 'two_times_target_achieved': False}
            w = {'library_sha256': BASELINE, 'exit_code': 0, 'cleanup_finished': True, 'library_unchanged': True,
                 'source_sha256': {'synthetic': 'a'}, 'source_after_sha256': {'synthetic': 'a'},
                 'runtime_dependencies': {'garnet.dll': BASELINE}, 'runtime_dependencies_after': {'garnet.dll': BASELINE},
                 'environment_before': {'synthetic_sdk': 'b'}, 'environment_after': {'synthetic_sdk': 'b'}}
            if fault == 'live': w['cleanup_finished'] = False
            elif fault == 'source': w['source_after_sha256'] = {}
            elif fault == 'library': p['library_sha256'] = 'drift'
            elif fault == 'sdk': w['environment_after'] = {}
            elif fault == 'missing_states': p['preserved_model']['complete_states'] = 28
            elif fault == 'mtp_claim': p['mtp_qualified'] = True
            if fault == 'valid': validate_parent(p, w, BASELINE, 'synthetic_digest')
            else:
                with self.subTest(fault=fault), self.assertRaises(AssertionError): validate_parent(p, w, BASELINE, 'synthetic_digest')

    def test_actual_probe_and_owner_are_exact_recorded_derivations(self):
        for name in ('probe', 'owner'):
            row = json.loads((M / ('evidence/paged-batch8-mtp-' + name + '-derivation.json')).read_bytes())
            raw = (M / row['source']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_sha256'])
            text = raw.decode().replace('\r\n', '\n')
            for before, after, count in row['changes']:
                self.assertEqual(text.count(before), count)
                text = text.replace(before, after)
            actual = (M / row['result']).read_bytes()
            self.assertEqual(text.encode(), actual)
            self.assertEqual(hashlib.sha256(actual).hexdigest(), row['result_sha256'])
        bat = (M / 'test_paged_batch8_mtp.bat').read_bytes()
        self.assertEqual(bat.replace(b'probe_paged_batch8_mtp.py', b'probe_paged_batch8_model.py'), (M / 'test_paged_batch8_model.bat').read_bytes())


if __name__ == '__main__':
    unittest.main()
