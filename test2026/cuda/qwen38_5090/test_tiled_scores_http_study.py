"""Reject candidate/baseline confusion and retain the full qualified raw checker."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import unittest
from .native_tiled_scores_http_admission import BASELINE, CANDIDATE, MODEL_ARCHIVE
from .tiled_scores_http_study import validate_lifecycle

M = Path(__file__).resolve().parent


def fixture():
    p = {'files': {'synthetic': 'a'}, 'runtime_baseline': {'garnet.dll': BASELINE, 'synthetic_dependency': 'b'},
         'runtime_dependencies': {'garnet.dll': CANDIDATE, 'synthetic_dependency': 'b'},
         'environment': {'synthetic_sdk': 'c'}, 'tiled_model_parent': {'proof_sha256': 'synthetic_proof'}}
    w = {'exit_code': 0, 'cleanup_finished': True, 'source_sha256': p['files'], 'source_after_sha256': p['files'],
         'runtime_baseline': p['runtime_baseline'], 'restored_runtime': p['runtime_baseline'],
         'runtime_dependencies': p['runtime_dependencies'], 'runtime_after_before_restore': p['runtime_dependencies'],
         'library_sha256': CANDIDATE, 'baseline_library_restored': True,
         'environment_before': p['environment'], 'environment_after': p['environment'], 'selector_before': None,
         'selector_after': None, 'performance_admitted': False, 'candidate_library_installed': True,
         'tiled_model_parent': p['tiled_model_parent'], 'admission': {'candidate_library_sha256': CANDIDATE,
             'baseline_library_sha256': BASELINE, 'tiled_scores_enabled': True, 'tiled_model_archive_sha256': MODEL_ARCHIVE,
             'tiled_model_proof_sha256': 'synthetic_proof'}}
    return copy.deepcopy(w), copy.deepcopy(p)


class Controls(unittest.TestCase):
    def test_success_requires_actual_candidate_and_baseline_restoration(self):
        w, p = fixture()
        validate_lifecycle(w, p)
        faults = ('live', 'source', 'private_dependency', 'restoration', 'sdk', 'flag', 'candidate_never_installed',
                  'old_identity_rewritten', 'route_disabled', 'model_proof_drift', 'performance_promotion', 'boolean_exit')
        for fault in faults:
            w, p = fixture()
            if fault == 'live': w['cleanup_finished'] = False
            elif fault == 'source': w['source_after_sha256'] = {}
            elif fault == 'private_dependency': w['runtime_after_before_restore']['synthetic_dependency'] = 'drift'
            elif fault == 'restoration': w['restored_runtime']['garnet.dll'] = CANDIDATE
            elif fault == 'sdk': w['environment_after'] = {}
            elif fault == 'flag': w['selector_after'] = '1'
            elif fault == 'candidate_never_installed': w['candidate_library_installed'] = False
            elif fault == 'old_identity_rewritten': w['admission']['baseline_library_sha256'] = CANDIDATE
            elif fault == 'route_disabled': w['admission']['tiled_scores_enabled'] = False
            elif fault == 'model_proof_drift': w['admission']['tiled_model_proof_sha256'] = 'drift'
            elif fault == 'boolean_exit': w['exit_code'] = False
            else: w['performance_admitted'] = True
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                validate_lifecycle(w, p)

    def test_failure_before_installation_preserved_without_success_admission(self):
        w, p = fixture()
        w.update(exit_code=1, candidate_library_installed=False, admission=None,
                 runtime_after_before_restore=copy.deepcopy(p['runtime_baseline']))
        validate_lifecycle(w, p)

    def test_entire_checker_is_exact_recorded_derivation(self):
        row = json.loads((M / 'evidence/tiled-scores-http-verifier-derivation.json').read_bytes())
        raw = (M / row['source']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_sha256'])
        text = raw.decode().replace('\r\n', '\n')
        for before, after, count in row['changes']:
            self.assertEqual(text.count(before), count)
            text = text.replace(before, after)
        actual = (M / row['result']).read_bytes()
        self.assertEqual(text.encode(), actual)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), row['result_sha256'])

    def test_all_raw_benchmark_and_native_request_arithmetic_is_unchanged(self):
        for name in ('benchmark_replay', 'native_match'):
            def function(filename):
                return next(n for n in ast.walk(ast.parse((M / filename).read_bytes()))
                            if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(function('verify_prefill16_http.py'), include_attributes=False),
                             ast.dump(function('verify_tiled_scores_http.py'), include_attributes=False))


if __name__ == '__main__':
    unittest.main()
