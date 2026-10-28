"""Synthetic CPU source/closure fixtures, never profiler or model measurements."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest
import zipfile

from .native_cuda_phase_study import TRIAL, closed_trial
from .run_native_cuda_phases import test_ids

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-cuda-study-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'results' / TRIAL).mkdir(parents=True)

    def fixture(self, code=0, mutate=None):
        w = {'exit_code': code, 'cleanup_finished': True, 'source_sha256': {},
             'source_after_sha256': {}, 'runtime_dependencies': {},
             'runtime_dependencies_after': {}, 'library_unchanged': True}
        if mutate:
            w.update(mutate)
        raw = json.dumps(w).encode()
        (self.root / 'results' / TRIAL / 'workflow.json').write_bytes(raw)
        archive = self.root / 'results' / (TRIAL + '-evidence.zip')
        with zipfile.ZipFile(archive, 'w') as z:
            name = 'results/' + TRIAL + '/workflow.json'
            z.writestr(name, raw)
            z.writestr('evidence-manifest.json', json.dumps({'files': {name: hashlib.sha256(raw).hexdigest()}}))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        proof = {'status': 'prefill16_fixed4k_http_replayed' if code == 0 else 'prefill16_fixed4k_http_failure_preserved',
                 'archive_sha256': digest}
        (self.root / 'results' / (TRIAL + '-verified.json')).write_text(json.dumps(proof))
        return digest

    def test_actual_terminal_pass_or_preserved_failure_required_for_sequencing(self):
        for code in (0, 1):
            result = closed_trial(self.root, self.fixture(code))
            self.assertEqual(result['exit_code'], code)
            self.assertTrue(result['cleanup_finished'])

    def test_running_cleanup_drift_bool_exit_and_wrong_proof_rejected(self):
        for fault in ({'exit_code': None}, {'exit_code': False}, {'cleanup_finished': False},
                      {'source_after_sha256': {'changed': 'bad'}}, {'runtime_dependencies_after': {'changed': 'bad'}},
                      {'library_unchanged': False}):
            with self.subTest(fault=fault):
                digest = self.fixture(mutate=fault)
                with self.assertRaises(AssertionError):
                    closed_trial(self.root, digest)
        digest = self.fixture()
        path = self.root / 'results' / (TRIAL + '-verified.json')
        p = json.loads(path.read_bytes())
        p['status'] = 'unverified'
        path.write_text(json.dumps(p))
        with self.assertRaises(AssertionError):
            closed_trial(self.root, digest)
        for digest in ('', 'A' * 64, 'x' * 64):
            with self.assertRaises(ValueError):
                closed_trial(self.root, digest)

    def test_all_recorded_derivations_reproduce_complete_worker_launcher_and_probe(self):
        derivations = json.loads((M / 'evidence/native-cuda-phase-worker-derivation.json').read_bytes())
        probe = json.loads((M / 'evidence/native-cuda-phase-probe-derivation.json').read_bytes())
        derivations['probe_native_cuda_phases.py'] = probe
        for target, d in derivations.items():
            raw = (M / d['source']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), d['source_sha256'])
            text = raw.decode().replace('\r\n', '\n')
            for old, new in d['replacements']:
                self.assertEqual(text.count(old), 1)
                text = text.replace(old, new)
            raw = text.encode()
            self.assertEqual(raw, (M / target).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), d['candidate_sha256'])

    def test_test_id_flattening_binds_real_test_objects_and_cpu_result_faults(self):
        suite = unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_native_cuda_phases')
        names = test_ids(suite)
        self.assertEqual(len(names), suite.countTestCases())
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith('test2026.cuda.qwen38_5090.test_native_cuda_phases.Controls.test_') for name in names))


if __name__ == '__main__':
    unittest.main()
