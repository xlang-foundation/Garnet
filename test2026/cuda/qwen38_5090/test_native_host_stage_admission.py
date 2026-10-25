"""Synthetic CPU-only complete trace replay and exclusive-reference release controls."""
from pathlib import Path
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout

from .native_host_stage_admission import closed_observed_reference, replay_report
from . import test_native_host_stage_probe as probe_controls


def reference_fixture(root, **changes):
    name = 'sglang-observed-fixed4k-ordinary-attempt1'
    folder = root / 'results' / name
    folder.mkdir(parents=True)
    workflow = {'exit_code': 0, 'cleanup_finished': True, 'server_exit_code': 0,
                'source_sha256': {'source': 'hash'}, 'source_after_sha256': {'source': 'hash'},
                'runtime_dependencies': {'runtime': 'hash'}, 'runtime_dependencies_after': {'runtime': 'hash'}}
    workflow.update(changes)
    raw = json.dumps(workflow).encode()
    (folder / 'workflow.json').write_bytes(raw)
    path = 'results/' + name + '/workflow.json'
    archive = root / 'results' / (name + '-evidence.zip')
    with zipfile.ZipFile(archive, 'x') as z:
        z.writestr(path, raw)
        z.writestr('evidence-manifest.json', json.dumps({'files': {path: hashlib.sha256(raw).hexdigest()}}))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    status = ('reference_fixed4k_observed_profile_replayed' if workflow['exit_code'] == 0
              else 'reference_fixed4k_observed_failure_preserved')
    (root / 'results' / (name + '-verified.json')).write_text(json.dumps({'archive_sha256': digest, 'status': status}))
    return name, digest


class AdmissionControls(unittest.TestCase):
    def test_actual24_fixture_events_recomputed_not_saved_summary(self):
        fixture, report, cases, read = self.fixture()
        fresh = replay_report(report, read, cases)
        self.assertEqual(fresh['requests'], 24)
        self.assertFalse(fresh['performance_admitted'])
        self.assertFalse(fresh['isolated_gpu_or_kernel_time_measured'])
        self.assertTrue(all(child.closed for child in fixture.created))

    def fixture(self):
        fixture = probe_controls.ProbeControls('test_full24_synthetic_requests_and_both_owned_children_close')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with redirect_stdout(io.StringIO()):
            _, report = fixture.run_probe('synthetic-admission')
        folder = fixture.root / 'results/synthetic-admission'
        cases = json.loads((fixture.root / 'Garnet/test2026/cuda/qwen38_5090/evidence/native-host-stage-cases.json').read_bytes())
        return fixture, report, cases, lambda name: (folder / name).read_bytes()

    def test_changed_retained_ids_order_saved_arithmetic_or_worker_exit_rejected(self):
        _, report, cases, read = self.fixture()
        candidates = []
        a = copy.deepcopy(report); a['modes']['mtp']['requests'][0]['repeat'] = 1; candidates.append((a, cases))
        a = copy.deepcopy(report); a['modes']['ordinary']['worker_exit_code'] = True; candidates.append((a, cases))
        a = copy.deepcopy(report); a['modes']['mtp']['replay']['requests'][0]['host_wall_seconds'] += 1; candidates.append((a, cases))
        a = copy.deepcopy(cases); a['cases'][0]['generated_ids_by_mode']['mtp'][-1] = 999; candidates.append((report, a))
        a = copy.deepcopy(cases); a['orders'][0].reverse(); candidates.append((report, a))
        for current, controls in candidates:
            with self.assertRaises(AssertionError):
                replay_report(current, read, controls)

    def test_raw_stage_mutation_rejected_even_with_saved_summary_unchanged(self):
        _, report, cases, read = self.fixture()
        name = 'ordinary/host-stages.jsonl'
        rows = [json.loads(line) for line in read(name).splitlines()]
        rows[0]['first_token_observed'] = rows[0]['started']
        altered = b'\n'.join(json.dumps(row).encode() for row in rows)
        with self.assertRaises(AssertionError):
            replay_report(report, lambda path: altered if path == name else read(path), cases)

    def test_closed_success_and_preserved_failure_release_diagnostic_only(self):
        for code in (0, 1):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); name, digest = reference_fixture(root, exit_code=code)
                self.assertEqual(closed_observed_reference(root, name, digest)['exit_code'], code)

    def test_live_wrong_source_runtime_or_boolean_exit_is_rejected(self):
        for changes in ({'cleanup_finished': False}, {'exit_code': None}, {'exit_code': True},
                        {'server_exit_code': None}, {'server_exit_code': True},
                        {'source_after_sha256': {}}, {'runtime_dependencies_after': {}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); name, digest = reference_fixture(root, **changes)
                with self.assertRaises(AssertionError):
                    closed_observed_reference(root, name, digest)

    def test_wrong_namespace_hash_proof_and_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name, digest = reference_fixture(root)
            for wrong_name, wrong_digest in (('../other', digest), (name, digest + '0')):
                with self.assertRaises(ValueError):
                    closed_observed_reference(root, wrong_name, wrong_digest)
            with self.assertRaises(AssertionError):
                closed_observed_reference(root, name, '0' * 64)
            path = root / 'results' / (name + '-verified.json')
            p = json.loads(path.read_bytes()); p['status'] = 'reference_fixed4k_observed_failure_preserved'
            path.write_text(json.dumps(p))
            with self.assertRaises(AssertionError):
                closed_observed_reference(root, name, digest)


if __name__ == '__main__':
    unittest.main()
