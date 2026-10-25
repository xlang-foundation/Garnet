"""Synthetic lifecycle/actual-exit guards; no model runtime or remote access."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
import zipfile

from .prefill16_http_study import closed_host_diagnostic, finish_worker


def fixture(root, **changes):
    name = 'native-host-stage-full4k-attempt1'
    folder = root / 'results' / name
    folder.mkdir(parents=True)
    w = {'exit_code': 0, 'cleanup_finished': True, 'library_unchanged': True,
         'source_sha256': {'s': 'hash'}, 'source_after_sha256': {'s': 'hash'},
         'runtime_dependencies': {'r': 'hash'}, 'runtime_dependencies_after': {'r': 'hash'}}
    w.update(changes)
    raw = json.dumps(w).encode()
    (folder / 'workflow.json').write_bytes(raw)
    path = 'results/' + name + '/workflow.json'
    archive = root / 'results' / (name + '-evidence.zip')
    with zipfile.ZipFile(archive, 'x') as z:
        z.writestr(path, raw)
        z.writestr('evidence-manifest.json', json.dumps({'files': {path: hashlib.sha256(raw).hexdigest()}}))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    status = 'native_host_stage_full4k_replayed' if w['exit_code'] == 0 else 'native_host_stage_full4k_failure_preserved'
    (root / 'results' / (name + '-verified.json')).write_text(json.dumps({'status': status, 'archive_sha256': digest}))
    return digest


class Controls(unittest.TestCase):
    def test_actual_closed_success_or_failure_only_releases_next_job(self):
        for code in (0, 1):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); digest = fixture(root, exit_code=code)
                self.assertEqual(closed_host_diagnostic(root, digest)['exit_code'], code)

    def test_live_source_runtime_library_or_boolean_exit_is_rejected(self):
        for changes in ({'cleanup_finished': False}, {'exit_code': None}, {'exit_code': True},
                        {'library_unchanged': False}, {'source_after_sha256': {}}, {'runtime_dependencies_after': {}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); digest = fixture(root, **changes)
                with self.assertRaises(AssertionError):
                    closed_host_diagnostic(root, digest)

    def test_wrong_hash_proof_or_local_workflow_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); digest = fixture(root)
            with self.assertRaises(ValueError):
                closed_host_diagnostic(root, digest + '0')
            with self.assertRaises(AssertionError):
                closed_host_diagnostic(root, '0' * 64)
            path = root / 'results/native-host-stage-full4k-attempt1-verified.json'
            p = json.loads(path.read_bytes()); p['status'] = 'native_host_stage_full4k_failure_preserved'; path.write_text(json.dumps(p))
            with self.assertRaises(AssertionError):
                closed_host_diagnostic(root, digest)

    def test_owned_worker_actual_success_is_recorded_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); closed = []
            worker = SimpleNamespace(close=lambda: closed.append(True), process=SimpleNamespace(returncode=0))
            finish_worker(worker, root)
            self.assertEqual(closed, [True])
            self.assertEqual(json.loads((root / 'worker-exit.json').read_bytes())['exit_code'], 0)
            with self.assertRaises(FileExistsError):
                finish_worker(worker, root)

    def test_failed_or_unknown_owned_exit_is_recorded_and_rejected(self):
        for code in (None, True, 1, -15):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                worker = SimpleNamespace(close=lambda: None, process=SimpleNamespace(returncode=code))
                with self.assertRaises(AssertionError):
                    finish_worker(worker, root)
                self.assertEqual(json.loads((root / 'worker-exit.json').read_bytes())['exit_code'], code)


if __name__ == '__main__':
    unittest.main()
