"""Synthetic CPU archive/parser controls only; no model/GPU evidence produced."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import tempfile
import unittest
import zipfile

from .native_host_stage_cpu_source import JOB, ORIGINAL, PREFIX, closed_original, replay_supplement
from .run_native_host_stage_cpu_source import ids

M = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def write_archive(path, files):
    with zipfile.ZipFile(path, 'w') as z:
        for name, raw in files.items():
            z.writestr(name, raw)
        z.writestr('evidence-manifest.json', json.dumps({'files': {name: sha(raw) for name, raw in files.items()}}))


class RestoreOriginal(ast.NodeTransformer):
    def __init__(self, original):
        self.original_assert = next(node for node in ast.walk(original) if isinstance(node, ast.Assert)
                                    and 'expected_cpu_test_count' in ast.unparse(node))
    def visit_ImportFrom(self, node):
        return None if node.module and node.module.endswith('.native_host_stage_cpu_source') else node
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == 'cpu_supplement' for target in node.targets):
            return None
        return self.generic_visit(node)
    def visit_Assert(self, node):
        if any(isinstance(value, ast.Constant) and value.value == b'Ran 27 tests' for value in ast.walk(node)):
            return copy.deepcopy(self.original_assert)
        return self.generic_visit(node)
    def visit_Dict(self, node):
        pairs = [(key, value) for key, value in zip(node.keys, node.values)
                 if not (isinstance(key, ast.Constant) and key.value == 'cpu_source_supplement')]
        node.keys = [key for key, _ in pairs]; node.values = [value for _, value in pairs]
        return self.generic_visit(node)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-cpu-source-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proofs = self.root / 'results'; self.proofs.mkdir()
        names = ('test_native_host_stage_trace', 'test_native_host_stage_probe',
                 'test_native_host_stage_admission', 'test_fixed_length_worker_contract',
                 'test_native_host_stage_worker_io')
        test_sources = [PREFIX + name + '.py' for name in names]
        files = {path: (M / Path(path).name).read_bytes() for path in test_sources}
        files[PREFIX + 'native_host_stage_cpu_source.py'] = (M / 'native_host_stage_cpu_source.py').read_bytes()
        files[PREFIX + 'verify_native_host_stage.py'] = (M / 'verify_native_host_stage.py').read_bytes()
        original_sources = {name: sha(raw) for name, raw in files.items() if name in test_sources[:-1]}
        self.original_p = {'expected_cpu_test_count': 29, 'files': original_sources, 'runtime_dependencies': {'fake.dll': 'fixture-hash'}}
        self.original_w = {'exit_code': 0, 'cleanup_finished': True, 'library_unchanged': True,
                           'source_sha256': original_sources, 'source_after_sha256': original_sources,
                           'runtime_dependencies': self.original_p['runtime_dependencies'],
                           'runtime_dependencies_after': self.original_p['runtime_dependencies']}
        self.original_w['preparation_sha256'] = sha(json.dumps(self.original_p).encode())
        original_folder = self.proofs / ORIGINAL; original_folder.mkdir()
        raw = json.dumps(self.original_w).encode(); (original_folder / 'workflow.json').write_bytes(raw)
        self.original_archive = self.proofs / (ORIGINAL + '-evidence.zip')
        write_archive(self.original_archive, {'results/' + ORIGINAL + '/workflow.json': raw,
            PREFIX + 'evidence/native-host-stage-run-preparation.json': json.dumps(self.original_p).encode(),
            'results/' + ORIGINAL + '/cpu-ledger/commands.log': b'EXPLICIT SYNTHETIC CPU FIXTURE\nRan 27 tests\nOK\n'})
        modules = ['test2026.cuda.qwen38_5090.' + name for name in names]
        loader = unittest.TestLoader(); test_ids = ids(loader.loadTestsFromNames(modules))
        self.assertFalse(loader.errors); self.assertEqual(len(test_ids), 29)
        p = {'files': {name: sha(raw) for name, raw in files.items()}, 'test_source_files': test_sources,
             'original_preparation_sha256': self.original_w['preparation_sha256'],
             'test_modules': modules, 'test_ids': test_ids,
             'new_io_source_sha256': sha(files[test_sources[-1]]),
             'original_verifier_sha256': sha(files[PREFIX + 'verify_native_host_stage.py'])}
        prep_raw = json.dumps(p).encode(); files[PREFIX + 'evidence/native-host-stage-cpu-source-preparation.json'] = prep_raw
        w = {'status': 'native_host_stage_cpu_source_supplement_collected', 'exit_code': 0,
             'cleanup_finished': True, 'gpu_executed': False, 'preparation_sha256': sha(prep_raw),
             'original_archive_sha256': sha(self.original_archive.read_bytes()), 'original_workflow': self.original_w,
             'original_sources_before': original_sources, 'original_sources_after': original_sources,
             'runtime_before': self.original_p['runtime_dependencies'], 'runtime_after': self.original_p['runtime_dependencies'],
             'source_sha256': p['files'], 'source_after_sha256': p['files'],
             'original_io_test_postrun_sha256': sha(b'EXPLICIT POSTRUN SOURCE FIXTURE')}
        report = {'test_ids': test_ids, 'source_sha256': p['files'], 'tests_run': 29, 'successful': True,
                  'errors': [], 'failures': [], 'skipped': [], 'expected_failures': [], 'unexpected_successes': []}
        self.base = 'results/' + JOB + '/'
        files[self.base + 'workflow.json'] = json.dumps(w).encode()
        files[self.base + 'cpu-result.json'] = json.dumps(report).encode()
        files[self.base + 'original-worker-io-test-postrun.py'] = b'EXPLICIT POSTRUN SOURCE FIXTURE'
        for ledger, code in (('initial-replay-ledger', 1), ('cpu-ledger', 0)):
            files[self.base + ledger + '/commands.jsonl'] = b'\n'.join(json.dumps(row).encode() for row in (
                {'event': 'started'}, {'event': 'finished', 'exit_code': code}))
        files[self.base + 'initial-replay-ledger/commands.log'] = b"EXPLICIT CPU FIXTURE p['expected_cpu_test_count'] AssertionError"
        files[self.base + 'cpu-ledger/commands.log'] = b'EXPLICIT SYNTHETIC CPU FIXTURE\nRan 29 tests\nOK\n'
        self.files = files; self.archive = self.proofs / (JOB + '-evidence.zip')
        write_archive(self.archive, files)

    def replay(self):
        return replay_supplement(self.original_archive, self.original_w, self.original_p, self.proofs)

    def test_original27_preserved_and_fresh29_source_guard_admitted_only_as_cpu(self):
        w, p = closed_original(self.root, sha(self.original_archive.read_bytes()))
        self.assertEqual((w, p), (self.original_w, self.original_p))
        result = self.replay()
        self.assertEqual(result['original_tests_before_model'], 27)
        self.assertEqual(result['fresh_pinned_tests_after_model'], 29)
        self.assertFalse(result['gpu_model_rerun'])

    def test_false_counts_skips_errors_or_source_drift_cannot_replace_guard(self):
        for name, changes in (('cpu-result.json', {'tests_run': 28}), ('cpu-result.json', {'skipped': ['fake']}),
                              ('cpu-result.json', {'errors': ['fake']}), ('cpu-result.json', {'test_ids': []}),
                              ('workflow.json', {'cleanup_finished': False}), ('workflow.json', {'gpu_executed': True}),
                              ('workflow.json', {'source_after_sha256': {}}), ('workflow.json', {'original_sources_after': {}})):
            with self.subTest(name=name, changes=changes):
                files = dict(self.files); path = self.base + name; row = json.loads(files[path]); row.update(changes); files[path] = json.dumps(row).encode()
                write_archive(self.archive, files)
                with self.assertRaises(AssertionError): self.replay()

    def test_supplement_must_preserve_initial_failure_and_clean_fresh_run(self):
        for path in (self.base + 'initial-replay-ledger/commands.log', self.base + 'cpu-ledger/commands.log'):
            files = dict(self.files); files[path] = b'EXPLICIT TRUNCATED CPU FIXTURE'; write_archive(self.archive, files)
            with self.assertRaises(AssertionError): self.replay()

    def test_unfinished_original_or_wrong_archive_is_not_released(self):
        with self.assertRaises(ValueError): closed_original(self.root, 'bad-hash')
        with self.assertRaises(AssertionError): closed_original(self.root, '0' * 64)
        path = self.proofs / ORIGINAL / 'workflow.json'; row = json.loads(path.read_bytes()); row['cleanup_finished'] = False; path.write_text(json.dumps(row))
        with self.assertRaises(AssertionError): closed_original(self.root, sha(self.original_archive.read_bytes()))

    def test_original_verifier_whole_ast_unchanged_except_explicit_cpu_supplement(self):
        original = ast.parse((M / 'verify_native_host_stage.py').read_bytes())
        candidate = ast.parse((M / 'verify_native_host_stage_cpu_source.py').read_bytes())
        candidate = RestoreOriginal(original).visit(candidate)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(candidate, include_attributes=False))


if __name__ == '__main__': unittest.main()
