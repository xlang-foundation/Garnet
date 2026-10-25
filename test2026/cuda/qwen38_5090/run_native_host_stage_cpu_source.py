"""Run explicitly pinned CPU controls after original model diagnosis is terminal."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import unittest
import zipfile

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parent)]
from test2026.cuda.qwen38_5090.native_host_stage_cpu_source import JOB, ORIGINAL, PREFIX, closed_original
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def ids(suite):
    result = []
    for value in suite:
        result.extend(ids(value) if isinstance(value, unittest.TestSuite) else [value.id()])
    return result


def child(p, out):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(p['test_modules'])
    assert not loader.errors and ids(suite) == p['test_ids'] and suite.countTestCases() == 29
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'test_ids': p['test_ids'], 'source_sha256': p['files'], 'tests_run': result.testsRun,
              'successful': result.wasSuccessful(), 'errors': [case.id() for case, _ in result.errors],
              'failures': [case.id() for case, _ in result.failures], 'skipped': [case.id() for case, _ in result.skipped],
              'expected_failures': [case.id() for case, _ in result.expectedFailures],
              'unexpected_successes': [case.id() for case in result.unexpectedSuccesses]}
    with (out / 'cpu-result.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
    return 0 if result.wasSuccessful() and result.testsRun == 29 and not any(report[k] for k in ('errors', 'failures', 'skipped', 'expected_failures', 'unexpected_successes')) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-archive-sha256')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    root = Path('D:/CantorAI/5090-study-20261023')
    methods = Path(__file__).resolve().parent
    assert methods == (root / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    prep = methods / 'evidence/native-host-stage-cpu-source-preparation.json'
    p = json.loads(prep.read_bytes())
    assert p['status'] == 'native_host_stage_cpu_source_supplement_prepared_unrun'
    for name, expected in p['files'].items():
        assert sha(root / name) == expected
    out = root / 'results' / JOB
    if args.child:
        assert out.is_dir()
        return child(p, out)
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    w_original, p_original = closed_original(root, args.original_archive_sha256 or '')
    assert w_original['preparation_sha256'] == p['original_preparation_sha256']
    assert sha(methods / 'verify_native_host_stage.py') == p['original_verifier_sha256']
    for name, expected in p_original['files'].items():
        assert sha(root / name) == expected
    for name, expected in p_original['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / name) == expected
    archive = root / 'results' / (JOB + '-evidence.zip')
    assert not out.exists() and not archive.exists() and not (root / 'results' / (ORIGINAL + '-verified.json')).exists()
    idle()
    out.mkdir()
    os.chdir(root / 'Garnet')
    original_io = methods / 'test_native_worker_io.py'
    (out / 'original-worker-io-test-postrun.py').write_bytes(original_io.read_bytes())
    w = {'status': 'running', 'preparation_sha256': sha(prep), 'original_archive_sha256': args.original_archive_sha256,
         'original_workflow': w_original, 'source_sha256': p['files'], 'original_sources_before': p_original['files'],
         'runtime_before': p_original['runtime_dependencies'], 'original_io_test_postrun_sha256': sha(original_io),
         'cleanup_finished': False, 'gpu_executed': False}
    def save():
        (out / 'workflow.json').write_text(json.dumps(w, indent=2), encoding='utf8')
    save()
    common = [root / path for path in p['files']] + [prep]
    code = 1
    try:
        initial = run([sys.executable, str(methods / 'verify_native_host_stage.py'), '--proofs', str(root / 'results')],
                      out / 'initial-replay-ledger', 'correctness', 'Preserve original strict metadata-count failure before source supplement', common)
        log = (out / 'initial-replay-ledger/commands.log').read_bytes()
        assert initial == 1 and b"p['expected_cpu_test_count']" in log and b'AssertionError' in log
        code = run([sys.executable, str(Path(__file__)), '--child'], out / 'cpu-ledger', 'correctness',
                   'Fresh29explicit source-pinned host/contract/IO tests after original model closure, no model/GPU', common)
        if code:
            raise RuntimeError('Pinned supplemental CPU controls failed')
        w['status'] = 'native_host_stage_cpu_source_supplement_collected'
    except BaseException as error:
        w.update(status='failed', error=repr(error)); code = code or 1
    finally:
        w.update(exit_code=code, source_after_sha256={path: sha(root / path) for path in p['files']},
                 original_sources_after={path: sha(root / path) for path in p_original['files']},
                 runtime_after={path: sha(root / 'build/bin' / path) for path in p_original['runtime_dependencies']})
        idle(); w['cleanup_finished'] = True; save()
        paths = [path for path in out.rglob('*') if path.is_file()] + common
        raw = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path, content in raw.items():
                z.writestr(path, content)
            z.writestr('evidence-manifest.json', json.dumps({'files': {path: hashlib.sha256(content).hexdigest() for path, content in raw.items()}}, indent=2))
        print(json.dumps({'status': w['status'], 'archive_sha256': sha(archive), 'bytes': archive.stat().st_size}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
