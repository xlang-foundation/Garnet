"""Replay a separate29test source supplement; preserve the original27test ledger."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

JOB = 'native-host-stage-cpu-source-supplement-attempt1'
ORIGINAL = 'native-host-stage-full4k-attempt1'
PREFIX = 'Garnet/test2026/cuda/qwen38_5090/'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def closed_original(root, digest):
    if re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('Actual closed original archive required')
    archive = root / 'results' / (ORIGINAL + '-evidence.zip')
    assert sha(archive.read_bytes()) == digest
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            assert sha(z.read(name)) == expected
        w = json.loads(z.read('results/' + ORIGINAL + '/workflow.json'))
        p_raw = z.read(PREFIX + 'evidence/native-host-stage-run-preparation.json')
        p = json.loads(p_raw)
        assert sha(p_raw) == w['preparation_sha256']
        log = z.read('results/' + ORIGINAL + '/cpu-ledger/commands.log')
    assert w == json.loads((root / 'results' / ORIGINAL / 'workflow.json').read_bytes())
    assert type(w['exit_code']) is int and w['exit_code'] == 0 and w['cleanup_finished'] is True
    assert w['library_unchanged'] is True and w['source_sha256'] == w['source_after_sha256'] == p['files']
    assert w['runtime_dependencies'] == w['runtime_dependencies_after'] == p['runtime_dependencies']
    assert p['expected_cpu_test_count'] == 29 and PREFIX + 'test_native_worker_io.py' not in p['files']
    assert b'Ran 27 tests' in log and b'\nOK' in log and b'FAILED' not in log
    return w, p


def replay_supplement(original_archive, original_workflow, original_preparation, proofs):
    """Only the documented missing test-source pin can use this explicit repair."""
    assert original_preparation['expected_cpu_test_count'] == 29
    assert PREFIX + 'test_native_worker_io.py' not in original_preparation['files']
    original_sha = sha(original_archive.read_bytes())
    with zipfile.ZipFile(original_archive) as before:
        original_log = before.read('results/' + ORIGINAL + '/cpu-ledger/commands.log')
        original_prep_raw = before.read(PREFIX + 'evidence/native-host-stage-run-preparation.json')
        assert b'Ran 27 tests' in original_log and b'\nOK' in original_log and b'FAILED' not in original_log
    archive = proofs / (JOB + '-evidence.zip')
    base = 'results/' + JOB + '/'
    with zipfile.ZipFile(archive) as z:
        doc = lambda path: json.loads(z.read(path))
        manifest = doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for path, expected in manifest['files'].items():
            assert sha(z.read(path)) == expected
        w = doc(base + 'workflow.json')
        p_raw = z.read(PREFIX + 'evidence/native-host-stage-cpu-source-preparation.json')
        p = json.loads(p_raw)
        assert p['original_preparation_sha256'] == sha(original_prep_raw) == original_workflow['preparation_sha256']
        assert sha(p_raw) == w['preparation_sha256']
        assert w['status'] == 'native_host_stage_cpu_source_supplement_collected'
        assert w['gpu_executed'] is False
        assert type(w['exit_code']) is int and w['exit_code'] == 0 and w['cleanup_finished'] is True
        assert w['original_archive_sha256'] == original_sha
        assert w['original_workflow'] == original_workflow
        assert w['original_sources_before'] == w['original_sources_after'] == original_preparation['files']
        assert w['runtime_before'] == w['runtime_after'] == original_preparation['runtime_dependencies']
        assert w['source_sha256'] == w['source_after_sha256'] == p['files']
        for path, expected in p['files'].items():
            assert sha(z.read(path)) == expected
        assert sha(Path(__file__).read_bytes()) == p['files'][PREFIX + 'native_host_stage_cpu_source.py']
        expected_modules = ['test2026.cuda.qwen38_5090.' + name for name in (
            'test_native_host_stage_trace', 'test_native_host_stage_probe',
            'test_native_host_stage_admission', 'test_fixed_length_worker_contract',
            'test_native_host_stage_worker_io')]
        assert p['test_modules'] == expected_modules
        for path in p['test_source_files']:
            if path == PREFIX + 'test_native_host_stage_worker_io.py':
                assert p['files'][path] == p['new_io_source_sha256']
            else:
                assert p['files'][path] == original_preparation['files'][path]
        assert len(p['test_source_files']) == 5 and len(p['test_ids']) == len(set(p['test_ids'])) == 29
        report = doc(base + 'cpu-result.json')
        assert report['test_ids'] == p['test_ids'] and report['source_sha256'] == p['files']
        assert type(report['tests_run']) is int and report['tests_run'] == 29 and report['successful'] is True
        for field in ('errors', 'failures', 'skipped', 'expected_failures', 'unexpected_successes'):
            assert report[field] == []
        for ledger, expected in (('initial-replay-ledger', 1), ('cpu-ledger', 0)):
            events = [json.loads(line) for line in z.read(base + ledger + '/commands.jsonl').splitlines()]
            assert len(events) == 2 and events[0]['event'] == 'started' and events[-1]['event'] == 'finished'
            assert type(events[-1]['exit_code']) is int and events[-1]['exit_code'] == expected
        initial = z.read(base + 'initial-replay-ledger/commands.log')
        assert b"p['expected_cpu_test_count']" in initial and b'AssertionError' in initial
        log = z.read(base + 'cpu-ledger/commands.log')
        assert b'Ran 29 tests' in log and b'\nOK' in log and b'FAILED' not in log
        assert sha(z.read(PREFIX + 'verify_native_host_stage.py')) == p['original_verifier_sha256']
        original_io = z.read(base + 'original-worker-io-test-postrun.py')
        assert sha(original_io) == w['original_io_test_postrun_sha256']
    return {'status': 'native_host_stage_cpu_source_supplement_replayed',
            'archive_sha256': sha(archive.read_bytes()), 'original_archive_sha256': original_sha,
            'original_tests_before_model': 27, 'original_metadata_expected': 29,
            'fresh_pinned_tests_after_model': 29, 'original_archive_unchanged': True,
            'gpu_model_rerun': False, 'scope': 'Explicit source-pinned supplemental CPU qualification only. Original27test ledger and29count mismatch preserved. Every host trace/token/state/metric guard remains in the original diagnostic verifier. Original IO source is observed post-run, not retroactively certified for its earlier execution.'}
