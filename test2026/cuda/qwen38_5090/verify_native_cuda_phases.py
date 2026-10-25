"""Independently bind full4K CUDA phases to real profiler records and native IDs."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import JOB, TRIAL, NSYS_SHA256, sha
from test2026.cuda.qwen38_5090.native_host_stage_admission import replay_report
from test2026.cuda.qwen38_5090.replay_native_cuda_phases import replay
from test2026.cuda.qwen38_5090.verify_native_host_stage_cpu_source import bind_actual_retained_cases

PREFIX = 'Garnet/test2026/cuda/qwen38_5090/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    assert args.scratch.is_dir()
    archive = args.proofs / (JOB + '-evidence.zip')
    base = 'results/' + JOB + '/'
    actual = host = peak = binding = None
    with zipfile.ZipFile(archive) as z:
        doc = lambda name: json.loads(z.read(name))
        manifest = doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, digest in manifest['files'].items():
            with z.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        w = doc(base + 'workflow.json')
        raw = z.read(PREFIX + 'evidence/native-cuda-phase-run-preparation.json')
        p = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == w['preparation_sha256']
        assert p['status'] == 'native_cuda_phases_prepared_gpu_unrun'
        assert w['cleanup_finished'] is True and w['library_unchanged'] is True
        assert w['source_sha256'] == w['source_after_sha256'] == p['files']
        assert w['runtime_dependencies'] == w['runtime_dependencies_after'] == p['runtime_dependencies']
        assert w['library_sha256'] == p['runtime_dependencies']['garnet.dll']
        assert w['profiler'] == w['profiler_after'] == p['profiler']
        assert w['profiler']['sha256'] == NSYS_SHA256
        assert hashlib.sha256(z.read(w['profiler']['setup_path'])).hexdigest() == w['profiler']['setup_sha256']
        for name, digest in p['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        for name in ('native_cuda_phase_study.py', 'native_cuda_phase_trace.py', 'replay_native_cuda_phases.py',
                     'replay_native_host_stage_trace.py', 'native_host_stage_admission.py', 'verify_native_cuda_phases.py'):
            assert sha(Path(__file__).with_name(name)) == p['files'][PREFIX + name]
        for field in ('closed_host', 'closed_prefill16_http'):
            parent = w[field]
            assert parent['cleanup_finished'] is True and type(parent['exit_code']) is int
            assert sha(args.proofs / (parent['job'] + '-evidence.zip')) == parent['archive_sha256']
            proof_raw = z.read('results/' + parent['job'] + '-verified.json')
            proof = json.loads(proof_raw)
            assert hashlib.sha256(proof_raw).hexdigest() == parent['proof_sha256']
            assert proof['archive_sha256'] == parent['archive_sha256'] and proof['status'] == parent['proof_status']
            with zipfile.ZipFile(args.proofs / (parent['job'] + '-evidence.zip')) as prior:
                closed = json.loads(prior.read('results/' + parent['job'] + '/workflow.json'))
            assert closed == doc('results/' + parent['job'] + '/workflow.json')
            assert closed['cleanup_finished'] is True and closed['exit_code'] == parent['exit_code']
            assert closed['source_sha256'] == closed['source_after_sha256'] and closed['runtime_dependencies'] == closed['runtime_dependencies_after']
            if field == 'closed_host':
                assert parent['exit_code'] == 0 and parent['proof_status'] == 'native_host_stage_full4k_replayed'
                assert parent['archive_sha256'] == p['host_parent']['archive_sha256'] and parent['proof_sha256'] == p['host_parent']['proof_sha256']
            else:
                assert parent['job'] == TRIAL
                assert parent['proof_status'] == ('prefill16_fixed4k_http_replayed' if parent['exit_code'] == 0 else 'prefill16_fixed4k_http_failure_preserved')
        assert type(w['exit_code']) is int
        passed = w['exit_code'] == 0
        if passed:
            assert w['status'] == 'native_cuda_phases_collected_needs_replay'
            for ledger in ('cpu-ledger', 'gpu-ledger'):
                events = [json.loads(line) for line in z.read(base + ledger + '/commands.jsonl').splitlines()]
                assert len(events) == 2 and events[0]['event'] == 'started' and events[-1]['event'] == 'finished'
                assert type(events[-1]['exit_code']) is int and events[-1]['exit_code'] == 0
            cpu = doc(base + 'cpu-result.json')
            assert cpu['test_ids'] == p['test_ids'] and len(cpu['test_ids']) == len(set(cpu['test_ids'])) == p['expected_cpu_test_count']
            assert type(cpu['tests_run']) is int and cpu['tests_run'] == p['expected_cpu_test_count'] and cpu['successful'] is True and cpu['gpu_executed'] is False
            assert cpu['source_sha256'] == p['files']
            assert all(cpu[field] == [] for field in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses'))
            log = z.read(base + 'cpu-ledger/commands.log')
            assert ('Ran ' + str(p['expected_cpu_test_count']) + ' tests').encode() in log and b'\nOK' in log and b'FAILED' not in log
            cases = doc(PREFIX + 'evidence/native-host-stage-cases.json')
            report = doc(base + 'native/report.json')
            assert report['cuda_phase_status'] == 'both_modes_actual_cuda_markers_collected_needs_independent_replay'
            binding = bind_actual_retained_cases(cases, args.proofs)
            host = replay_report(report, lambda name: z.read(base + 'native/' + name), cases)
            actual = {}
            for mode in ('ordinary', 'mtp'):
                prefix = base + 'native/' + mode + '/'
                stats = [json.loads(line) for line in z.read(prefix + 'stats-ledger/commands.jsonl').splitlines()]
                assert len(stats) == 2 and stats[-1]['event'] == 'finished' and type(stats[-1]['exit_code']) is int and stats[-1]['exit_code'] == 0
                assert report['modes'][mode]['stats_exit_code'] == 0
                rows = [json.loads(line) for line in z.read(prefix + 'host-stages.jsonl').splitlines()]
                markers = [json.loads(line) for line in z.read(prefix + 'host-stages.cuda-markers.jsonl').splitlines()]
                cleanup = doc(prefix + 'host-stages.model-closed.json')
                assert cleanup == report['modes'][mode]['native_model_cleanup']
                assert cleanup['status'] == 'native_model_close_calls_returned' and cleanup['mode'] == mode
                assert type(cleanup['marker_count']) is int and cleanup['marker_count'] == 144
                assert cleanup['library_sha256'] == w['library_sha256']
                # Use existing exact remote SQLite where possible; otherwise extract to
                # one fixed local scratch file and preserve any different prior bytes.
                name = prefix + 'cuda-trace.sqlite'
                sqlite_path = args.proofs / JOB / 'native' / mode / 'cuda-trace.sqlite'
                if not sqlite_path.exists():
                    sqlite_path = args.scratch / (JOB + '-' + mode + '.sqlite')
                if sqlite_path.exists():
                    assert sha(sqlite_path) == manifest['files'][name]
                else:
                    with z.open(name) as source, sqlite_path.open('xb') as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    assert sha(sqlite_path) == manifest['files'][name]
                with sqlite3.connect('file:' + sqlite_path.resolve().as_posix() + '?mode=ro', uri=True) as c:
                    fresh = replay(c, markers, rows)
                assert fresh == doc(prefix + 'cuda-phases-replayed.json')
                assert fresh['requests'] == 12 and fresh['unique_markers'] == 144
                assert report['modes'][mode]['cuda_phase_replay'] == {key: fresh[key] for key in ('status', 'requests', 'unique_markers', 'owner')}
                actual[mode] = fresh
            resource = doc(base + 'resources/resource-monitor.json')
            samples = [json.loads(line) for line in z.read(base + 'resources/resource-vram.jsonl').splitlines()]
            peak = max(row['used_bytes'] for row in samples)
            assert type(resource['child_exit_code']) is int and resource['child_exit_code'] == 0
            assert resource['gpu']['uuid'] == 'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
            assert resource['sampled_memory']['sampled_peak_adapter_used_bytes'] == peak
        result = {'status': 'native_cuda_phases_full4k_replayed' if passed else 'native_cuda_phases_full4k_failure_preserved',
                  'archive_sha256': sha(archive), 'files_verified': len(manifest['files']),
                  'library_sha256': w['library_sha256'], 'profiler': w['profiler'], 'retained_case_binding': binding,
                  'host_replay': host, 'cuda_phase_replay': actual,
                  'instrumented_correctness_sampled_peak_adapter_used_bytes': peak,
                  'performance_admitted': False, 'two_times_target_achieved': False,
                  'replay_verifier_sha256': sha(Path(__file__)), 'scope': p['scope']}
    target = args.proofs / (JOB + '-verified.json')
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps({'status': result['status'], 'archive_sha256': result['archive_sha256'],
                      'files_verified': result['files_verified'], 'proof_sha256': sha(target)}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
