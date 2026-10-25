"""Replay linked source/actual bootstrap control; failure stays failure."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import sha
from test2026.cuda.qwen38_5090.replay_native_host_stage_trace import replay

JOB = 'native-cuda-phase-logged-bootstrap-attempt1'
PARENT = 'native-cuda-phase-full4k-attempt1'
PREFIX = 'Garnet/test2026/cuda/qwen38_5090/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    args = parser.parse_args()
    archive = args.proofs / (JOB + '-evidence.zip')
    base = 'results/' + JOB + '/'
    control = None
    with zipfile.ZipFile(archive) as z:
        doc = lambda name: json.loads(z.read(name))
        manifest = doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        w = doc(base + 'workflow.json')
        raw = z.read(PREFIX + 'evidence/native-cuda-logged-bootstrap-preparation.json')
        p = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == w['preparation_sha256']
        assert w['source_sha256'] == w['source_after_sha256'] == p['files']
        assert w['runtime_dependencies'] == w['runtime_dependencies_after'] == p['runtime_dependencies']
        assert w['cleanup_finished'] is True and w['library_unchanged'] is True and w['profiled'] is False
        parent = args.proofs / (PARENT + '-evidence.zip')
        assert sha(parent) == w['parent_archive_sha256'] == p['parent_archive_sha256'] == manifest['linked_parent_archive_sha256']
        proof_raw = z.read('results/' + PARENT + '-verified.json')
        assert hashlib.sha256(proof_raw).hexdigest() == w['parent_proof_sha256'] == p['parent_proof_sha256']
        q = json.loads(proof_raw)
        assert q['status'] == 'native_cuda_phases_full4k_failure_preserved' and q['archive_sha256'] == sha(parent)
        with zipfile.ZipFile(parent) as prior:
            old_manifest = json.loads(prior.read('evidence-manifest.json'))
            assert prior.testzip() is None
            for name, expected in old_manifest['files'].items():
                assert hashlib.sha256(prior.read(name)).hexdigest() == expected
            old = json.loads(prior.read('results/' + PARENT + '/workflow.json'))
            assert old['cleanup_finished'] is True and old['source_sha256'] == old['source_after_sha256']
            assert old['runtime_dependencies'] == old['runtime_dependencies_after'] == p['runtime_dependencies']
            assert set(p['files']) == set(old['source_sha256']) | set(p['new_files'])
            for name, digest in p['files'].items():
                raw = z.read(name) if name in p['new_files'] else prior.read(name)
                assert hashlib.sha256(raw).hexdigest() == digest
            cases = json.loads(prior.read(PREFIX + 'evidence/native-host-stage-cases.json'))
        assert sha(Path(__file__)) == p['files'][PREFIX + 'verify_native_cuda_logged_bootstrap.py']
        assert type(w['exit_code']) is int
        passed = w['exit_code'] == 0
        if passed:
            assert w['status'] == 'native_logged_bootstrap_collected_needs_replay'
            for ledger in ('cpu-ledger', 'embedded-cpu-ledger', 'gpu-ledger'):
                events = [json.loads(line) for line in z.read(base + ledger + '/commands.jsonl').splitlines()]
                assert len(events) == 2 and events[-1]['event'] == 'finished'
                assert type(events[-1]['exit_code']) is int and events[-1]['exit_code'] == 0
            log = z.read(base + 'cpu-ledger/commands.log')
            assert b'Ran 13 tests' in log and b'\nOK' in log and b'FAILED' not in log
            embedded = doc(base + 'embedded-cpu/report.json')
            assert embedded['status'] == 'embedded_python_trace_cpu_compatibility_passed' and embedded['gpu_executed'] is False
            report = doc(base + 'native/report.json')
            assert report['status'] == 'native_logged_bootstrap_retained_control_completed' and report['profiled'] is False
            assert type(report['worker_exit_code']) is int and report['worker_exit_code'] == 0 and len(report['requests']) == 1
            case = cases['cases'][0]
            key = report['requests'][0]['id']
            events = []
            for line in z.read(base + 'native/worker/worker-stdout.jsonl').splitlines():
                try:
                    event = json.loads(json.loads(line)['line'])
                except ValueError:
                    continue
                assert isinstance(event, dict) and event.get('event') in ('ready', 'token', 'completed')
                events.append(event)
            rows = [json.loads(line) for line in z.read(base + 'native/host-stages.jsonl').splitlines()]
            expected = {key: {'case': 'text', 'prompt_ids': case['prompt_ids'], 'generated_ids': case['generated_ids_by_mode']['ordinary'], 'image_sha256': None}}
            control = replay(rows, events, expected)
            assert control == report['host_replay'] and len(control['requests']) == 1
            markers = [json.loads(line) for line in z.read(base + 'native/host-stages.cuda-markers.jsonl').splitlines()]
            assert len(markers) == report['marker_count'] == 12 and [v['ordinal'] for v in markers] == list(range(12))
            assert all(v['id'] == key and v['status'] == 'completed' for v in markers)
            close = doc(base + 'native/host-stages.model-closed.json')
            assert close == report['native_model_cleanup'] and close['marker_count'] == 12 and close['library_sha256'] == w['library_sha256']
        logs = {name.removeprefix(base): [json.loads(line) for line in z.read(name).splitlines()]
                for name in manifest['files'] if name.startswith(base) and name.endswith('.bootstrap.jsonl')}
        result = {'status': 'native_cuda_logged_bootstrap_replayed' if passed else 'native_cuda_logged_bootstrap_failure_preserved',
                  'archive_sha256': sha(archive), 'linked_parent_archive_sha256': sha(parent), 'files_verified': len(manifest['files']),
                  'library_sha256': w['library_sha256'], 'control_replay': control, 'bootstrap_logs': logs,
                  'profiled': False, 'performance_admitted': False, 'two_times_target_achieved': False, 'scope': p['scope']}
    target = args.proofs / (JOB + '-verified.json')
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps({'status': result['status'], 'archive_sha256': result['archive_sha256'], 'files_verified': result['files_verified'], 'proof_sha256': sha(target)}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
