"""Independently replay closed host diagnostics; host wall time is not GPU time."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_host_stage_admission import replay_report
from test2026.cuda.qwen38_5090.native_host_stage_cpu_source import replay_supplement

sha = lambda raw: hashlib.sha256(raw).hexdigest()
METHODS = 'Garnet/test2026/cuda/qwen38_5090/'


def bind_actual_retained_cases(cases, proofs):
    parent = 'fixed-length-http-study-attempt1'
    archive = proofs / (parent + '-evidence.zip')
    assert sha(archive.read_bytes()) == cases['parent_archive_sha256']
    proof_path = proofs / (parent + '-verified.json')
    assert sha(proof_path.read_bytes()) == cases['parent_proof_sha256']
    q = json.loads(proof_path.read_bytes())
    assert q['status'] == 'fixed_length_http_study_replayed' and q['native_fixed_4k_metrics_replayed'] is True
    assert q['archive_sha256'] == cases['parent_archive_sha256'] and q['library_sha256'] == cases['library_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, digest in manifest['files'].items():
            assert sha(z.read(name)) == digest
        workers = {}
        for mode in ('ordinary', 'mtp'):
            events = [json.loads(json.loads(line)['line']) for line in z.read(
                'results/' + parent + '/' + mode + '/worker/worker-stdout.jsonl').splitlines()]
            completed = [e for e in events if e.get('event') == 'completed']
            assert len({e['id'] for e in completed}) == len(completed)
            workers[mode] = {e['id']: e for e in completed}
        for case in cases['cases']:
            assert len(case['prompt_ids']) == 3968
            content = case['body']['messages'][0]['content']
            image_sha = sha(base64.b64decode(content[0]['image_url']['url'].split(',', 1)[1], validate=True)) if case['case'] == 'image' else None
            assert image_sha == case['image_sha256']
            for mode in ('ordinary', 'mtp'):
                raw = z.read('results/' + parent + '/' + mode + '/benchmark4k/' + case['parent_client_keys'][mode] + '-request.json')
                assert sha(raw) == case['request_sha256'] and json.loads(raw) == case['body']
                actual = workers[mode][case['parent_worker_request_ids'][mode]]
                assert actual['prompt_ids'] == case['prompt_ids'] and actual['generated_ids'] == case['generated_ids_by_mode'][mode]
                assert len(actual['generated_ids']) == 128 and actual['image_sha256'] == image_sha and actual['ignore_eos'] is True
    return {'retained_cases': 4, 'ordinary_and_mtp_controls': 8,
            'parent_archive_sha256': cases['parent_archive_sha256'], 'actual_worker_ids_and_request_bytes_bound': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    args = parser.parse_args()
    job = 'native-host-stage-full4k-attempt1'
    base = 'results/' + job + '/'
    archive = args.proofs / (job + '-evidence.zip')
    fresh = peak = binding = None
    cpu_supplement = None
    with zipfile.ZipFile(archive) as z:
        doc = lambda name: json.loads(z.read(name))
        manifest = doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, digest in manifest['files'].items():
            assert sha(z.read(name)) == digest
        workflow = doc(base + 'workflow.json')
        raw = z.read(METHODS + 'evidence/native-host-stage-run-preparation.json')
        p = json.loads(raw)
        assert sha(raw) == workflow['preparation_sha256'] and p['status'] == 'native_host_stage_prepared_gpu_unrun'
        assert workflow['cleanup_finished'] is True and workflow['library_unchanged'] is True
        assert workflow['source_sha256'] == workflow['source_after_sha256'] == p['files']
        assert workflow['runtime_dependencies'] == workflow['runtime_dependencies_after'] == p['runtime_dependencies']
        assert workflow['library_sha256'] == p['runtime_dependencies']['garnet.dll']
        for name, digest in p['files'].items():
            assert sha(z.read(name)) == digest
        for helper in ('native_host_stage_admission.py', 'replay_native_host_stage_trace.py'):
            assert sha(Path(__file__).with_name(helper).read_bytes()) == sha(z.read(METHODS + helper))
        for field in ('closed_reference', 'closed_native_full4k'):
            entry = workflow[field]
            assert entry['cleanup_finished'] is True
            parent_archive = args.proofs / (entry['job'] + '-evidence.zip')
            assert sha(parent_archive.read_bytes()) == entry['archive_sha256']
            raw = z.read('results/' + entry['job'] + '-verified.json')
            proof = json.loads(raw)
            assert sha(raw) == entry['proof_sha256'] and proof['archive_sha256'] == entry['archive_sha256'] and proof['status'] == entry['proof_status']
            with zipfile.ZipFile(parent_archive) as prior:
                closed = json.loads(prior.read('results/' + entry['job'] + '/workflow.json'))
            assert closed == doc('results/' + entry['job'] + '/workflow.json')
            assert type(closed['exit_code']) is int and closed['exit_code'] == entry['exit_code'] and closed['cleanup_finished'] is True
            assert closed['source_sha256'] == closed['source_after_sha256'] and closed['runtime_dependencies'] == closed['runtime_dependencies_after']
            if field == 'closed_reference':
                assert type(closed['server_exit_code']) is int
            else:
                assert closed['library_unchanged'] is True and entry['archive_sha256'] == p['full4k_parent']['archive_sha256'] and entry['proof_sha256'] == p['full4k_parent']['proof_sha256']
        assert type(workflow['exit_code']) is int
        passed = workflow['exit_code'] == 0
        if passed:
            assert workflow['status'] == 'native_host_stage_collected_needs_replay'
            for ledger in ('cpu-ledger', 'gpu-ledger'):
                events = [json.loads(line) for line in z.read(base + ledger + '/commands.jsonl').splitlines()]
                assert len(events) == 2 and events[0]['event'] == 'started' and events[-1]['event'] == 'finished'
                assert type(events[-1]['exit_code']) is int and events[-1]['exit_code'] == 0
            log = z.read(base + 'cpu-ledger/commands.log')
            assert b'Ran 27 tests' in log and b'\nOK' in log
            cpu_supplement = replay_supplement(archive, workflow, p, args.proofs)
            raw = z.read(METHODS + 'evidence/native-host-stage-cases.json')
            cases = json.loads(raw)
            report = doc(base + 'native/report.json')
            assert sha(raw) == report['cases_sha256'] and workflow['library_sha256'] == cases['library_sha256']
            binding = bind_actual_retained_cases(cases, args.proofs)
            fresh = replay_report(report, lambda name: z.read(base + 'native/' + name), cases)
            resource = doc(base + 'resources/resource-monitor.json')
            samples = [json.loads(line) for line in z.read(base + 'resources/resource-vram.jsonl').splitlines()]
            peak = max(row['used_bytes'] for row in samples)
            assert type(resource['child_exit_code']) is int and resource['child_exit_code'] == 0
            assert resource['gpu']['uuid'] == 'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
            assert resource['sampled_memory']['sampled_peak_adapter_used_bytes'] == peak
        result = {'status': 'native_host_stage_full4k_replayed' if passed else 'native_host_stage_full4k_failure_preserved',
                  'archive_sha256': sha(archive.read_bytes()), 'files_verified': len(manifest['files']),
                  'library_sha256': workflow['library_sha256'], 'retained_case_binding': binding, 'cpu_source_supplement': cpu_supplement,
                  'replayed_diagnostic': fresh, 'correctness_sampled_peak_adapter_used_bytes': peak,
                  'performance_admitted': False, 'isolated_gpu_or_kernel_time_measured': False,
                  'two_times_target_achieved': False, 'replay_verifier_sha256': sha(Path(__file__).read_bytes()),
                  'archived_replay_verifier_sha256': sha(z.read(METHODS + 'verify_native_host_stage.py')),
                  'scope': p['scope']}
    target = args.proofs / (job + '-verified.json')
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
