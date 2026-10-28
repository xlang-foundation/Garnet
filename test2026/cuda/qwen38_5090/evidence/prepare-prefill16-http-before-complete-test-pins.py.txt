"""Pin explicit wider-MTP HTTP study to actual full4K and native ordinary proofs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sha = lambda raw: hashlib.sha256(raw).hexdigest()
METHODS = ('native_prefill16_http_admission.py', 'native_prefill16_stream_worker.py',
           'serve_native_prefill16_worker.bat', 'probe_prefill16_http.py', 'test_prefill16_http_worker.py',
           'prefill16_http_study.py', 'test_prefill16_http_study.py', 'run_prefill16_http.py',
           'verify_prefill16_http.py', 'prepare_prefill16_http.py', 'test_prefill16_http_replay.py',
           'evidence/prefill16-http-worker-derivation.json', 'evidence/prefill16-http-verifier-derivation.json',
           'cohort_memory_replay.py', 'verify_fixed_length_http_snapshot.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    parents = {}
    for name, status in (('fixed-length-http-study-attempt1', 'fixed_length_http_study_replayed'),
                         ('mtp-prefill16-full4k-admission-attempt1', 'mtp_prefill16_full4k_target_state_replayed')):
        raw = (args.proofs / (name + '-verified.json')).read_bytes()
        proof = json.loads(raw)
        assert proof == json.loads((args.proofs / (name + '-remote-verified.json')).read_bytes())
        archive = args.proofs / (name + '-evidence.zip')
        assert proof['status'] == status and sha(archive.read_bytes()) == proof['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('evidence-manifest.json'))
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
            for path, digest in manifest['files'].items():
                assert sha(z.read(path)) == digest
            workflow = json.loads(z.read('results/' + name + '/workflow.json'))
            if name == 'fixed-length-http-study-attempt1':
                old = json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/fixed-length-http-preparation.json'))
            else:
                files = dict(workflow['source_sha256'])
                runtime = workflow['runtime_dependencies']
        assert type(workflow['exit_code']) is int and workflow['exit_code'] == 0 and workflow['cleanup_finished'] is True
        assert workflow['library_unchanged'] is True and workflow['source_sha256'] == workflow['source_after_sha256']
        assert workflow['runtime_dependencies'] == workflow['runtime_dependencies_after']
        parents[name] = {'archive_sha256': proof['archive_sha256'], 'proof_sha256': sha(raw)}
    for name in METHODS:
        files['Garnet/test2026/cuda/qwen38_5090/' + name] = sha((methods / name).read_bytes())
    for name, digest in files.items():
        if name.startswith('Garnet/'):
            assert sha((repo / name.removeprefix('Garnet/')).read_bytes()) == digest, ('Qualified source drift', name)
    tests = ['test2026.cuda.qwen38_5090.' + name for name in (
        'test_prefill16_http_worker', 'test_prefill16_http_study', 'test_prefill16_http_replay',
        'test_mtp_prefill16_study', 'test_fixed_length_worker_contract', 'test_mmmu_http_quality',
        'test_mmmu_quality_protocol', 'test_fixed_length_native_stream', 'test_fixed_length_reference',
        'test_cohort_memory_replay')]
    sys.path[:0] = [str(repo), str(methods)]
    import unittest
    loader = unittest.TestLoader()
    count = loader.loadTestsFromNames(tests).countTestCases()
    assert not loader.errors
    scope = ('One exclusive native Windows B1 wider-MTP prefill16/T4decode fixed4K HTTP trial after actual host-diagnostic terminal closure and replay. '
             'Pinned1a1d target/runtime/checkpoint unchanged; explicit extra draft engine only. Full4K target-state admission required before model construction. '
             '12protocol/805natural-stop adapted quality cases/two full4K boundaries precede frozen124streams in eight text/image C1/2/4/8 cohorts. '
             'All actual quality/boundary/full4K output IDs must equal the actual closed ordinary native controls. No short pilot, CPU pipeline timings or249vs993call-count speed inference. '
             'Raw SSE/worker IDs/token usage/metrics/cohort VRAM samples/gaps, failures/actual worker exit/source hashes and owned cache retirement retained. '
             'Physical GPU batch remains1 and clients queue. Separate independent replay mandatory. Strongest tuned/MTP reference matching, repetitions, INT4/4B and2x remain open.')
    result = {'status': 'prefill16_http_prepared_gpu_unrun', 'files': files,
              'runtime_dependencies': runtime, 'test_modules': tests, 'expected_cpu_test_count': count,
              'native_parent': parents['fixed-length-http-study-attempt1'],
              'full4k_parent': parents['mtp-prefill16-full4k-admission-attempt1'],
              'scope': scope}
    for key in ('parent_proof_sha256', 'parent_archive_sha256', 'eos_proof_sha256', 'eos_archive_sha256'):
        result[key] = old[key]
    target = methods / 'evidence/prefill16-http-run-preparation.json'
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    uploads = [{'local': str(methods / name), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + name,
                'sha256': sha((methods / name).read_bytes())} for name in METHODS]
    uploads.append({'local': str(target), 'target': 'Garnet/test2026/cuda/qwen38_5090/evidence/' + target.name,
                    'sha256': sha(target.read_bytes())})
    raw = json.dumps({'uploads': uploads, 'scope': scope}, indent=2).encode()
    assert not args.plan.exists() or args.plan.read_bytes() == raw
    if not args.plan.exists():
        args.plan.write_bytes(raw)
    print(json.dumps({'status': result['status'], 'preparation_sha256': sha(target.read_bytes()),
                      'pinned_files': len(files), 'upload_targets': len(uploads), 'cpu_tests': count, 'gpu_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
