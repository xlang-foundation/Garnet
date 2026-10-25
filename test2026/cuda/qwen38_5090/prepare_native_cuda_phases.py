"""Prepare current B1 full4K GPU diagnosis from actual closed host evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import NSYS_SHA256, sha
from test2026.cuda.qwen38_5090.run_native_cuda_phases import test_ids

METHODS = ('native_cuda_phase_trace.py', 'replay_native_cuda_phases.py', 'native_cuda_phase_stream_worker.py',
           'serve_native_cuda_phase_worker.bat', 'probe_native_cuda_phases.py', 'native_cuda_phase_study.py',
           'run_native_cuda_phases.py', 'verify_native_cuda_phases.py', 'prepare_native_cuda_phases.py',
           'test_native_cuda_phases.py', 'test_native_cuda_phase_study.py',
           'collect_native_cuda_phases.py', 'test_native_cuda_profiled_driver.py',
           'evidence/native-cuda-phase-profiled-driver-smoke-report.json',
           'evidence/native-cuda-phase-worker-derivation.json', 'evidence/native-cuda-phase-probe-derivation.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    host = 'native-host-stage-full4k-attempt1'
    raw = (args.proofs / (host + '-verified.json')).read_bytes()
    proof = json.loads(raw)
    assert proof == json.loads((args.proofs / (host + '-remote-verified.json')).read_bytes())
    assert proof['status'] == 'native_host_stage_full4k_replayed' and proof['cpu_source_supplement']['fresh_pinned_tests_after_model'] == 29
    archive = args.proofs / (host + '-evidence.zip')
    assert sha(archive) == proof['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, digest in manifest['files'].items():
            with z.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        w = json.loads(z.read('results/' + host + '/workflow.json'))
    assert type(w['exit_code']) is int and w['exit_code'] == 0 and w['cleanup_finished'] is True
    assert w['library_unchanged'] is True and w['source_sha256'] == w['source_after_sha256']
    assert w['runtime_dependencies'] == w['runtime_dependencies_after']
    files = dict(w['source_sha256'])
    current = json.loads((methods / 'evidence/prefill16-http-run-preparation.json').read_bytes())
    assert current['runtime_dependencies'] == w['runtime_dependencies']
    for name, digest in current['files'].items():
        assert name not in files or files[name] == digest, ('Parent source conflict', name)
        files[name] = digest
    for name in METHODS + ('test_native_host_stage_worker_io.py', 'run_native_host_stage_cpu_source.py'):
        files['Garnet/test2026/cuda/qwen38_5090/' + name] = sha(methods / name)
    for name, digest in files.items():
        if name.startswith('Garnet/'):
            assert sha(repo / name.removeprefix('Garnet/')) == digest, ('Qualified source drift', name)
    smoke = args.proofs / 'nsight-cuda-smoke-2026-attempt1-evidence.zip'
    q = json.loads((args.proofs / 'nsight-cuda-smoke-2026-attempt1-verified.json').read_bytes())
    assert sha(smoke) == q['archive_sha256']
    with zipfile.ZipFile(smoke) as z:
        setup_path = 'results/nsight-2026-cab-extraction/setup.json'
        setup_raw = z.read(setup_path)
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert hashlib.sha256(setup_raw).hexdigest() == manifest['files'][setup_path]
        setup = json.loads(setup_raw)
    assert setup['status'] == 'windows_payload_and_cli_version_verified' and setup['nsys_sha256'] == NSYS_SHA256
    tool = {'setup_path': setup_path, 'setup_sha256': hashlib.sha256(setup_raw).hexdigest(),
            'path': 'tools/n26/target-windows-x64/nsys.exe', 'sha256': NSYS_SHA256, 'version': setup['version']}
    tests = ['test2026.cuda.qwen38_5090.' + name for name in (
        'test_native_cuda_phases', 'test_native_cuda_phase_study', 'test_native_cuda_profiled_driver', 'test_native_host_stage_trace',
        'test_native_host_stage_probe', 'test_native_host_stage_admission',
        'test_fixed_length_worker_contract', 'test_native_host_stage_worker_io')]
    assert all('Garnet/' + module.replace('.', '/') + '.py' in files for module in tests)
    sys.path.insert(0, str(methods))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(tests)
    names = test_ids(suite)
    assert not loader.errors and len(names) == len(set(names))
    result = {'status': 'native_cuda_phases_prepared_gpu_unrun', 'files': files,
              'runtime_dependencies': w['runtime_dependencies'], 'test_modules': tests,
              'test_ids': names, 'expected_cpu_test_count': len(names), 'profiler': tool,
              'profiled_driver_cpu_smoke_sha256': sha(methods / 'evidence/native-cuda-phase-profiled-driver-smoke-report.json'),
              'host_parent': {'job': host, 'archive_sha256': proof['archive_sha256'], 'proof_sha256': hashlib.sha256(raw).hexdigest()},
              'scope': 'Exclusive native Windows B1/full4096 ordinary and originalT4MTP CUDA diagnosis after wider-MTP HTTP terminal closure and replay. Existing1a1d source/runtime/checkpoint unchanged. Four actual retained3968+128 text/image controls,three orders eachmode,24requests; all IDs/imagebytes and raw worker/host intervals replayed. Nsight2026 node collection from launch;144 unique HtoD markers permode bind request/phase windows, never translate Python and Nsight clocks. First-use/late-repeat delays retained. Clipped kernel/copy/memset unions and separate API durations include profiler/marker overhead; do not attribute inactive gaps or claim uninstrumented speed/HTTP/2x. Fixed root, pinned CPU test IDs, owned cleanup/cache retirement and streamed raw archive.'}
    target = methods / 'evidence/native-cuda-phase-run-preparation.json'
    data = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == data
    if not target.exists():
        target.write_bytes(data)
    uploads = [{'local': str(methods / name), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + name, 'sha256': sha(methods / name)} for name in METHODS]
    uploads.append({'local': str(target), 'target': 'Garnet/test2026/cuda/qwen38_5090/evidence/' + target.name, 'sha256': sha(target)})
    data = json.dumps({'uploads': uploads, 'scope': result['scope']}, indent=2).encode()
    assert not args.plan.exists() or args.plan.read_bytes() == data
    if not args.plan.exists():
        args.plan.write_bytes(data)
    print(json.dumps({'status': result['status'], 'preparation_sha256': sha(target), 'pinned_files': len(files),
                      'upload_targets': len(uploads), 'cpu_tests': len(names), 'gpu_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
