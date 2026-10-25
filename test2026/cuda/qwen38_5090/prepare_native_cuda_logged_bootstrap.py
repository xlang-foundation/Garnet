"""Pin logger-only bootstrap to the immutable, replayed failed profile archive."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import sha
from test2026.cuda.qwen38_5090.run_native_cuda_phases import test_ids

NAMES = ('native_cuda_phase_bootstrap_log.py', 'native_cuda_phase_logged_worker.py',
         'serve_native_cuda_phase_logged_worker.bat', 'test_native_cuda_bootstrap_log.py',
         'probe_native_cuda_logged_bootstrap.py', 'run_native_cuda_logged_bootstrap.py',
         'probe_native_cuda_trace_embedded_cpu.py', 'prepare_native_cuda_logged_bootstrap.py',
         'verify_native_cuda_logged_bootstrap.py', 'evidence/native-cuda-phase-logged-worker-derivation.json')
PREFIX = 'Garnet/test2026/cuda/qwen38_5090/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    name = 'native-cuda-phase-full4k-attempt1'
    proof_path = args.proofs / (name + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    assert proof == json.loads((args.proofs / (name + '-remote-verified.json')).read_bytes())
    assert proof['status'] == 'native_cuda_phases_full4k_failure_preserved'
    archive = args.proofs / (name + '-evidence.zip')
    assert sha(archive) == proof['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for path, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == expected
        w = json.loads(z.read('results/' + name + '/workflow.json'))
    assert type(w['exit_code']) is int and w['exit_code'] == 1 and w['cleanup_finished'] is True
    assert w['source_sha256'] == w['source_after_sha256'] and w['runtime_dependencies'] == w['runtime_dependencies_after']
    assert w['library_unchanged'] is True
    files = dict(w['source_sha256'])
    new = {PREFIX + path: sha(methods / path) for path in NAMES}
    assert not (set(new) & set(files))
    files.update(new)
    for path, digest in files.items():
        if path.startswith('Garnet/'):
            assert sha(repo / path.removeprefix('Garnet/')) == digest, ('Original/candidate source drift', path)
    tests = ['test2026.cuda.qwen38_5090.test_native_cuda_bootstrap_log', 'test2026.cuda.qwen38_5090.test_native_cuda_phases']
    assert all('Garnet/' + module.replace('.', '/') + '.py' in files for module in tests)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(tests)
    ids = test_ids(suite)
    assert not loader.errors and len(ids) == len(set(ids)) == 13
    p = {'status': 'native_cuda_logged_bootstrap_prepared_unrun', 'files': files, 'new_files': new,
         'runtime_dependencies': w['runtime_dependencies'], 'parent_archive_sha256': proof['archive_sha256'],
         'parent_proof_sha256': sha(proof_path), 'test_modules': tests, 'test_ids': ids, 'expected_cpu_tests': 13,
         'scope': 'One exclusive unprofiled bootstrap after original failed profile terminal closure and identical two-machine failure replay. Unchanged1a1d model/default/runtime/SDK/source633, explicit logger subclass only. Actual13CPU guards and embedded-Python fake-IO compatibility precede native model construction. One actual retained3968+128text control with original IDs; persistent bootstrap and Python exception logs, actual worker close, host intervals/12synthetic-copy-request records and owned cache retirement. Unprofiled markers are requests, not GPU timing evidence. No HTTP benchmark, new numerical-kernel admission, quality parity or2x claim. Base source remains linked to original archive rather than duplicated.'}
    target = methods / 'evidence/native-cuda-logged-bootstrap-preparation.json'
    raw = json.dumps(p, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    uploads = [{'local': str(methods / path), 'target': PREFIX + path, 'sha256': sha(methods / path)} for path in NAMES]
    uploads.append({'local': str(target), 'target': PREFIX + 'evidence/' + target.name, 'sha256': sha(target)})
    raw = json.dumps({'uploads': uploads, 'scope': p['scope']}, indent=2).encode()
    assert not args.plan.exists() or args.plan.read_bytes() == raw
    if not args.plan.exists():
        args.plan.write_bytes(raw)
    print(json.dumps({'status': p['status'], 'preparation_sha256': sha(target), 'source_files': len(files), 'new_targets': len(uploads), 'cpu_tests': 13}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
