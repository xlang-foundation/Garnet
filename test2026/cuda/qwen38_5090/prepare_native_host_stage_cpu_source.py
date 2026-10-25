"""Freeze every supplemental test source and the unchanged original verifier."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import unittest

sha = lambda raw: hashlib.sha256(raw).hexdigest()
NEW = ('native_host_stage_cpu_source.py', 'run_native_host_stage_cpu_source.py',
       'verify_native_host_stage_cpu_source.py', 'prepare_native_host_stage_cpu_source.py',
       'test_native_host_stage_cpu_source.py', 'test_native_host_stage_worker_io.py',
       'evidence/native-host-stage-cpu-source-verifier-derivation.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    sys.path[:0] = [str(repo), str(methods)]
    from test2026.cuda.qwen38_5090.run_native_host_stage_cpu_source import ids
    original_path = methods / 'evidence/native-host-stage-run-preparation.json'
    original = json.loads(original_path.read_bytes())
    names = ('test_native_host_stage_trace', 'test_native_host_stage_probe',
             'test_native_host_stage_admission', 'test_fixed_length_worker_contract',
             'test_native_host_stage_worker_io')
    test_files = ['Garnet/test2026/cuda/qwen38_5090/' + name + '.py' for name in names]
    for path in test_files[:-1]:
        assert sha((repo / path.removeprefix('Garnet/')).read_bytes()) == original['files'][path]
    sources = list(NEW) + [name + '.py' for name in names[:-1]] + ['native_worker_io.py', 'verify_native_host_stage.py']
    files = {'Garnet/test2026/cuda/qwen38_5090/' + name: sha((methods / name).read_bytes()) for name in sources}
    modules = ['test2026.cuda.qwen38_5090.' + name for name in names]
    loader = unittest.TestLoader(); suite = loader.loadTestsFromNames(modules)
    test_ids = ids(suite)
    assert not loader.errors and len(test_ids) == len(set(test_ids)) == 29
    p = {'status': 'native_host_stage_cpu_source_supplement_prepared_unrun',
         'files': files, 'test_modules': modules, 'test_ids': test_ids, 'test_source_files': test_files,
         'new_io_source_sha256': files[test_files[-1]], 'expected_cpu_test_count': 29,
         'original_preparation_sha256': sha(original_path.read_bytes()),
         'original_verifier_sha256': sha((methods / 'verify_native_host_stage.py').read_bytes()),
         'scope': 'CPU-only explicit source supplement after original host diagnosis is closed; original27test log and29metadata count mismatch retained. Every29test ID and all5test module files frozen, including new distinct IO5test copy. Original599source/runtime checks before/after; original strict verifier count failure retained, fresh29CPU result has no skips/errors. Old IO source captured post-run only. No model/kernel/GPU execution or numeric/timing tolerance change; separate independent supplemented diagnostic replay still required.'}
    target = methods / 'evidence/native-host-stage-cpu-source-preparation.json'
    raw = json.dumps(p, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists(): target.write_bytes(raw)
    paths = [methods / name for name in sources] + [target]
    uploads = [{'local': str(path), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + path.relative_to(methods).as_posix(),
                'sha256': sha(path.read_bytes())} for path in paths]
    assert len({row['target'] for row in uploads}) == len(uploads)
    raw = json.dumps({'uploads': uploads, 'scope': p['scope']}, indent=2).encode()
    assert not args.plan.exists() or args.plan.read_bytes() == raw
    if not args.plan.exists(): args.plan.write_bytes(raw)
    print(json.dumps({'status': p['status'], 'preparation_sha256': sha(target.read_bytes()),
                      'upload_targets': len(uploads), 'pinned_files': len(files), 'cpu_tests': 29, 'gpu_executed': False}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
