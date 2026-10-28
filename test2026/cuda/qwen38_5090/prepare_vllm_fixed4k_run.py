"""Pin a matched ordinary vLLM4K trial to closed native and SGLang evidence."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import unittest
import zipfile

METHODS = ('reference_fixed4k_vllm_identity.py', 'test_reference_fixed4k_vllm_identity.py',
           'inspect_vllm_fixed4k_config_linux.py', 'prepare_vllm_fixed4k_metadata.py',
           'reference_fixed4k_vllm_boundary.py', 'probe_vllm_fixed4k.py', 'run_vllm_fixed4k.py',
           'verify_vllm_fixed4k.py', 'test_vllm_fixed4k_owner.py', 'test_vllm_fixed4k_pipeline.py',
           'prepare_vllm_fixed4k_run.py', 'evidence/vllm-fixed4k-config-schema-attempt1.json',
           'evidence/vllm-fixed4k-pipeline-derivation.json', 'serve_vllm_fixed4k.sh',
           'verify_vllm_fixed4k_launcher_completed.py', 'test_vllm_fixed4k_dependencies.py',
           'evidence/vllm-fixed4k-launcher-completed-verifier-derivation.json',
           'evidence/vllm-fixed4k-run-preparation.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
REQUIRED_RUNTIME_FILES = ('serve_vllm_fixed4k.sh', 'run_vllm_fixed4k.py', 'probe_vllm_fixed4k.py',
                          'prepare_vllm_fixed4k_metadata.py', 'inspect_reference_fixed4k_linux.py',
                          'inspect_vllm_fixed4k_config_linux.py', 'benchmark_fixed_length_reference.py',
                          'stop_reference_server.py', 'vllm_cache_control.py', 'reference_fixed4k_vllm_identity.py',
                          'reference_fixed4k_vllm_boundary.py', 'verify_vllm_fixed4k_launcher_completed.py',
                          'evidence/vllm-fixed4k-run-preparation.json')


def require_runtime_dependencies(files):
    for name in REQUIRED_RUNTIME_FILES:
        key = 'Garnet/test2026/cuda/qwen38_5090/' + name
        if key not in files:
            raise ValueError('Runtime dependency is not pinned: ' + name)
        if not isinstance(files[key], str) or re.fullmatch('[0-9a-f]{64}', files[key]) is None:
            raise ValueError('Invalid runtime dependency digest: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.proofs = args.proofs.resolve(strict=True)
    args.work = args.work.resolve(strict=True)
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    inherited_path = methods / 'evidence/paged-batch8-model-preparation.json'
    inherited = json.loads(inherited_path.read_bytes())
    files = dict(inherited['files'])
    parents = {}
    for name, status in (('fixed-length-http-study-attempt1', 'fixed_length_http_study_replayed'),
                         ('sglang-observed-fixed4k-ordinary-attempt1', 'reference_fixed4k_observed_profile_replayed')):
        path = args.proofs / (name + '-verified.json')
        raw = path.read_bytes()
        q = json.loads(raw)
        assert q == json.loads((args.proofs / (name + '-remote-verified.json')).read_bytes())
        archive = args.proofs / (name + '-evidence.zip')
        assert q['status'] == status and sha(archive.read_bytes()) == q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('evidence-manifest.json'))
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
            for relative, digest in manifest['files'].items():
                assert sha(z.read(relative)) == digest
            w = json.loads(z.read('results/' + name + '/workflow.json'))
            assert type(w['exit_code']) is int and w['exit_code'] == 0 and w['cleanup_finished'] is True
            assert w['source_sha256'] == w['source_after_sha256'] and w['runtime_dependencies'] == w['runtime_dependencies_after'] == inherited['runtime']
            if name.startswith('sglang'):
                assert type(w['server_exit_code']) is int
                for relative, digest in w['source_after_sha256'].items():
                    assert relative not in files or files[relative] == digest
                    files[relative] = digest
        parents[name] = {'status': status, 'proof_sha256': sha(raw), 'archive_sha256': q['archive_sha256']}
    schema = methods / 'evidence/vllm-fixed4k-config-schema-attempt1.json'
    assert sha(schema.read_bytes()) == 'a8b9431b47317d00341945e79ad5110ae4e95afba77735a6b32839e6f56a4229'
    new = {'Garnet/test2026/cuda/qwen38_5090/' + name: sha((methods / name).read_bytes()) for name in METHODS}
    assert not set(new) & set(files)
    files.update(new)
    require_runtime_dependencies(files)
    candidate = json.loads((methods / 'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for name, digest in files.items():
        if name.startswith('Garnet/'):
            relative = name.removeprefix('Garnet/')
            assert sha((repo / relative).read_bytes()) == candidate.get(relative, digest), ('Pinned source changed', name)
    for name in METHODS:
        if name.endswith('.py'):
            ast.parse((methods / name).read_bytes())
    sys.path[:0] = [str(repo), str(methods)]
    from test2026.cuda.qwen38_5090.prepare_sglang_fixed4k_run import TEST_MODULES
    from test2026.cuda.qwen38_5090.run_logged import run
    modules = [*TEST_MODULES, *('test2026.cuda.qwen38_5090.' + name for name in
                              ('test_reference_fixed4k_vllm_identity', 'test_vllm_fixed4k_owner', 'test_vllm_fixed4k_pipeline', 'test_vllm_fixed4k_dependencies'))]
    assert all('Garnet/' + name.replace('.', '/') + '.py' in files for name in modules)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(modules)
    assert not loader.errors
    count = suite.countTestCases()
    os.chdir(repo)
    code = run([sys.executable, '-m', 'unittest', *modules], args.work / 'vllm-fixed4k-cpu-ledger', 'correctness',
               'All retained fixed-length, quality, stream/cache, ownership controls plus actual vLLM config and exact new boundary/probe/owner/checker derivation faults',
               [methods / name for name in METHODS])
    assert code == 0
    native = parents['fixed-length-http-study-attempt1']
    p = {'status': 'vllm_fixed4k_run_prepared_unrun', 'preparation_version': 'launcher_dependencies_complete_v2',
         'replay_verifier': 'verify_vllm_fixed4k_launcher_completed.py', 'files': files, 'new_files': new,
         'runtime_dependencies': inherited['runtime'], 'environment': inherited['environment'],
         'inherited_preparation_sha256': sha(inherited_path.read_bytes()), 'parents': parents,
         'native_proof_sha256': native['proof_sha256'], 'native_archive_sha256': native['archive_sha256'],
         'test_modules': modules, 'expected_cpu_tests': count,
         'scope': 'One exclusive ordinary vLLM0.32.0 WSL baseline on the same RTX5090/checkpoint, actual3968+128 workload and frozen124requests. Existing launch recipe BF16KV/F32SSM/modelopt_mixed/full-decode graphs1/2/4/8, TP1 and max8running sequences; actual JSON config/version, resolved8full4K cache capacity and exact installed sources/PID must pass after readiness. Two early full4K boundaries, fourVLM/16short actual-ID controls, fresh805adapted screen, all124serial4KID controls, then124raw warmup/timed streams in eight text/image C1/2/4/8cohorts. Existing dual idle processor/encoder reset and zero cache-hit delta; full counts/usage/text and memory samples/gaps retained. Common streamed individual IDs remain unobserved; no token reconstruction. Preserve failures/output variants; strict profile failure requires diagnosis, never erase/cherry-pick. Same source/runtime/SDK, owned PID shutdown/actual exit and independent raw replay mandatory. No GPU/server/benchmark yet. Native comparison, strongest MTP/tuning, broad quality parity, repeat statistics, INT4/4B and overall2x remain open.'}
    target = methods / 'evidence/vllm-fixed4k-launcher-completed-preparation.json'
    raw = json.dumps(p, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists(): target.write_bytes(raw)
    rows = [{'local': str(methods / name), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + name,
             'sha256': sha((methods / name).read_bytes())} for name in METHODS]
    rows.append({'local': str(target), 'target': 'Garnet/test2026/cuda/qwen38_5090/evidence/' + target.name, 'sha256': sha(raw)})
    (args.work / 'vllm-fixed4k-launcher-completed-upload-plan.json').write_bytes(json.dumps({'uploads': rows}, indent=2).encode())
    print(json.dumps({'status': p['status'], 'source_pins': len(files), 'new_targets': len(rows),
                      'cpu_controls': count, 'preparation_sha256': sha(raw), 'gpu_or_server_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
