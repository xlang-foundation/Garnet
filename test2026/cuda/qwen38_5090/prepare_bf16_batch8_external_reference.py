"""Bind actual two-machine B8 preservation proof before preparing a fresh full-cache B4 reference."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import sys
import unittest
import zipfile

METHODS = ('bf16_batch8_reference_gate.py', 'test_bf16_batch8_reference_gate.py', 'bf16_batch8_external_reference.py', 'test_bf16_batch8_external_reference.py', 'probe_bf16_batch8_external_reference.py', 'test_bf16_batch8_external_reference.bat', 'run_bf16_batch8_external_reference.py', 'verify_bf16_batch8_external_reference.py', 'test_bf16_batch8_external_pipeline.py', 'prepare_bf16_batch8_external_reference.py', 'evidence/bf16-batch8-external-reference-pipeline-derivation.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.proofs = args.proofs.resolve(strict=True)
    args.work = args.work.resolve(strict=True)
    methods = Path(__file__).resolve().parent
    repo = methods.parents[2]
    sys.path[:0] = [str(repo), str(methods)]
    from test2026.cuda.qwen38_5090.bf16_batch8_external_reference import PARENT, BASELINE, validate_parent
    from test2026.cuda.qwen38_5090.run_logged import run
    path = args.proofs / (PARENT + '-verified.json')
    proof = json.loads(path.read_bytes())
    assert proof == json.loads((args.proofs / (PARENT + '-remote-verified.json')).read_bytes())
    archive = args.proofs / (PARENT + '-evidence.zip')
    digest = sha(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            assert sha(z.read(name)) == expected
        w = json.loads(z.read('results/' + PARENT + '/workflow.json'))
    validate_parent(proof, w, BASELINE, digest)
    files = dict(w['source_after_sha256'])
    new = {'Garnet/test2026/cuda/qwen38_5090/' + name: sha((methods / name).read_bytes()) for name in METHODS}
    assert not set(new) & set(files)
    files.update(new)
    candidate = json.loads((methods / 'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for name, expected in files.items():
        if name.startswith('Garnet/'):
            relative = name.removeprefix('Garnet/')
            assert sha((repo / relative).read_bytes()) == candidate.get(relative, expected), ('Baseline/candidate source drift', name)
    for name in METHODS:
        if name.endswith('.py'):
            ast.parse((methods / name).read_bytes())
    modules = ['test2026.cuda.qwen38_5090.' + name for name in ('test_bf16_batch8_admission', 'test_bf16_batch8_replay',
               'test_paged_batch8_model', 'test_paged_prompt_rows', 'test_paged_prompt_component',
               'test_bf16_same_precision_gate', 'test_bf16_batch8_reference_gate', 'test_bf16_batch8_external_reference', 'test_bf16_batch8_external_pipeline')]
    assert all('Garnet/' + name.replace('.', '/') + '.py' in files for name in modules)
    def ids(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite): yield from ids(test)
            else: yield test.id()
    loader = unittest.TestLoader()
    tests = sorted(ids(loader.loadTestsFromNames(modules)))
    assert not loader.errors and len(tests) == len(set(tests)) == 43
    os.chdir(repo)
    code = run([sys.executable, '-m', 'unittest', *tests], args.work / 'bf16-batch8-external-reference-cpu-ledger', 'correctness',
               '43existing B8/state/paged controls and strict fresh B4 cache/raw-vector/metadata/parent/lifecycle dependency faults',
               [methods / name for name in METHODS])
    assert code == 0
    p = {'status': 'bf16_batch8_external_reference_prepared_gpu_unrun', 'files': files, 'new_files': new,
         'runtime': w['runtime_dependencies_after'], 'environment': w['environment_after'],
         'parents': {PARENT: {'status': proof['status'], 'proof_sha256': sha(path.read_bytes()), 'archive_sha256': digest}},
         'test_ids': tests, 'expected_cpu_test_count': len(tests), 'gpu_uuid': 'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6',
         'scope': 'Actual fresh ordinary B4/T32prefill/T4decode full-FP32-cache with explicit BF16-value roundtrip using baseline1a1d and bounded128/two-page input. Require actual two-machine closed paged B8 parent, all nine same-input cases split into18four-row references;58complete logical captures/116raw finite vectors/all IDs exact to original B8. Existing guards and shared projections/linear arithmetic unchanged. This isolates cache representation/batch geometry only, not independent framework or B1/all-model math. Actual full-cache model/vision must fit; preserve failures. Source/runtime/SDK unchanged, raw CPU/GPU ledgers/memory samples/gaps/cache retirement and independent replay mandatory. No MTP/805quality/HTTP/timing/performance admission; serving peaks, strongest tuned references, matched vLLM4K/repetitions/INT4/4B and overall2x remain open.'}
    target = methods / 'evidence/bf16-batch8-external-reference-preparation.json'
    raw = json.dumps(p, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists(): target.write_bytes(raw)
    rows = [{'local': str(methods / name), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + name,
             'sha256': sha((methods / name).read_bytes())} for name in METHODS]
    rows.append({'local': str(target), 'target': 'Garnet/test2026/cuda/qwen38_5090/evidence/' + target.name, 'sha256': sha(raw)})
    (args.work / 'bf16-batch8-external-reference-upload-plan.json').write_bytes(json.dumps({'uploads': rows}, indent=2).encode())
    print(json.dumps({'status': p['status'], 'source_pins': len(files), 'cpu_tests': 43, 'new_targets': len(rows), 'preparation_sha256': sha(raw)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
