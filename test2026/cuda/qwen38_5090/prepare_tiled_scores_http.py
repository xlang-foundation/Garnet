"""Pin the tiled HTTP trial to closed full-model and HTTP evidence."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import sys
import unittest
import zipfile

METHODS = ('native_tiled_scores_http_admission.py', 'native_tiled_scores_stream_worker.py',
           'serve_native_tiled_scores_worker.bat', 'probe_tiled_scores_http.py', 'test_tiled_scores_http_worker.py',
           'tiled_scores_http_study.py', 'run_tiled_scores_http.py', 'verify_tiled_scores_http.py',
           'test_tiled_scores_http_study.py', 'prepare_tiled_scores_http.py',
           'evidence/tiled-scores-http-worker-probe-derivation.json', 'evidence/tiled-scores-http-verifier-derivation.json')
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
    parents = {}
    names = (('fixed-length-http-study-attempt1', 'fixed_length_http_study_replayed'),
             ('mtp-prefill16-full4k-admission-attempt1', 'mtp_prefill16_full4k_target_state_replayed'),
             ('bf16-scores-tiled-api-model-admission-attempt1', 'native_tiled_scores_model_replayed'),
             ('prefill16-fixed4k-http-attempt1', 'prefill16_fixed4k_http_replayed'))
    for name, status in names:
        raw = (args.proofs / (name + '-verified.json')).read_bytes()
        q = json.loads(raw)
        assert q == json.loads((args.proofs / (name + '-remote-verified.json')).read_bytes())
        archive = args.proofs / (name + '-evidence.zip')
        assert q['status'] == status and sha(archive.read_bytes()) == q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('evidence-manifest.json'))
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
            for path, digest in manifest['files'].items():
                assert sha(z.read(path)) == digest
            w = json.loads(z.read('results/' + name + '/workflow.json'))
            assert type(w['exit_code']) is int and w['exit_code'] == 0 and w['cleanup_finished'] is True
            assert w['source_sha256'] == w['source_after_sha256'] and w['runtime_dependencies'] == w['runtime_dependencies_after']
            if name == 'prefill16-fixed4k-http-attempt1':
                old = json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/prefill16-http-run-preparation.json'))
            if name == 'bf16-scores-tiled-api-model-admission-attempt1':
                model_workflow = w
                model_proof = q
        parents[name] = {'status': status, 'archive_sha256': q['archive_sha256'], 'proof_sha256': sha(raw)}
    inherited_preparation = methods / 'evidence/paged-batch8-model-preparation.json'
    inherited = json.loads(inherited_preparation.read_bytes())
    files = dict(inherited['files'])
    # Original HTTP tests/helpers are pinned as a whole, including every executed test module.
    for name, digest in old['files'].items():
        assert name not in files or files[name] == digest
        files[name] = digest
    for name, digest in model_workflow['source_after_sha256'].items():
        assert files[name] == digest
    new = {'Garnet/test2026/cuda/qwen38_5090/' + name: sha((methods / name).read_bytes()) for name in METHODS}
    assert not set(new) & set(files)
    files.update(new)
    candidate_sources = json.loads((methods / 'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for name, digest in files.items():
        if name.startswith('Garnet/'):
            relative = name.removeprefix('Garnet/')
            assert sha((repo / relative).read_bytes()) == candidate_sources.get(relative, digest), ('Qualified source drift', name)
    for name in METHODS:
        if name.endswith('.py'):
            ast.parse((methods / name).read_bytes())
    runtime = inherited['runtime']
    assert runtime == old['runtime_dependencies'] == model_workflow['runtime_dependencies_after']
    assert inherited['environment'] == model_workflow['environment_after']
    sys.path[:0] = [str(repo), str(methods)]
    from test2026.cuda.qwen38_5090.native_tiled_scores_http_admission import BASELINE, CANDIDATE, validate_model
    from test2026.cuda.qwen38_5090.run_logged import run
    validate_model(model_proof, model_workflow, CANDIDATE)
    assert runtime['garnet.dll'] == BASELINE
    modules = old['test_modules'] + ['test2026.cuda.qwen38_5090.' + name for name in
                                   ('test_tiled_scores_http_worker', 'test_tiled_scores_http_study')]
    assert all('Garnet/' + name.replace('.', '/') + '.py' in files for name in modules)
    def ids(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from ids(test)
            else:
                yield test.id()
    loader = unittest.TestLoader()
    test_ids = sorted(ids(loader.loadTestsFromNames(modules)))
    assert not loader.errors and len(test_ids) == len(set(test_ids))
    os.chdir(repo)
    code = run([sys.executable, '-m', 'unittest', *test_ids], args.work / 'tiled-scores-http-cpu-ledger', 'correctness',
               'All retained protocol/quality/SSE/memory controls plus tiled admission, literal worker/probe/checker derivation and restoration failures',
               [methods / name for name in METHODS])
    assert code == 0
    scope = ('One exclusive native Windows B1 candidate2533 tiled-score MTP3/T16draftprefill/T4draftdecode/T32targetprefill HTTP trial. '
             'Actual default/tiled full-model states/raw vectors/IDs admitted against baseline1a1d; old NFC/EOS/full4K proof identities remain baseline. '
             'Require actual12protocol/805natural-stop ordinary-ID controls/two3968+128boundaries before124frozen4Kstreams in eight text/image C1/2/4/8cohorts. '
             'Same existing generation/probe/raw checker arithmetic, no short pilot. Preserve actual943worker completions/raw SSE/timings/VRAM gaps/source/SDK hashes/failures. '
             'Only candidate DLL temporarily installed; always restore baseline DLL and caller selector; owned engine cache retired. '
             'GPUbatch1/queued clients; correctness-resource peaks are not serving memory savings. Separate raw replay required; single trial no confidence intervals or causal speedup. '
             'Matched vLLM4K/tuned MTP references, repetitions, external quality parity, B8HTTP, INT4/4B and overall2x remain open.')
    p = {'status': 'tiled_scores_http_prepared_gpu_unrun', 'files': files, 'new_files': new,
         'inherited_preparation_sha256': sha(inherited_preparation.read_bytes()), 'inherited_source_files': inherited['files'],
         'runtime_baseline': runtime, 'runtime_dependencies': {**runtime, 'garnet.dll': CANDIDATE}, 'environment': inherited['environment'],
         'parents': parents, 'native_parent': {k: parents[names[0][0]][k] for k in ('archive_sha256', 'proof_sha256')},
         'full4k_parent': {k: parents[names[1][0]][k] for k in ('archive_sha256', 'proof_sha256')},
         'tiled_model_parent': parents[names[2][0]], 'test_modules': modules, 'test_ids': test_ids,
         'expected_cpu_test_count': len(test_ids), 'scope': scope}
    for key in ('parent_proof_sha256', 'parent_archive_sha256', 'eos_proof_sha256', 'eos_archive_sha256'):
        p[key] = old[key]
    target = methods / 'evidence/tiled-scores-http-preparation.json'
    raw = json.dumps(p, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    uploads = [{'local': str(methods / name), 'target': 'Garnet/test2026/cuda/qwen38_5090/' + name,
                'sha256': sha((methods / name).read_bytes())} for name in METHODS]
    uploads.append({'local': str(target), 'target': 'Garnet/test2026/cuda/qwen38_5090/evidence/' + target.name, 'sha256': sha(raw)})
    (args.work / 'tiled-scores-http-upload-plan.json').write_bytes(json.dumps({'uploads': uploads}, indent=2).encode())
    print(json.dumps({'status': p['status'], 'cpu_tests': len(test_ids), 'source_pins': len(files),
                      'upload_targets': len(uploads), 'preparation_sha256': sha(raw), 'gpu_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
