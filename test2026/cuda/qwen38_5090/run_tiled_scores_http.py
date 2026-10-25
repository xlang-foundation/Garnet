"""Own one exclusive tiled-kernel HTTP trial and restore the baseline DLL."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_tiled_scores_http_admission import BASELINE, CANDIDATE, MODEL, require
from test2026.cuda.qwen38_5090.prefill16_http_study import closed_host_diagnostic
from test2026.cuda.qwen38_5090.tiled_scores_http_study import validate_lifecycle

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
ROOT = Path('D:/CantorAI/5090-study-20261023')
JOB = 'tiled-scores-fixed4k-http-attempt1'
FLAG = 'GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-host-archive-sha256', required=True)
    args = parser.parse_args()
    methods = Path(__file__).resolve().parent
    assert methods == (ROOT / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    sys.path.insert(0, str(methods))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    prep = methods / 'evidence/tiled-scores-http-preparation.json'
    p = json.loads(prep.read_bytes())
    assert p['status'] == 'tiled_scores_http_prepared_gpu_unrun'
    for name, digest in p['files'].items():
        assert sha(ROOT / name) == digest, ('HTTP source/input drift', name)
    for name, digest in p['runtime_baseline'].items():
        assert sha(ROOT / 'build/bin' / name) == digest
    for name, digest in p['environment'].items():
        assert sha(Path(name)) == digest
    library = ROOT / 'build/bin/garnet.dll'
    baseline = ROOT / 'build/bin/garnet-native-tokenizer-nfc-attempt1.dll'
    candidate = ROOT / 'build/bin/garnet-native-bf16-scores-tiled-api-attempt1.dll'
    assert sha(library) == sha(baseline) == BASELINE and sha(candidate) == CANDIDATE
    for name, entry in p['parents'].items():
        proof_path = ROOT / 'results' / (name + '-verified.json')
        proof = json.loads(proof_path.read_bytes())
        assert sha(proof_path) == entry['proof_sha256'] and proof['status'] == entry['status']
        assert sha(ROOT / 'results' / (name + '-evidence.zip')) == proof['archive_sha256'] == entry['archive_sha256']
        closed = json.loads((ROOT / 'results' / name / 'workflow.json').read_bytes())
        assert type(closed['exit_code']) is int and closed['exit_code'] == 0 and closed['cleanup_finished'] is True
    host = closed_host_diagnostic(ROOT, args.closed_host_archive_sha256)
    parent = 'fixed-length-http-study-attempt1'
    receipt = ROOT / 'results/study-ssd-engine-cache/receipt.json'
    scratch = json.loads(receipt.read_bytes())
    cache = Path(scratch['cache_root'])
    assert scratch['status'] == 'ssd_engine_scratch_prepared' and cache.resolve().drive.upper() == 'C:'
    assert json.loads((cache / 'study-owner.json').read_bytes())['study'] == str(ROOT)
    assert shutil.disk_usage(cache).free >= 96 * 1024**3
    out = ROOT / 'results' / JOB
    archive = ROOT / 'results' / (JOB + '-evidence.zip')
    assert not out.exists() and not archive.exists()
    idle()
    with socket.socket() as endpoint:
        endpoint.bind(('127.0.0.1', 18082))
    out.mkdir()
    os.chdir(ROOT / 'Garnet')
    before = {path.name for path in cache.iterdir()}
    started = time.time()
    code = 1
    old_flag = os.environ.get(FLAG)
    common = [ROOT / name for name in p['files']] + [prep, receipt,
        ROOT / 'results' / host['job'] / 'workflow.json', ROOT / 'results' / (host['job'] + '-verified.json')]
    common += [ROOT / 'results' / (name + '-verified.json') for name in p['parents']]
    w = {'status': 'running', 'preparation_sha256': sha(prep), 'source_sha256': p['files'],
         'runtime_baseline': p['runtime_baseline'], 'runtime_dependencies': p['runtime_dependencies'],
         'library_sha256': CANDIDATE, 'environment_before': p['environment'],
         'parent_proof_sha256': p['parent_proof_sha256'], 'parent_archive_sha256': p['parent_archive_sha256'],
         'native_parent': p['native_parent'], 'full4k_parent': p['full4k_parent'], 'tiled_model_parent': p['tiled_model_parent'],
         'admission': None, 'closed_host_diagnostic': host, 'candidate_library_installed': False,
         'selector_before': old_flag, 'cleanup_finished': False, 'performance_admitted': False, 'scope': p['scope']}
    def save():
        (out / 'workflow.json').write_bytes(json.dumps(w, indent=2).encode())
    save()
    try:
        code = run([sys.executable, '-m', 'unittest', *p['test_ids']], out / 'cpu-ledger', 'correctness',
                   'Pinned existing protocol/quality/SSE/memory controls plus explicit tiled proof and baseline restoration faults', common)
        if code:
            raise RuntimeError('Tiled HTTP CPU controls failed')
        library.write_bytes(candidate.read_bytes())
        assert sha(library) == CANDIDATE
        w['candidate_library_installed'] = True
        os.environ[FLAG] = '1'
        w['admission'] = require(ROOT, 'mtp', sha(library))
        save()
        command = [sys.executable, str(methods / 'monitor_native_probe.py'), '--study', str(ROOT),
                   '--output', str(out / 'mtp-resources'), '--', sys.executable,
                   str(methods / 'probe_tiled_scores_http.py'), '--root', str(ROOT), '--output', str(out / 'mtp'),
                   '--corpus', str(ROOT / 'tools/mmmu-quality-corpus'), '--mode', 'mtp',
                   '--ordinary-controls', str(ROOT / 'results' / parent / 'ordinary')]
        code = run(command, out / 'mtp-ledger', 'correctness',
                   'Actual12protocol/805natural-stop ordinary-ID checks/two4Kboundaries then124fixed3968+128 streams; exact candidate API checked before model loading', common)
        if code:
            raise RuntimeError('Actual tiled HTTP quality or benchmark failed')
        report = json.loads((out / 'mtp/report.json').read_bytes())
        assert report['status'] == 'fixed_length_http_profile_collected' and report['fixed_context_benchmark_collected'] is True
        assert report['ordinary_mtp_screening_token_mismatches'] == []
        assert json.loads((out / 'mtp/worker-exit.json').read_bytes())['exit_code'] == 0
        w['status'] = 'tiled_scores_fixed4k_http_collected_needs_replay'
    except BaseException as error:
        w.update(status='failed', error=repr(error))
        code = code or 1
    finally:
        idle()
        w['runtime_after_before_restore'] = {name: sha(ROOT / 'build/bin' / name) for name in p['runtime_dependencies']}
        library.write_bytes(baseline.read_bytes())
        if old_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old_flag
        w.update(exit_code=code, source_after_sha256={name: sha(ROOT / name) for name in p['files']},
                 restored_runtime={name: sha(ROOT / 'build/bin' / name) for name in p['runtime_baseline']},
                 baseline_library_restored=sha(library) == BASELINE, selector_after=os.environ.get(FLAG),
                 environment_after={name: sha(Path(name)) for name in p['environment']})
        w['retired_private_engine_files'] = retire(cache, ROOT, before, started)
        w['cleanup_finished'] = True
        save()
        # Preserve failures even when identity checks fail; replay will reject promotion.
        paths = [path for path in out.rglob('*') if path.is_file()] + common
        names = {path.relative_to(ROOT).as_posix(): path for path in paths}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, path in names.items():
                z.write(path, name)
            z.writestr('evidence-manifest.json', json.dumps({'files': {name: sha(path) for name, path in names.items()},
                'parent_archive_sha256': p['parent_archive_sha256'], 'scope': p['scope']}, indent=2))
        print(json.dumps({'status': w['status'], 'archive_sha256': sha(archive), 'bytes': archive.stat().st_size, 'files': len(names)}), flush=True)
        validate_lifecycle(w, p)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
