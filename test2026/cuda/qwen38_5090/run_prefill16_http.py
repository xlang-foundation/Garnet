"""Own the wider-MTP805quality/fixed4K HTTP trial after host diagnosis closes."""
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
from test2026.cuda.qwen38_5090.native_prefill16_http_admission import require
from test2026.cuda.qwen38_5090.prefill16_http_study import closed_host_diagnostic
from test2026.cuda.qwen38_5090.run_bf16_batch8_admission import retire

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-host-archive-sha256', required=True)
    args = parser.parse_args()
    root = Path('D:/CantorAI/5090-study-20261023')
    methods = Path(__file__).resolve().parent
    assert methods == (root / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    sys.path.insert(0, str(methods))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    preparation = methods / 'evidence/prefill16-http-run-preparation.json'
    p = json.loads(preparation.read_bytes())
    assert p['status'] == 'prefill16_http_prepared_gpu_unrun'
    assert all('Garnet/' + module.replace('.', '/') + '.py' in p['files'] for module in p['test_modules'])
    for name, digest in p['files'].items():
        assert sha(root / name) == digest, ('HTTP method/source/input drift', name)
    for name, digest in p['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / name) == digest
    library = root / 'build/bin/garnet.dll'
    admission = require(root, 'mtp', sha(library))
    assert admission['full4k_archive_sha256'] == p['full4k_parent']['archive_sha256']
    assert admission['full4k_proof_sha256'] == p['full4k_parent']['proof_sha256']
    host = closed_host_diagnostic(root, args.closed_host_archive_sha256)
    parent = 'fixed-length-http-study-attempt1'
    proof_path = root / 'results' / (parent + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    assert sha(proof_path) == p['native_parent']['proof_sha256']
    assert proof['status'] == 'fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed'] is True
    assert sha(root / 'results' / (parent + '-evidence.zip')) == proof['archive_sha256'] == p['native_parent']['archive_sha256']
    closed = json.loads((root / 'results' / parent / 'workflow.json').read_bytes())
    assert closed['cleanup_finished'] is True and type(closed['exit_code']) is int and closed['exit_code'] == 0
    assert closed['source_sha256'] == closed['source_after_sha256'] and closed['runtime_dependencies'] == closed['runtime_dependencies_after']
    receipt = root / 'results/study-ssd-engine-cache/receipt.json'
    scratch = json.loads(receipt.read_bytes())
    cache = Path(scratch['cache_root'])
    assert scratch['status'] == 'ssd_engine_scratch_prepared' and cache.resolve().drive.upper() == 'C:'
    assert json.loads((cache / 'study-owner.json').read_bytes())['study'] == str(root)
    assert shutil.disk_usage(cache).free >= 96 * 1024**3
    job = 'prefill16-fixed4k-http-attempt1'
    out = root / 'results' / job
    archive = root / 'results' / (job + '-evidence.zip')
    assert not out.exists() and not archive.exists()
    idle()
    with socket.socket() as endpoint:
        endpoint.bind(('127.0.0.1', 18082))
    out.mkdir()
    os.chdir(root / 'Garnet')
    code = 1
    before = {path.name for path in cache.iterdir()}
    started = time.time()
    common = [root / name for name in p['files']] + [preparation, receipt, proof_path,
        root / 'results' / host['job'] / 'workflow.json', root / 'results' / (host['job'] + '-verified.json'),
        root / 'results/mtp-prefill16-full4k-admission-attempt1-verified.json']
    workflow = {'status': 'running', 'preparation_sha256': sha(preparation), 'source_sha256': p['files'],
                'runtime_dependencies': p['runtime_dependencies'], 'library_sha256': sha(library),
                'parent_proof_sha256': p['parent_proof_sha256'], 'parent_archive_sha256': p['parent_archive_sha256'],
                'native_parent': p['native_parent'], 'full4k_parent': p['full4k_parent'],
                'admission': admission, 'closed_host_diagnostic': host, 'cleanup_finished': False,
                'performance_admitted': False, 'scope': p['scope']}
    def save():
        (out / 'workflow.json').write_text(json.dumps(workflow, indent=2), encoding='utf8')
    save()
    try:
        test_code = ('import sys,unittest;sys.path.insert(0,' + repr(str(methods)) + ');'
                     'unittest.main(module=None,argv=' + repr(['unittest', *p['test_modules']]) + ')')
        code = run([sys.executable, '-c', test_code], out / 'cpu-ledger', 'correctness',
                   'Explicit worker/probe AST, actual full4K proof/host closure and stream/quality fault controls', common)
        if code:
            raise RuntimeError('Candidate HTTP CPU controls failed')
        command = [sys.executable, str(methods / 'monitor_native_probe.py'), '--study', str(root),
                   '--output', str(out / 'mtp-resources'), '--', sys.executable,
                   str(methods / 'probe_prefill16_http.py'), '--root', str(root), '--output', str(out / 'mtp'),
                   '--corpus', str(root / 'tools/mmmu-quality-corpus'), '--mode', 'mtp',
                   '--ordinary-controls', str(root / 'results' / parent / 'ordinary')]
        code = run(command, out / 'mtp-ledger', 'correctness',
                   'Actual12protocol/805natural-stop ordinary-ID checks/two4Kboundaries then124fixed3968+128 streams; no short pilot', common)
        if code:
            raise RuntimeError('Actual candidate HTTP quality or benchmark failed')
        report = json.loads((out / 'mtp/report.json').read_bytes())
        assert report['status'] == 'fixed_length_http_profile_collected' and report['fixed_context_benchmark_collected'] is True
        assert report['ordinary_mtp_screening_token_mismatches'] == []
        assert json.loads((out / 'mtp/worker-exit.json').read_bytes())['exit_code'] == 0
        workflow['status'] = 'prefill16_fixed4k_http_collected_needs_replay'
    except BaseException as error:
        workflow.update(status='failed', error=repr(error))
        code = code or 1
    finally:
        workflow.update(exit_code=code, source_after_sha256={name: sha(root / name) for name in p['files']},
                        runtime_dependencies_after={name: sha(root / 'build/bin' / name) for name in p['runtime_dependencies']},
                        library_unchanged=sha(library) == workflow['library_sha256'])
        save()
        idle()
        workflow['retired_private_engine_files'] = retire(cache, root, before, started)
        workflow['cleanup_finished'] = True
        save()
        paths = [path for path in out.rglob('*') if path.is_file()] + common
        raw = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, data in raw.items():
                z.writestr(name, data)
            z.writestr('evidence-manifest.json', json.dumps({'files': {
                name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
                'parent_archive_sha256': p['parent_archive_sha256'], 'scope': p['scope']}, indent=2))
        print(json.dumps({'status': workflow['status'], 'archive_sha256': sha(archive),
                          'bytes': archive.stat().st_size, 'files': len(raw)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
