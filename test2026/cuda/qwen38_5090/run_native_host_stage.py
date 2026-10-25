"""Own an exclusive B1 full4K host diagnostic after the reference has closed."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_host_stage_admission import closed_observed_reference
from test2026.cuda.qwen38_5090.run_sglang_observed_fixed4k import closed_native_full4k
from test2026.cuda.qwen38_5090.run_bf16_batch8_admission import retire

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-reference-job', required=True)
    parser.add_argument('--closed-reference-archive-sha256', required=True)
    parser.add_argument('--closed-native-archive-sha256', required=True)
    args = parser.parse_args()
    root = Path('D:/CantorAI/5090-study-20261023')
    methods = Path(__file__).resolve().parent
    assert methods == (root / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    sys.path.insert(0, str(methods))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    preparation = methods / 'evidence/native-host-stage-run-preparation.json'
    p = json.loads(preparation.read_bytes())
    assert p['status'] == 'native_host_stage_prepared_gpu_unrun'
    for name, digest in p['files'].items():
        assert sha(root / name) == digest, ('Host diagnostic input drift', name)
    for name, digest in p['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / name) == digest
    reference = closed_observed_reference(root, args.closed_reference_job, args.closed_reference_archive_sha256)
    native = closed_native_full4k(root, args.closed_native_archive_sha256)
    assert native['archive_sha256'] == p['full4k_parent']['archive_sha256']
    assert native['proof_sha256'] == p['full4k_parent']['proof_sha256']
    receipt = root / 'results/study-ssd-engine-cache/receipt.json'
    scratch = json.loads(receipt.read_bytes())
    cache = Path(scratch['cache_root'])
    assert scratch['status'] == 'ssd_engine_scratch_prepared' and cache.resolve().drive.upper() == 'C:'
    assert json.loads((cache / 'study-owner.json').read_bytes())['study'] == str(root)
    assert shutil.disk_usage(cache).free >= 96 * 1024**3
    job = 'native-host-stage-full4k-attempt1'
    out = root / 'results' / job
    archive = root / 'results' / (job + '-evidence.zip')
    assert not out.exists() and not archive.exists()
    idle()
    out.mkdir()
    os.chdir(root / 'Garnet')
    code = 1
    before = {f.name for f in cache.iterdir()}
    started = time.time()
    library = root / 'build/bin/garnet.dll'
    common = [root / name for name in p['files']] + [preparation, receipt]
    for parent in (native, reference):
        common.extend([root / 'results' / parent['job'] / 'workflow.json',
                       root / 'results' / (parent['job'] + '-verified.json')])
    workflow = {'status': 'running', 'preparation_sha256': sha(preparation),
                'source_sha256': p['files'], 'runtime_dependencies': p['runtime_dependencies'],
                'library_sha256': sha(library), 'closed_reference': reference,
                'closed_native_full4k': native, 'cleanup_finished': False,
                'performance_admitted': False, 'scope': p['scope']}
    def save():
        (out / 'workflow.json').write_text(json.dumps(workflow, indent=2), encoding='utf8')
    save()
    try:
        test_code = ('import sys,unittest;sys.path.insert(0,' + repr(str(methods)) + ');'
                     'unittest.main(module=None,argv=' + repr(['unittest', *p['test_modules']]) + ')')
        code = run([sys.executable, '-c', test_code], out / 'cpu-ledger',
                   'correctness', 'Whole-worker AST, actual-event stage arithmetic and closed-reference guards', common)
        if code:
            raise RuntimeError('Host diagnostic CPU controls failed')
        command = [sys.executable, str(methods / 'monitor_native_probe.py'), '--study', str(root),
                   '--output', str(out / 'resources'), '--', sys.executable,
                   str(methods / 'probe_native_host_stage.py'), '--root', str(root),
                   '--output', str(out / 'native'), '--cache', str(cache)]
        code = run(command, out / 'gpu-ledger', 'correctness',
                   '24 actual retained3968+128 B1 requests, instrumented flat host phases, no HTTP timing', common)
        if code:
            raise RuntimeError('Actual host diagnostic failed')
        report = json.loads((out / 'native/report.json').read_bytes())
        assert report['status'] == 'native_host_stage_diagnostic_collected_needs_independent_replay'
        workflow['status'] = 'native_host_stage_collected_needs_replay'
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
        paths = [f for f in out.rglob('*') if f.is_file()] + common
        raw = {f.relative_to(root).as_posix(): f.read_bytes() for f in paths}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, data in raw.items():
                z.writestr(name, data)
            z.writestr('evidence-manifest.json', json.dumps({
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
                'scope': p['scope']}, indent=2))
        print(json.dumps({'status': workflow['status'], 'archive_sha256': sha(archive),
                          'bytes': archive.stat().st_size, 'files': len(raw)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
