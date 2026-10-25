"""Exclusive logged bootstrap; keep the failed profile immutable and linked."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import sha
from test2026.cuda.qwen38_5090.run_bf16_batch8_admission import retire

JOB = 'native-cuda-phase-logged-bootstrap-attempt1'
PARENT = 'native-cuda-phase-full4k-attempt1'


def main():
    root = Path('D:/CantorAI/5090-study-20261023')
    methods = Path(__file__).resolve().parent
    assert methods == (root / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    sys.path.insert(0, str(methods))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    preparation = methods / 'evidence/native-cuda-logged-bootstrap-preparation.json'
    p = json.loads(preparation.read_bytes())
    parent = root / 'results' / (PARENT + '-evidence.zip')
    proof_path = root / 'results' / (PARENT + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    assert sha(parent) == proof['archive_sha256'] == p['parent_archive_sha256']
    assert sha(proof_path) == p['parent_proof_sha256'] and proof['status'] == 'native_cuda_phases_full4k_failure_preserved'
    closed = json.loads((root / 'results' / PARENT / 'workflow.json').read_bytes())
    assert type(closed['exit_code']) is int and closed['exit_code'] == 1 and closed['cleanup_finished'] is True
    assert closed['source_sha256'] == closed['source_after_sha256'] and closed['runtime_dependencies'] == closed['runtime_dependencies_after']
    assert closed['library_unchanged'] is True
    for name, digest in p['files'].items():
        assert sha(root / name) == digest, ('Logged bootstrap source/input drift', name)
    for name, digest in p['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / name) == digest
    receipt = root / 'results/study-ssd-engine-cache/receipt.json'
    cache = Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper() == 'C:'
    assert json.loads((cache / 'study-owner.json').read_bytes())['study'] == str(root)
    assert shutil.disk_usage(cache).free >= 96 * 1024**3
    out = root / 'results' / JOB
    archive = root / 'results' / (JOB + '-evidence.zip')
    assert not out.exists() and not archive.exists()
    idle()
    out.mkdir()
    os.chdir(root / 'Garnet')
    before = {path.name for path in cache.iterdir()}
    started = time.time()
    code = 1
    w = {'status': 'running', 'source_sha256': p['files'], 'runtime_dependencies': p['runtime_dependencies'],
         'preparation_sha256': sha(preparation), 'parent_archive_sha256': sha(parent), 'parent_proof_sha256': sha(proof_path),
         'library_sha256': sha(root / 'build/bin/garnet.dll'), 'cleanup_finished': False,
         'profiled': False, 'performance_admitted': False, 'scope': p['scope']}
    def save():
        (out / 'workflow.json').write_text(json.dumps(w, indent=2), encoding='utf8')
    save()
    common = [root / name for name in p['new_files']] + [preparation, receipt, proof_path]
    try:
        tests = ['unittest', *p['test_modules']]
        code = run([sys.executable, '-m', *tests], out / 'cpu-ledger', 'correctness', 'Actual13source-pinned bootstrap exception and unchanged model AST controls', common)
        if code:
            raise RuntimeError('Logged bootstrap CPU controls failed')
        code = run([str(root / 'build/bin/xlang3.exe'), str(methods / 'probe_native_cuda_trace_embedded_cpu.py'), str(root), str(out / 'embedded-cpu')],
                   out / 'embedded-cpu-ledger', 'correctness', 'Actual embedded Python subclass/exception compatibility with fake IO, no model', common)
        if code:
            raise RuntimeError('Embedded Python trace CPU compatibility failed before model construction')
        embedded = json.loads((out / 'embedded-cpu/report.json').read_bytes())
        assert embedded['status'] == 'embedded_python_trace_cpu_compatibility_passed' and embedded['gpu_executed'] is False
        code = run([sys.executable, str(methods / 'monitor_native_probe.py'), '--study', str(root), '--output', str(out / 'resources'), '--',
                    sys.executable, str(methods / 'probe_native_cuda_logged_bootstrap.py'), '--root', str(root), '--output', str(out / 'native'), '--cache', str(cache)],
                   out / 'gpu-ledger', 'correctness', 'One actual retained3968+128 native text control without profiler; persistent bootstrap/exception log', common)
        if code:
            raise RuntimeError('Actual logged bootstrap failed; inspect retained native bootstrap log')
        report = json.loads((out / 'native/report.json').read_bytes())
        assert report['status'] == 'native_logged_bootstrap_retained_control_completed'
        w['status'] = 'native_logged_bootstrap_collected_needs_replay'
    except BaseException as error:
        w.update(status='failed', error=repr(error))
        code = code or 1
    finally:
        w.update(exit_code=code, source_after_sha256={name: sha(root / name) for name in p['files']},
                 runtime_dependencies_after={name: sha(root / 'build/bin' / name) for name in p['runtime_dependencies']},
                 library_unchanged=sha(root / 'build/bin/garnet.dll') == w['library_sha256'])
        save()
        idle()
        w['retired_private_engine_files'] = retire(cache, root, before, started)
        w['cleanup_finished'] = True
        save()
        paths = {path.relative_to(root).as_posix(): path for path in [*out.rglob('*'), *common] if path.is_file()}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for name, path in paths.items():
                z.write(path, name)
            z.writestr('evidence-manifest.json', json.dumps({'files': {name: sha(path) for name, path in paths.items()},
                'linked_parent_archive_sha256': sha(parent), 'scope': p['scope']}, indent=2))
        print(json.dumps({'status': w['status'], 'archive_sha256': sha(archive), 'bytes': archive.stat().st_size, 'files': len(paths)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
