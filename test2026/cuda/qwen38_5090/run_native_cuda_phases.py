"""Own a full4K Nsight diagnosis only after the wider-MTP HTTP job closes."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import JOB, closed_trial, profiler, sha
from test2026.cuda.qwen38_5090.prefill16_http_study import closed_host_diagnostic
from test2026.cuda.qwen38_5090.run_bf16_batch8_admission import retire


def test_ids(suite):
    if isinstance(suite, unittest.TestSuite):
        return [name for child in suite for name in test_ids(child)]
    return [suite.id()]


def cpu_child(methods, output):
    p = json.loads((methods / 'evidence/native-cuda-phase-run-preparation.json').read_bytes())
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(p['test_modules'])
    actual = test_ids(suite)
    assert not loader.errors and actual == p['test_ids'] and len(actual) == len(set(actual))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'test_ids': actual, 'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
              'source_sha256': p['files'], 'gpu_executed': False}
    for field in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses'):
        report[field] = [str(item) for item in getattr(result, field)]
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
    return 0 if result.wasSuccessful() and result.testsRun == len(actual) and not any(report[k] for k in ('errors', 'failures', 'skipped', 'expectedFailures', 'unexpectedSuccesses')) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-prefill16-http-archive-sha256')
    parser.add_argument('--cpu-child', type=Path)
    args = parser.parse_args()
    root = Path('D:/CantorAI/5090-study-20261023')
    methods = Path(__file__).resolve().parent
    assert methods == (root / 'Garnet/test2026/cuda/qwen38_5090').resolve()
    sys.path.insert(0, str(methods))
    if args.cpu_child is not None:
        assert args.cpu_child.resolve().is_relative_to((root / 'results' / JOB).resolve())
        return cpu_child(methods, args.cpu_child)
    assert args.closed_prefill16_http_archive_sha256
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    preparation = methods / 'evidence/native-cuda-phase-run-preparation.json'
    p = json.loads(preparation.read_bytes())
    assert p['status'] == 'native_cuda_phases_prepared_gpu_unrun'
    assert all('Garnet/' + module.replace('.', '/') + '.py' in p['files'] for module in p['test_modules'])
    for name, digest in p['files'].items():
        assert sha(root / name) == digest, ('CUDA diagnosis source/input drift', name)
    for name, digest in p['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / name) == digest
    host = closed_host_diagnostic(root, p['host_parent']['archive_sha256'])
    assert host['proof_status'] == 'native_host_stage_full4k_replayed' and host['exit_code'] == 0
    assert host['proof_sha256'] == p['host_parent']['proof_sha256']
    trial = closed_trial(root, args.closed_prefill16_http_archive_sha256)
    tool = profiler(root)
    assert tool == p['profiler']
    smoke_path = root / 'results/native-cuda-phase-profiled-driver-smoke-attempt1/report.json'
    smoke = json.loads(smoke_path.read_bytes())
    assert sha(smoke_path) == p['profiled_driver_cpu_smoke_sha256']
    assert smoke['status'] == 'profiled_cpu_driver_ipc_verified' and smoke['gpu_model_executed'] is False
    assert [row['requested_driver_exit_code'] for row in smoke['cases']] == [0, 7]
    assert [row['actual_nsys_wrapper_exit_code'] for row in smoke['cases']] == [0, 7]
    receipt = root / 'results/study-ssd-engine-cache/receipt.json'
    scratch = json.loads(receipt.read_bytes())
    cache = Path(scratch['cache_root'])
    assert scratch['status'] == 'ssd_engine_scratch_prepared' and cache.resolve().drive.upper() == 'C:'
    assert json.loads((cache / 'study-owner.json').read_bytes())['study'] == str(root)
    assert shutil.disk_usage(cache).free >= 96 * 1024**3
    assert shutil.disk_usage(root).free >= 32 * 1024**3, 'Reserve disk for actual CUDA records before profiling'
    out = root / 'results' / JOB
    archive = root / 'results' / (JOB + '-evidence.zip')
    assert not out.exists() and not archive.exists()
    idle()
    out.mkdir()
    os.chdir(root / 'Garnet')
    code = 1
    before = {path.name for path in cache.iterdir()}
    started = time.time()
    common = [root / name for name in p['files']] + [preparation, receipt, root / tool['setup_path'], smoke_path]
    common.extend(path for path in smoke_path.parent.rglob('*') if path.is_file() and path.suffix in ('.json', '.py', '.log'))
    for parent in (host, trial):
        common.extend([root / 'results' / parent['job'] / 'workflow.json', root / 'results' / (parent['job'] + '-verified.json')])
    workflow = {'status': 'running', 'preparation_sha256': sha(preparation), 'source_sha256': p['files'],
                'runtime_dependencies': p['runtime_dependencies'], 'library_sha256': sha(root / 'build/bin/garnet.dll'),
                'closed_host': host, 'closed_prefill16_http': trial, 'profiler': tool,
                'profiled_driver_cpu_smoke_sha256': sha(smoke_path),
                'cleanup_finished': False, 'performance_admitted': False, 'scope': p['scope']}
    def save():
        (out / 'workflow.json').write_text(json.dumps(workflow, indent=2), encoding='utf8')
    save()
    try:
        code = run([sys.executable, str(Path(__file__)), '--cpu-child', str(out / 'cpu-result.json')],
                   out / 'cpu-ledger', 'correctness', 'Source-pinned CPU test IDs, marker ownership, real interval arithmetic and complete worker model AST', common)
        if code:
            raise RuntimeError('CUDA diagnostic CPU controls failed')
        code = run([sys.executable, str(methods / 'monitor_native_probe.py'), '--study', str(root),
                    '--output', str(out / 'resources'), '--', sys.executable, str(methods / 'collect_native_cuda_phases.py'),
                    '--root', str(root), '--output', str(out / 'native'), '--cache', str(cache)],
                   out / 'gpu-ledger', 'correctness', '24 actual3968+128 retained-ID requests, two Nsight node traces; all first-use and rotated repeats retained', common)
        if code:
            raise RuntimeError('Actual CUDA diagnostic failed; preserve raw trace and owned cleanup')
        report = json.loads((out / 'native/report.json').read_bytes())
        assert report['cuda_phase_status'] == 'both_modes_actual_cuda_markers_collected_needs_independent_replay'
        workflow['status'] = 'native_cuda_phases_collected_needs_replay'
    except BaseException as error:
        workflow.update(status='failed', error=repr(error))
        code = code or 1
    finally:
        workflow.update(exit_code=code, source_after_sha256={name: sha(root / name) for name in p['files']},
                        runtime_dependencies_after={name: sha(root / 'build/bin' / name) for name in p['runtime_dependencies']},
                        profiler_after=profiler(root), library_unchanged=sha(root / 'build/bin/garnet.dll') == workflow['library_sha256'])
        save()
        idle()
        workflow['retired_private_engine_files'] = retire(cache, root, before, started)
        workflow['cleanup_finished'] = True
        save()
        # Stream potentially large profiler artifacts; never load all traces into RAM.
        files = {path.relative_to(root).as_posix(): path for path in [*out.rglob('*'), *common] if path.is_file()}
        manifest = {'files': {name: sha(path) for name, path in files.items()}, 'scope': p['scope']}
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for name, path in files.items():
                z.write(path, name)
            z.writestr('evidence-manifest.json', json.dumps(manifest, indent=2))
        print(json.dumps({'status': workflow['status'], 'archive_sha256': sha(archive),
                          'bytes': archive.stat().st_size, 'files': len(files)}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
