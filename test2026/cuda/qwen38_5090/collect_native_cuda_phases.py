"""Profile the request driver; keep native worker IPC inside the process tree."""
from pathlib import Path
import argparse
import copy
import json
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import profiler
from test2026.cuda.qwen38_5090.replay_native_cuda_phases import replay
from test2026.cuda.qwen38_5090.run_logged import run


def merge_reports(reports):
    assert len(reports) == 2
    for mode, report in zip(('ordinary', 'mtp'), reports):
        assert report['status'] == 'native_host_stage_diagnostic_collected_needs_independent_replay'
        assert set(report['modes']) == {mode} and report['performance_admitted'] is False
        record = report['modes'][mode]
        assert record['ready']['generation_mode'] == mode and record['status'] == 'instrumented_host_stages_replayed'
        assert type(record['worker_exit_code']) is int and record['worker_exit_code'] == 0 and len(record['requests']) == 12
    for key in ('status', 'cases_sha256', 'parent_archive_sha256', 'performance_admitted', 'scope'):
        assert reports[0][key] == reports[1][key]
    merged = copy.deepcopy(reports[0])
    merged['modes']['mtp'] = copy.deepcopy(reports[1]['modes']['mtp'])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.output
    assert root.resolve() == Path('D:/CantorAI/5090-study-20261023').resolve()
    assert not out.exists() and out.resolve().is_relative_to((root / 'results').resolve())
    out.mkdir()
    methods = Path(__file__).resolve().parent
    tool = profiler(root)
    nsys = root / tool['path']
    reports = []
    for mode in ('ordinary', 'mtp'):
        application = subprocess.list2cmdline([sys.executable, str(methods / 'probe_native_cuda_phases.py'),
            '--root', str(root), '--output', str(out), '--cache', str(args.cache), '--mode', mode])
        stem = out / ('cuda-trace-' + mode)
        command = [str(nsys), 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                   '--cuda-graph-trace=node', '--force-overwrite=false', '--output=' + str(stem),
                   'cmd.exe', '/d', '/c', application]
        code = run(command, out / (mode + '-profile-ledger'), 'correctness',
                   'Outer Nsight process-tree collection; normal inner request/worker pipes; no live Nsight stdout dependency',
                   [nsys, methods / 'probe_native_cuda_phases.py', methods / 'serve_native_cuda_phase_worker.bat', Path(__file__)])
        assert type(code) is int and code == 0, ('Profiled request driver failed', mode, code)
        report = json.loads((out / ('report-' + mode + '.json')).read_bytes())
        record = report['modes'][mode]
        assert type(record['worker_exit_code']) is int and record['worker_exit_code'] == 0
        target = out / mode
        code = run([str(nsys), 'stats', '--report', 'cuda_gpu_kern_sum,cuda_api_sum', '--format', 'csv', str(stem) + '.nsys-rep'],
                   target / 'stats-ledger', 'correctness', 'Export actual process-tree CUDA records; no timing promotion',
                   [nsys, methods / 'replay_native_cuda_phases.py'])
        assert type(code) is int and code == 0
        for suffix in ('.nsys-rep', '.sqlite'):
            source = Path(str(stem) + suffix)
            destination = target / ('cuda-trace' + suffix)
            assert source.resolve().parent == out.resolve() and destination.resolve().is_relative_to(out.resolve())
            assert source.is_file() and not destination.exists()
            source.rename(destination)
        rows = [json.loads(line) for line in (target / 'host-stages.jsonl').read_bytes().splitlines()]
        markers = [json.loads(line) for line in (target / 'host-stages.cuda-markers.jsonl').read_bytes().splitlines()]
        with sqlite3.connect('file:' + (target / 'cuda-trace.sqlite').as_posix() + '?mode=ro', uri=True) as connection:
            gpu = replay(connection, markers, rows)
        assert gpu['requests'] == 12 and gpu['unique_markers'] == 144
        (target / 'cuda-phases-replayed.json').write_text(json.dumps(gpu, indent=2), encoding='utf8')
        record['stats_exit_code'] = code
        record['profiled_driver_exit_code'] = 0
        record['cuda_phase_replay'] = {key: gpu[key] for key in ('status', 'requests', 'unique_markers', 'owner')}
        reports.append(report)
    merged = merge_reports(reports)
    merged['cuda_phase_status'] = 'both_modes_actual_cuda_markers_collected_needs_independent_replay'
    with (out / 'report.json').open('x', encoding='utf8') as stream:
        json.dump(merged, stream, indent=2)
    print('NATIVE_CUDA_PHASE_PROFILED_DRIVERS_COLLECTED', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
