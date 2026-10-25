"""Actual embedded-Python trace compatibility with fake IO; no model/GPU calls."""
from pathlib import Path
import json
import sys

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root / 'Garnet'))
from test2026.cuda.qwen38_5090.native_cuda_phase_bootstrap_log import NativeCUDAPhaseLoggedTrace

assert not out.exists() and out.resolve().is_relative_to((root / 'results').resolve())
out.mkdir()
class CPUFixtureIO:
    def from_host(self, values, shape):
        assert shape == [1, len(values)]
        return object()

trace = NativeCUDAPhaseLoggedTrace(root, 'ordinary', out / 'synthetic-trace.jsonl')
trace.bind_io(CPUFixtureIO())
trace.observe({'event': 'ready', 'generation_mode': 'ordinary'})
trace.start('explicit-embedded-cpu-fixture')
try:
    with trace.stage('reset'):
        raise RuntimeError('explicit embedded CPU fixture error')
except RuntimeError as error:
    assert str(error) == 'explicit embedded CPU fixture error'
    trace.close()
else:
    raise AssertionError('CPU fixture error was suppressed')
rows = [json.loads(line) for line in (out / 'synthetic-trace.bootstrap.jsonl').read_bytes().splitlines()]
error = next(row for row in rows if row['stage'] == 'native_worker_finally_entered')['exception']
assert error['type'] == 'RuntimeError' and 'explicit embedded CPU fixture error' in error['traceback']
report = {'status': 'embedded_python_trace_cpu_compatibility_passed', 'gpu_executed': False,
          'model_executed': False, 'exception_preserved': True,
          'scope': 'Actual xlang3 embedded Python using fake Python IO/marker objects only; no Garnet import, device allocation or model. Synthetic markers are not CUDA evidence.'}
(out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print('EMBEDDED_PYTHON_TRACE_CPU_COMPATIBILITY_PASS', flush=True)
