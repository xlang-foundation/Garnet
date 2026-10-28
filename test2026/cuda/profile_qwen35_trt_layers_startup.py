"""Collect existing one-shot native layer records with a fixed startup flag.

The failed v1 used a runtime Python environment mutation and produced no native
records. This diagnostic passes the flag from the host before process startup.
First-call samples may be cold and are never primary latency/memory evidence.
"""
import os
import runpy
import sys
from pathlib import Path

layers = Path(sys.argv[2]).resolve().with_suffix('.layers.tsv')
assert not layers.exists(), 'Preserve prior native records'
assert Path(os.environ['GARNET_PROFILE_DECODE_LAYERS']).resolve() == layers
sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
repo = Path(__file__).resolve().parents[2]
assert (repo / 'runtime/cuda_serving.py').is_file()
sys.path.insert(0, str(repo))
from runtime.cuda_serving import Model

original_infer = Model.infer_batch
request_index = 0


def profiled(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    assert Path(os.environ['GARNET_PROFILE_DECODE_LAYERS']).resolve() == layers
    before = layers.stat().st_size if layers.exists() else 0
    result = original_infer(self, *args, **kwargs)
    assert layers.is_file() and layers.stat().st_size > 0, 'Startup native layer profiler returned no records'
    after = layers.stat().st_size
    assert after >= before
    result['startup_trt_layer_profile_diagnostic'] = {'request_index': index,
        'native_profile_environment_fixed': True, 'profile_bytes_before': before, 'profile_bytes_after': after,
        'layer_file': str(layers), 'primary_latency_accepted': False, 'memory_accepted': False,
        'scope': 'Profiler flag supplied by host before native process startup and fixed for all28 requests. '
                 'Existing IProfiler collects first execution per engine, potentially cold during setup/warmup. '
                 'Per-request file sizes locate observed growth, not exact layer GPU timestamps. Serial fast path '
                 'is disabled by this environment flag. No warmed timing, average/frequency-weighted layer share, '
                 'exact original kernel/grid attribution, primary latency/memory or causal speedup.'}
    return result


try:
    Model.infer_batch = profiled
    entry = Path(__file__).with_name('benchmark_qwen35_shared_mtp.py')
    sys.argv[0] = str(entry)
    sys.argv[3:] = ['baseline']
    runpy.run_path(str(entry), run_name='__main__')
    assert request_index == 28
finally:
    Model.infer_batch = original_infer
