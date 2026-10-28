"""Unwired one-request TensorRT layer profiler; no hardware counters required.

The existing native IProfiler path temporarily bypasses CUDA graph replay for
one execution per engine. All28 original-MTP signatures remain required. This
cannot be used as a primary latency or memory benchmark.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
repo = Path(__file__).resolve().parents[2]
assert (repo / 'runtime/cuda_serving.py').is_file()
sys.path.insert(0, str(repo))
from runtime.cuda_serving import Model

layers = Path(sys.argv[2]).resolve().with_suffix('.layers.tsv')
assert not layers.exists(), 'Preserve prior layer records'
assert os.environ.get('GARNET_PROFILE_DECODE_LAYERS') is None
original_infer = Model.infer_batch
request_index = 0


def profiled(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    capture = index == 8
    assert os.environ.get('GARNET_PROFILE_DECODE_LAYERS') is None
    if capture:
        os.environ['GARNET_PROFILE_DECODE_LAYERS'] = str(layers)
    try:
        result = original_infer(self, *args, **kwargs)
    finally:
        if capture:
            os.environ.pop('GARNET_PROFILE_DECODE_LAYERS', None)
    if capture:
        assert layers.is_file() and layers.stat().st_size > 0, 'Native layer profiler returned no evidence'
    result['trt_layer_profile_diagnostic'] = {'request_index': index, 'enabled': capture,
        'layer_file': str(layers), 'primary_latency_accepted': False, 'memory_accepted': False,
        'scope': 'Existing native OneShotTRTLayerProfiler enabled only in warmed request8. '
                 'First execution per engine uses ordinary TensorRT enqueue with layer profiling instead '
                 'of graph replay; remaining requests use original settings. All28 signatures must match. '
                 'Layer samples identify instrumented engine/layer cost, not exact original kernel/grid '
                 'attribution, representative average, primary latency/memory or causal speedup.'}
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
    os.environ.pop('GARNET_PROFILE_DECODE_LAYERS', None)
