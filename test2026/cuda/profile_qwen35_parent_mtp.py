"""One warmed request in a full preserved cohort; diagnostic, never timing acceptance."""
import ctypes
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
import qwen35_shared_mtp_experiment as experiment

strategy = sys.argv[3]
assert strategy in ('original', 'parent')
original_session = experiment.SharedMTPSession
original_infer = Model.infer_batch
runtime = ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
request_index = 0


def parent_session(*args, **kwargs):
    assert kwargs['mode'] == 'split' and 'parent_storage' not in kwargs
    return original_session(*args, parent_storage=True, **kwargs)


def traced(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    capture = index == 8
    if capture:
        assert runtime.cudaProfilerStart() == 0, 'cudaProfilerStart failed'
    try:
        result = original_infer(self, *args, **kwargs)
    finally:
        if capture:
            try:
                assert runtime.cudaDeviceSynchronize() == 0, 'capture tail synchronization failed'
            finally:
                assert runtime.cudaProfilerStop() == 0, 'cudaProfilerStop failed'
    result['gpu_trace_diagnostic'] = {'request_index': index, 'capture': capture,
        'capture_tail_synchronized': capture, 'storage_strategy': strategy,
        'scope': 'One warmed node-level Nsight capture with synchronized tail. All28 signatures retained. '
                 'Instrumentation and tail waits invalidate primary timing; source cohorts are historical controls.'}
    return result


try:
    if strategy == 'parent':
        experiment.SharedMTPSession = parent_session
    Model.infer_batch = traced
    entry = Path(__file__).with_name('benchmark_qwen35_shared_mtp.py')
    sys.argv[0] = str(entry)
    sys.argv[3] = 'baseline' if strategy == 'original' else 'split'
    runpy.run_path(str(entry), run_name='__main__')
    assert request_index == 28
finally:
    Model.infer_batch = original_infer
    experiment.SharedMTPSession = original_session
