"""Separate API-return wall time from GPU retirement; diagnostic only."""
import ctypes
import os
import runpy
import sys
import time
sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo)
from runtime.cuda_serving import Model

runtime = ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
original = Model.infer_batch
request_index = 0


def retired(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    entry = time.monotonic()
    assert runtime.cudaDeviceSynchronize() == 0, 'Prior request GPU retirement failed'
    start = time.monotonic()
    result = original(self, *args, **kwargs)
    api_return = time.monotonic()
    assert runtime.cudaDeviceSynchronize() == 0, 'Request GPU retirement failed'
    gpu_retired = time.monotonic()
    result['retirement_diagnostic'] = {
        'request_index': index,
        'prior_request_barrier_seconds': start - entry,
        'api_return_wall_seconds': api_return - start,
        'post_return_gpu_retirement_seconds': gpu_retired - api_return,
        'fully_retired_wall_seconds': gpu_retired - start,
        'primary_performance_eligible': False,
        'scope': 'Same-process CUDA device barrier before API start and after return. Tail includes '
                 'barrier/host overhead and any work in this CUDA context, not proof of MTP correction '
                 'cost alone. Outer benchmark wall includes diagnostic overhead; do not replace saved '
                 'primary API-return cohorts or compare directly to uninstrumented HTTP engines.'}
    return result


Model.infer_batch = retired
benchmark = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_qwen35_garnet.py')
sys.argv[0] = benchmark
runpy.run_path(benchmark, run_name='__main__')
assert request_index == 28, 'Expected the complete real-image cohort'
