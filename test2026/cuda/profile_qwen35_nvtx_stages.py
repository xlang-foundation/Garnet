"""Warmed stage annotations preserving original native execution policy.

No layer-profiler flag, tensor snapshots, altered graph sizes or extra model
calls. One synchronized Nsight request in the original full28 cohort.
"""
import ctypes
import os
import runpy
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
from runtime.cuda_serving import Model

assert os.environ.get('GARNET_PROFILE_DECODE_LAYERS') is None
bridge = ctypes.CDLL('D:/CantorAI/work/qwen35-cuda/qwen35_nvtx_bridge_v1.dll')
bridge.qwen35NvtxPush.argtypes = [ctypes.c_char_p]
bridge.qwen35NvtxPush.restype = ctypes.c_int
assert bridge.qwen35NvtxVersion() == 3
cuda = ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
original_infer = Model.infer_batch
request_index = 0

# Initialize annotation dispatch before the captured request; no GPU operation.
bridge.qwen35NvtxPush(b'qwen35_nvtx_bootstrap')
bridge.qwen35NvtxPop()


def observed(self, *args, **kwargs):
    global request_index
    index = request_index
    request_index += 1
    capture = index == 8
    marks, stack, restore = [], [], []
    thread = bridge.qwen35NvtxThreadId() if capture else None

    def push(label):
        assert bridge.qwen35NvtxThreadId() == thread
        bridge.qwen35NvtxPush(('garnet:' + label).encode('ascii'))
        stack.append(label)
        marks.append({'operation': 'push', 'label': label})

    def pop(label):
        assert bridge.qwen35NvtxThreadId() == thread and stack.pop() == label
        bridge.qwen35NvtxPop()
        marks.append({'operation': 'pop', 'label': label})

    def patch(obj, name, label):
        original = getattr(obj, name)
        assert callable(original) and hasattr(obj, '__dict__')
        restore.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))

        def call(*a, **k):
            resolved = label(*a, **k) if callable(label) else label
            push(resolved)
            try:
                return original(*a, **k)
            finally:
                pop(resolved)
        setattr(obj, name, call)

    def mtp_label(flags, positions, predict, stage):
        assert stage in ('prefill', 'draft', 'correction')
        if stage == 'prefill':
            return 'mtp_initial_draft' if predict else 'mtp_prompt_cache'
        return 'mtp_' + stage

    if capture:
        assert self.speculative is not None
        patch(self, '_prepare', 'input_row')
        patch(self.decoder, 'prefill_sequence', 'target_prefill')
        patch(self.decoder, 'chunk_hidden', 'target_chunk')
        patch(self.speculative, 'reset', 'spec_reset')
        patch(self.speculative, 'prefill', 'speculative_prefill')
        patch(self.speculative, '_call', mtp_label)
        patch(self.speculative, '_draft', 'draft_prepare')
        patch(self.speculative, '_correct', 'correction_prepare')
        patch(self.speculative, 'generate', 'generation')
        patch(self.speculative.verifier, 'verify', 'target_verify')
        patch(self.speculative.verifier, 'commit_prefix', 'target_commit')
        patch(self.speculative.verifier, '_read', 'verification_read')
    started = False
    try:
        if capture:
            assert cuda.cudaProfilerStart() == 0
            started = True
            push('request')
        try:
            result = original_infer(self, *args, **kwargs)
        finally:
            if capture and started:
                pop('request')
                assert not stack
    finally:
        try:
            if started:
                try:
                    assert cuda.cudaDeviceSynchronize() == 0
                finally:
                    assert cuda.cudaProfilerStop() == 0
        finally:
            for obj, name, existed, value in reversed(restore):
                if existed:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
    result['gpu_trace_diagnostic'] = {'request_index': index, 'capture': capture,
        'capture_tail_synchronized': capture, 'storage_strategy': 'original',
        'scope': 'One warmed NVTX/node-level diagnostic with synchronized tail; no primary timing/memory acceptance.'}
    result['nvtx_stage_diagnostic'] = {'request_index': index, 'capture': capture,
        'native_thread_id': thread, 'events': marks,
        'native_layer_profiler_enabled': False, 'primary_latency_accepted': False,
        'scope': 'CPU-only NVTX annotations around existing Python calls. Native execution policies and calls '
                 'unchanged except standard profiler start/stop/tail synchronization. Correlate GPU events '
                 'through their actual CUDA API correlation IDs and per-thread NVTX nesting, not GPU timestamp '
                 'containment in CPU ranges. Instrumentation invalidates primary latency/memory and speed claims.'}
    return result


try:
    Model.infer_batch = observed
    entry = Path(__file__).with_name('benchmark_qwen35_shared_mtp.py')
    sys.argv[0] = str(entry)
    sys.argv[3:] = ['baseline']
    runpy.run_path(str(entry), run_name='__main__')
    assert request_index == 28
finally:
    Model.infer_batch = original_infer
