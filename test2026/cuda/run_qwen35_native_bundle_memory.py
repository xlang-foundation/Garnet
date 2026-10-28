"""Separate phased WDDM diagnostic; never primary latency or true-peak evidence."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from qwen35_native_bundle_experiment import NativeBundleMTPSession
from qwen35_benchmark_workload import batch_inputs, input_hashes

source_path, output_path = map(Path, sys.argv[1:3])
mode = sys.argv[3]
assert mode in ('control', 'bundle') and not output_path.exists()
source = json.loads(source_path.read_text())
assert len(source) == 28
source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
profile = dict(source[0]['profile'])
name, batch = source[0]['model'], profile['batch']
assert profile['prefillChunk'] == 64 and profile['decodeChunk'] == 4 and profile['speculativeMTP'] == 3
checkpoint = str(source_path.parent / 'models' / name)
events = []
strategy = {'mode': mode, 'target_prefill': 64, 'target_decode': 4, 'depth': 3,
            'mtp_prefill': 4, 'mtp_step': 4, 'mtp_correction': 4,
            'native_execution_bundle': mode == 'bundle', 'experimental_only': mode == 'bundle',
            'driver_graph_memory_excluded_from_owned_ledger': True}



def emit(event, **values):
    row = {'event': event, 'unix_seconds': time.time(), 'process_id': os.getpid(), 'engine': 'garnet',
           'model': name, 'profile': profile, 'mtp_strategy': strategy,
           'source_cohort': str(source_path), 'source_cohort_sha256': source_sha,
           'native_build_policy': {
               'zero_aux_streams_requested': os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS') == '1',
               'whole_program_cuda_requested': os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA') == '1',
               'reuse_static_bindings': os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS') == '1',
               'sync_temporary_allocator': os.environ.get('GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR') == '1'}, **values}
    events.append(row)
    output_path.write_text(json.dumps(events, indent=2) + '\n')
    print('NATIVE_BUNDLE_MEMORY_EVENT', event, os.getpid(), flush=True)


# Establish an explicit CUDA/SDK context baseline without a model or weights.
# Its one FP32 word is read back and released before the phase begins.
baseline_io = garnet.cuda_tensor_io()
sentinel = baseline_io.zeros([1])
baseline_io.zero(sentinel)
assert len(baseline_io.finite_f32_bytes(sentinel)) == 4
del sentinel
baseline_io.close()
del baseline_io
emit('baseline_begin', scope='Context-only baseline; missing process counter instances remain unobserved, never zero-filled')
time.sleep(10)
emit('baseline_end')
emit('load_begin', scope='WDDM diagnostic with polling and10-second holds; not latency or lifetime-peak evidence')
saved_constructor = specmod.CUDASpeculativeSession
model = None
try:
    if mode == 'bundle':
        specmod.CUDASpeculativeSession = NativeBundleMTPSession
    try:
        model = Model(checkpoint, profile, name)
    finally:
        specmod.CUDASpeculativeSession = saved_constructor
    emit('loaded', owned_memory=model.memory_info())

    def infer(index):
        reference = source[index]
        names, paths = batch_inputs(reference['dataset'], reference['trial'], reference['workload_group'], batch, True)
        hashes = input_hashes(paths)
        assert hashes == reference['input_sha256']
        result = model.infer_batch(['Describe the main objects in this image in one sentence.'] * batch,
                                   paths, reference['request_output_limit'])
        keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')
        assert result['profile'] == profile and result['vision_compile_ms'] == 0
        assert all(result[k] == reference[k] for k in keys)
        assert result['speculative_metrics'] == reference['speculative_metrics']
        assert model.memory_info()['owned_model_bytes'] == reference['owned_memory']['owned_model_bytes']
        return result, {'images': names, 'input_sha256': hashes,
                        'source_record_index': index, 'source_workload_key': [reference['trial'], reference['workload_group']],
                        'prompt_token_ids': result['prompt_token_ids'],
                        'output_signature': {k: result[k] for k in ('generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids')},
                        'visual_tokens_per_row': result['visual_tokens_per_row'],
                        'speculative_metrics': result['speculative_metrics']}

    for index in range(2):
        result, inputs = infer(index)
        emit('warmup_complete', trial=index, sampled_tokens=result['sampled_tokens_including_stop'], **inputs)
    emit('steady_before_begin', owned_memory=model.memory_info())
    time.sleep(10)
    emit('steady_before_end')
    emit('inference_begin')
    for index in range(2, 6):
        result, inputs = infer(index)
        emit('request_complete', trial=index, sampled_tokens=result['sampled_tokens_including_stop'],
             owned_memory=model.memory_info(), **inputs)
    emit('inference_end')
    memory = model.memory_info()
    graph=memory['speculative']['mtp_graph']
    assert graph['contexts']==graph['graph_captures']==1 and graph['graph_capture_failures']==0
    if mode == 'bundle':
        bundle=model.speculative.bundle_info()
        assert bundle['fallback_calls']==2 and bundle['submissions']>0
        assert bundle['bundle']['reserved_model_contexts']==1 and not bundle['bundle']['poisoned']
    emit('steady_after_begin', owned_memory=memory)
    time.sleep(10)
    emit('steady_after_end')
finally:
    specmod.CUDASpeculativeSession = saved_constructor
    if model is not None:
        model.close()
    emit('closed_begin')
    time.sleep(10)
    emit('closed_end')
emit('complete')
print('NATIVE_BUNDLE_MEMORY_COMPLETE', name, mode, flush=True)
