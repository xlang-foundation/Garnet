"""Unwired matched-cohort timing entry point; caller must qualify controls first.

One process owns one model/engine. The controller pins source/runtime/input
identity, sequences ABBA roles, observes external processes and excludes loading.
No state hashing, sanitizer or other GPU workload is allowed in a timing role.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from qwen35_shared_mtp_experiment import SharedMTPSession
from qwen35_benchmark_workload import batch_inputs, input_hashes

source_path, output_path = map(Path, sys.argv[1:3])
mode = sys.argv[3]
assert mode in ('baseline', 'split') and not output_path.exists()
source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
source = json.loads(source_path.read_text())
assert len(source) == 28 and all(r['warmup'] == (i < 8) for i, r in enumerate(source))
profile = source[0]['profile']
assert all(r['profile'] == profile for r in source)
assert profile['prefillChunk'] == 64 and profile['decodeChunk'] == 4 and profile['speculativeMTP'] == 3
batch = profile['batch']
checkpoint = str(source_path.parent / 'models' / source[0]['model'])
original_constructor = specmod.CUDASpeculativeSession
if mode == 'split':
    def constructor(*args):
        return SharedMTPSession(*args, mode='split')
    specmod.CUDASpeculativeSession = constructor
print('SHARED_MTP_TIMING_LOAD_BEGIN', mode, source[0]['model'], time.time(), flush=True)
try:
    model = Model(checkpoint, profile, source[0]['model'])
finally:
    specmod.CUDASpeculativeSession = original_constructor
print('SHARED_MTP_TIMING_LOAD_COMPLETE', mode, time.time(), flush=True)
rows = []
try:
    for reference in source:
        names, paths = batch_inputs(reference['dataset'], reference['trial'], reference['workload_group'], batch, True)
        hashes = input_hashes(paths)
        assert hashes == reference['input_sha256']
        prompt = 'Describe the main objects in this image in one sentence.'
        print('SHARED_MTP_TIMING_REQUEST_BEGIN', reference['trial'], reference['workload_group'], time.time(), flush=True)
        wall = time.monotonic()
        result = model.infer_batch([prompt] * batch, paths, reference['request_output_limit'])
        wall = time.monotonic() - wall
        result.update({key: reference[key] for key in ('engine', 'model', 'trial', 'warmup', 'dataset', 'variant_round',
                      'request_output_limit', 'workload_group', 'matched_batch_inputs')})
        result['native_build_policy'] = {
            'zero_aux_streams_requested': os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS') == '1',
            'whole_program_cuda_requested': os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA') == '1',
            'reuse_static_bindings': os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS') == '1',
            'sync_temporary_allocator': os.environ.get('GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR') == '1'}
        result.update({'input_sha256': hashes, 'images': names, 'wall_seconds': wall, 'process_id': os.getpid(),
                       'timing_scope': reference['timing_scope'], 'source_cohort': str(source_path), 'source_cohort_sha256': source_sha,
                       'experimental_mtp_graphs': {'mode': mode, 'prefill': 16 if mode == 'split' else 4,
                           'draft_step': 1 if mode == 'split' else 4, 'correction': 4,
                           'target_prefill': 64, 'target_decode': 4, 'depth': 3,
                           'shared_bf16_mtp_weights': mode == 'split', 'production_profile_opt_in_available': False}})
        # These small metadata observations happen after both timing scopes.
        result['io_memory'] = json.loads(model.decoder.io.memory_json())
        result['owned_memory'] = model.memory_info()
        rows.append(result)
        output_path.write_text(json.dumps(rows, indent=2) + '\n')
        assert result['profile'] == profile and result['vision_compile_ms'] == 0
        assert result['native_build_policy'] == reference['native_build_policy']
        for key in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row'):
            assert result[key] == reference[key], 'Timing changed target output: ' + key
        print('SHARED_MTP_TIMING_REQUEST_COMPLETE', len(rows), wall, flush=True)
finally:
    model.close()
print('SHARED_MTP_TIMING_COMPLETE', mode, len(rows), time.time(), flush=True)
