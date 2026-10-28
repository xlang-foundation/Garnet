"""Unwired matched native dispatch timing; qualify complete state first."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from qwen35_native_bundle_experiment import NativeBundleMTPSession, AST_PROOF
from qwen35_benchmark_workload import batch_inputs, input_hashes

source_path, output = map(Path, sys.argv[1:3])
mode = sys.argv[3]
assert mode in ('control', 'bundle') and not output.exists()
source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
source = json.loads(source_path.read_text())
assert len(source) == 28 and all(r['warmup'] == (i < 8) for i, r in enumerate(source))
profile = source[0]['profile']
assert all(r['profile'] == profile for r in source)
assert profile['prefillChunk'] == 64 and profile['decodeChunk'] == 4 and profile['speculativeMTP'] == 3
batch = profile['batch']
checkpoint = str(source_path.parent / 'models' / source[0]['model'])
saved_constructor = specmod.CUDASpeculativeSession
if mode == 'bundle':
    specmod.CUDASpeculativeSession = NativeBundleMTPSession
print('NATIVE_BUNDLE_TIMING_LOAD_BEGIN', mode, source[0]['model'], time.time(), flush=True)
try:
    model = Model(checkpoint, profile, source[0]['model'])
finally:
    specmod.CUDASpeculativeSession = saved_constructor
print('NATIVE_BUNDLE_TIMING_LOAD_COMPLETE', mode, time.time(), flush=True)
rows = []
try:
    for reference in source:
        names, paths = batch_inputs(reference['dataset'], reference['trial'], reference['workload_group'], batch, True)
        hashes = input_hashes(paths)
        assert hashes == reference['input_sha256']
        prompt = 'Describe the main objects in this image in one sentence.'
        before = model.speculative.bundle_info() if mode == 'bundle' else None
        print('NATIVE_BUNDLE_TIMING_REQUEST_BEGIN', reference['trial'], reference['workload_group'], time.time(), flush=True)
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
                       'experimental_native_bundle': {'mode': mode, 'ast_proof': AST_PROOF,
                           'native_bundle_before': before, 'native_bundle_after': model.speculative.bundle_info() if mode == 'bundle' else None,
                           'extra_model_passes': 0, 'serving_default_modified': False,
                           'driver_graph_memory_untracked_by_owned_ledger': True}})
        # Small metadata only, after both timing scopes. No tensor readback/hash.
        result['io_memory'] = json.loads(model.decoder.io.memory_json())
        result['owned_memory'] = model.memory_info()
        rows.append(result)
        output.write_text(json.dumps(rows, indent=2) + '\n')
        assert result['profile'] == profile and result['vision_compile_ms'] == 0
        assert result['native_build_policy'] == reference['native_build_policy']
        for key in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons',
                    'stop_token_ids', 'visual_tokens_per_row', 'speculative_metrics', 'cache_policy', 'arithmetic'):
            assert result[key] == reference[key], 'Timing changed output/work: ' + key
        assert result['owned_memory']['owned_model_bytes'] == reference['owned_memory']['owned_model_bytes']
        graph = result['owned_memory']['speculative']['mtp_graph']
        assert graph['contexts'] == graph['graph_captures'] == 1 and graph['graph_capture_failures'] == 0
        if mode == 'bundle':
            after = result['experimental_native_bundle']['native_bundle_after']
            total = sum(result['speculative_metrics'][key] for key in ('mtp_prefill_calls', 'mtp_draft_calls', 'mtp_correction_calls'))
            assert after['submissions'] - before['submissions'] + after['fallback_calls'] - before['fallback_calls'] == total
            assert after['fallback_calls'] == 2 and graph['execution_bundle_reservations'] == 1
            assert after['bundle']['submissions'] == after['submissions'] and not after['bundle']['poisoned']
        else:
            assert graph['execution_bundle_reservations'] == 0
        print('NATIVE_BUNDLE_TIMING_REQUEST_COMPLETE', len(rows), wall, flush=True)
finally:
    model.close()
print('NATIVE_BUNDLE_TIMING_COMPLETE', mode, len(rows), time.time(), flush=True)
