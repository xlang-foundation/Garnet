"""Complete-state real-image control; no timing or process-memory acceptance.

One mode/model per process. Source/fixture hashing belongs before model load;
tensor observation below is intentionally restricted to this correctness role.
"""
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

request_path, output = map(Path, sys.argv[1:3])
mode = sys.argv[3]
assert mode in ('control', 'bundle') and not output.exists()
request = json.loads(request_path.read_text())
for case in request['cases']:
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']] == case['input_sha256']
evidence = {'request': request, 'mode': mode, 'process_id': os.getpid(), 'ast_proof': AST_PROOF,
            'records': [], 'events': [], 'all_controls_passed': False,
            'primary_latency_accepted': False, 'process_memory_accepted': False, 'independent_quality_accepted': False}

def save():
    output.write_text(json.dumps(evidence, indent=2) + '\n')

def event(name):
    evidence['events'].append({'name': name, 'unix_seconds': time.time()})
    save()
    print('NATIVE_BUNDLE_REAL_CONTROL', mode, name, flush=True)

def state(model):
    d, s = model.decoder, model.speculative
    b = d.batch
    values = [('last_hidden', d.last_hidden, d.config['hidden_size']),
              ('canonical', d.state, d.programs[d.decode_chunk_size].state_width),
              ('mtp_key', s.key, s.cache_width), ('mtp_value', s.value, s.cache_width),
              ('mtp_last_hidden', s.last_hidden, s.hidden_size),
              ('mtp_hidden', s.hidden, s.size * s.hidden_size)]
    result = {}
    for name, value, width in values:
        flat = d.io.storage_view(value, [b, width])
        digest = hashlib.sha256()
        for start in range(0, width, 131072):
            count = min(131072, width - start)
            chunk = d.io.slice(flat, start, [b, count])
            data = d.io.finite_f32_bytes(chunk)
            assert len(data) == 4 * b * count
            digest.update(data)
            del chunk, data
        result[name] = {'sha256': digest.hexdigest(), 'finite_fp32_words': b * width}
    return result

def signature(result):
    return {k: result[k] for k in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
            'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row', 'speculative_metrics')}

saved_constructor = specmod.CUDASpeculativeSession
model = None
try:
    if mode == 'bundle':
        specmod.CUDASpeculativeSession = NativeBundleMTPSession
    event('load_begin')
    try:
        model = Model(request['checkpoint'], request['profile'], request['model_id'])
    finally:
        specmod.CUDASpeculativeSession = saved_constructor
    event('load_complete')
    for trial in range(2):
        for case in request['cases']:
            s = model.speculative
            before = s.bundle_info() if mode == 'bundle' else None
            result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
            observed = state(model)
            memory = model.memory_info()
            row = {'trial': trial, 'style': case['style'], 'result': result, 'state': observed,
                   'owned_memory': memory, 'mtp_slots': list(s.slots), 'target_slots': list(model.decoder.slots),
                   'pending': s.verifier.pending, 'poisoned': s.poisoned or model.decoder.poisoned}
            if mode == 'bundle':
                after = s.bundle_info()
                row['bundle_before'], row['bundle_after'] = before, after
            evidence['records'].append(row)
            save()
            assert result['profile'] == request['profile'] and result['vision_compile_ms'] == 0
            assert not row['poisoned'] and not row['pending']
            assert row['target_slots'] == [p + n - 1 for p, n in zip(result['prompt_tokens'], result['sampled_tokens_including_stop'])]
            graph = memory['speculative']['mtp_graph']
            assert graph['contexts'] == graph['graph_captures'] == 1 and graph['graph_capture_failures'] == 0
            if mode == 'bundle':
                total = sum(result['speculative_metrics'][key] for key in ('mtp_prefill_calls', 'mtp_draft_calls', 'mtp_correction_calls'))
                assert after['submissions'] - before['submissions'] + after['fallback_calls'] - before['fallback_calls'] == total
                assert after['fallback_calls'] == 2 and after['submissions'] > before['submissions']
                bundle = after['bundle']
                assert bundle['sealed'] and not bundle['closed'] and not bundle['poisoned']
                assert bundle['reserved_model_contexts'] == 1 and bundle['packed_slices'] == 3 and bundle['last_chunk_copies'] == 1
                assert bundle['new_tensor_allocations'] == 0 and bundle['submissions'] == after['submissions']
                assert graph['execution_bundle_reservations'] == 1
            else:
                assert graph['execution_bundle_reservations'] == 0
            if trial:
                old = next(x for x in evidence['records'] if x['trial'] == 0 and x['style'] == case['style'])
                assert signature(old['result']) == signature(result) and old['state'] == observed
                assert old['mtp_slots'] == row['mtp_slots'] and old['target_slots'] == row['target_slots']
            event('request_' + str(trial) + '_' + case['style'])
    if mode == 'bundle':
        evidence['bundle_before_close'] = model.speculative.bundle_info()
        retained_bundle = model.speculative.native_bundle
    model.close()
    model = None
    event('closed')
    if mode == 'bundle':
        evidence['bundle_after_close'] = json.loads(retained_bundle.stats_json())
        assert evidence['bundle_after_close']['closed'] and evidence['bundle_after_close']['reserved_model_contexts'] == 0
    evidence['all_controls_passed'] = True
    evidence['scope'] = 'Two resets of four actual image cases; full target canonical/last and MTP KV/hidden words, original host controls/work and lifecycle. Separate processes compared externally. No independent quality, performance or physical memory acceptance.'
    save()
    print('NATIVE_BUNDLE_REAL_CONTROLS_COMPLETE', mode, flush=True)
except Exception as error:
    evidence['failure'] = repr(error)
    save()
    raise
finally:
    specmod.CUDASpeculativeSession = saved_constructor
    if model is not None:
        model.close()
