"""Native parent-pool ownership and lifecycle, compared to immutable controls."""
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

plain_path, output_path = map(Path, sys.argv[1:3])
assert not output_path.exists(), 'Preserve parent-pool evidence'
plain = json.loads(plain_path.read_text())
assert plain['all_passed']
control = next(s for s in plain['stages'] if s['mode'] == 'split')
ordinary = next(s for s in plain['stages'] if s['mode'] == 'ordinary')
request = plain['request']
batch = request['profile']['batch']
evidence = {'plain_control': str(plain_path), 'plain_sha256': hashlib.sha256(plain_path.read_bytes()).hexdigest(),
            'request': request, 'records': [], 'events': [], 'lifecycle': {}, 'all_passed': False,
            'independent_quality_proven': False, 'performance_or_process_memory_measurement': False}


def save():
    output_path.write_text(json.dumps(evidence, indent=2) + '\n')


def event(name):
    evidence['events'].append({'name': name, 'unix_seconds': time.time(), 'pid': os.getpid()})
    save()
    print('PARENT_MTP_EVENT', name, flush=True)


def signature(result):
    return {k: result[k] for k in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                                  'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')}


def tensor_hash(io, tensor, width):
    flat = io.storage_view(tensor, [batch, width])
    digest = hashlib.sha256()
    for start in range(0, width, 131072):
        count = min(131072, width - start)
        chunk = io.slice(flat, start, [batch, count])
        data = io.finite_f32_bytes(chunk)
        assert len(data) == 4 * batch * count
        digest.update(data)
        del chunk, data
    return {'sha256': digest.hexdigest(), 'finite_fp32_words': batch * width}


def states(model):
    d, s = model.decoder, model.speculative
    values = [('last_hidden', d.last_hidden, d.config['hidden_size']),
              ('canonical', d.state, d.programs[d.decode_chunk_size].state_width),
              ('mtp_key', s.key, s.cache_width), ('mtp_value', s.value, s.cache_width),
              ('mtp_last_hidden', s.last_hidden, s.hidden_size)]
    return {name: tensor_hash(d.io, value, width) for name, value, width in values}


def protected_state(model):
    d = model.decoder
    result = states(model)
    width = d.io.tensor_bytes(d.output_pool) // (batch * 4)
    result['parent_output'] = tensor_hash(d.io, d.output_pool, width)
    for name, value, width in [('target_input', d.hidden, d.chunk_size * d.config['hidden_size']),
                               ('target_active', d.active, d.chunk_size),
                               ('target_position', d.position, d.chunk_size * 3),
                               ('target_slots', d.slot_tensor, d.chunk_size)]:
        result[name] = tensor_hash(d.io, value, width)
    return result


def graph_counts(session):
    return {str(w): {k: g[k] for k in ('graph_captures', 'graph_replays', 'contexts')}
            for w, p in session.experiment_programs.items() for g in [json.loads(p.graph.stats_json())]}


def reject_without_write(model, name):
    s, d = model.speculative, model.decoder
    before = protected_state(model)
    work = json.dumps(s.metrics, sort_keys=True)
    graphs = graph_counts(s)
    try:
        s._call([], [], False, 'draft')
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError('Unsafe parent-pool call was accepted: ' + name)
    assert protected_state(model) == before
    assert json.dumps(s.metrics, sort_keys=True) == work and graph_counts(s) == graphs
    return {'kind': name, 'error': message, 'full_finite_protected_words_unchanged': True,
            'mtp_graph_counts_and_work_unchanged': True}


saved_constructor = specmod.CUDASpeculativeSession
model = None
try:
    for case in request['cases']:
        assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']] == case['input_sha256']
    event('parent_load_begin')
    def constructor(*args):
        return SharedMTPSession(*args, mode='split', parent_storage=True)
    specmod.CUDASpeculativeSession = constructor
    try:
        model = Model(request['checkpoint'], request['profile'], request['model_id'])
    finally:
        specmod.CUDASpeculativeSession = saved_constructor
    event('parent_load_complete')
    s, d = model.speculative, model.decoder
    assert s.parent_storage and s.activation_pool is d.activation_pool and s.output_pool is d.output_pool
    for trial in range(2):
        for case in request['cases']:
            result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
            state = states(model)
            evidence['records'].append({'style': case['style'], 'trial': trial, 'result': result, 'state': state})
            save()
            ref = next(r for r in control['records'] if (r['style'], r['trial']) == (case['style'], trial))
            target = next(r for r in ordinary['records'] if (r['style'], r['trial']) == (case['style'], trial))
            assert signature(result) == signature(ref['result']) == signature(target['result'])
            assert state == ref['state']
            assert all(state[k] == target['state'][k] for k in ('last_hidden', 'canonical'))
            assert result['speculative_metrics'] == ref['result']['speculative_metrics']
            assert d.slots == [p + n - 1 for p, n in zip(result['prompt_tokens'], result['sampled_tokens_including_stop'])]
            assert not d.poisoned
            print('PARENT_MTP_REQUEST', trial, case['style'], flush=True)
    memory = model.memory_info()
    evidence['memory'] = memory
    ms, md = memory['speculative'], memory['decoder']
    old = control['memory']
    assert ms['mtp_parent_storage'] and ms['mtp_owned_activation_pool_bytes'] == ms['mtp_owned_output_pool_bytes'] == 0
    assert ms['mtp_borrowed_workspace_capacity_bytes'] == md['shared_activation_pool_bytes']
    assert ms['mtp_borrowed_output_capacity_bytes'] == md['shared_output_pool_bytes']
    assert md['owned_decoder_bytes'] == old['decoder']['owned_decoder_bytes']
    removed = old['speculative']['mtp_shared_activation_bytes'] + old['speculative']['mtp_shared_output_bytes']
    assert memory['owned_model_bytes'] == old['owned_model_bytes'] - removed
    for g in ms['mtp_experiment_graphs'].values():
        assert g['external_workspace_capacity_bytes'] == md['shared_activation_pool_bytes']
        assert g['external_output_storage'] and g['owned_device_output_bytes'] == 0 and g['contexts'] == 1
        assert g['external_output_storage_span_bytes'] >= g['device_output_bytes']
    evidence['removed_exclusive_pool_bytes'] = removed
    evidence['lifecycle']['same_python_parent_root_objects'] = True
    # A poisoned session/target must be rejected before touching shared outputs.
    s.poisoned = True
    try: evidence['lifecycle']['session_poison_guard'] = reject_without_write(model, 'session_poisoned')
    finally: s.poisoned = False
    d.poisoned = True
    try: evidence['lifecycle']['target_poison_guard'] = reject_without_write(model, 'target_poisoned')
    finally: d.poisoned = False
    initial = protected_state(model)
    saved_slots = list(d.slots)
    s.verifier.verify([list(s.current)], [[1] * batch], [[[p] * 3] for p in s.current_positions])
    assert s.verifier.pending and d.state_transaction is s.verifier
    evidence['lifecycle']['pending_guard'] = reject_without_write(model, 'real_pending_target_verification')
    s.verifier.rollback()
    after = protected_state(model)
    assert all(after[k] == v for k, v in initial.items() if k != 'parent_output')
    assert d.slots == saved_slots and d.state_transaction is None and not s.verifier.pending
    evidence['lifecycle']['rollback_restores_all_target_and_mtp_words_except_disposable_packed_output'] = True
    case = request['cases'][0]
    replay = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
    replay_state = states(model)
    ref = next(r for r in control['records'] if r['style'] == case['style'] and r['trial'] == 0)
    assert signature(replay) == signature(ref['result']) and replay_state == ref['state']
    assert replay['speculative_metrics'] == ref['result']['speculative_metrics']
    evidence['lifecycle']['reset_after_rollback'] = {'result': replay, 'state': replay_state}
    # Close through a real pending transaction. Retaining this explicit root
    # intentionally keeps its allocation alive for the following provider read.
    s.verifier.verify([list(s.current)], [[1] * batch], [[[p] * 3] for p in s.current_positions])
    retained = d.output_pool
    io_keep = d.io
    retained_width = io_keep.tensor_bytes(retained) // (batch * 4)
    retained_before = tensor_hash(io_keep, retained, retained_width)
    verifier_keep = s.verifier
    model.close()
    assert not verifier_keep.pending and d.state_transaction is None
    assert tensor_hash(io_keep, retained, retained_width) == retained_before
    evidence['lifecycle']['pending_close_rolls_back_and_retained_output_provider_survives'] = retained_before
    try: s._call([], [], False, 'draft')
    except ValueError as error: evidence['lifecycle']['closed_call_error'] = str(error)
    else: raise AssertionError('Closed parent-backed session accepted execution')
    del retained, io_keep, verifier_keep
    event('parent_closed')
    evidence['all_passed'] = True
    evidence['scope'] = ('Two reset repeats of four real-image cases plus pending/poison/no-write/rollback/reset/close/provider '
                         'controls. Complete finite target/MTP words match immutable split and ordinary controls. '
                         'Owned pool deltas counted once; held-output close read intentionally retains one allocation. '
                         'Not independent quality, process VRAM, lifetime peak, latency or concurrent-request acceptance.')
    save()
    print('PARENT_MTP_NATIVE_COMPLETE', request['model_id'], flush=True)
except Exception as error:
    evidence['failure'] = str(error)
    save()
    raise
finally:
    specmod.CUDASpeculativeSession = saved_constructor
    if model is not None: model.close()
