"""Real-image optimized static-batch admission, with a separate ordinary target.

Every request resets caches. Full finite canonical/hidden hashes invalidate
timing; these results establish neither independent quality nor performance.
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

request_path, output_path = map(Path, sys.argv[1:3])
assert not output_path.exists(), 'Preserve batch admission evidence'
request = json.loads(request_path.read_text())
batch = request['profile']['batch']
assert batch in (1, 2, 4, 8)
assert request['profile']['compactKVOutputs']
assert request['profile']['speculativeMTP'] == 3
assert request['profile']['maxInputTokens'] + request['profile']['maxOutputTokens'] == 512
evidence = {'request': request, 'events': [], 'candidate': [], 'ordinary': [],
            'all_passed': False, 'independent_quality_proven': False,
            'full_capacity_boundary_proven': False,
            'performance_or_process_memory_measurement': False}


def save():
    output_path.write_text(json.dumps(evidence, indent=2))


def event(name):
    evidence['events'].append({'event': name, 'unix_seconds': time.time(), 'pid': os.getpid()})
    save()
    print('OPTIMIZED_BATCH_EVENT', name, batch, flush=True)


def hashes(model):
    decoder = model.decoder
    result = {}
    for name, value, width in (
        ('last_hidden', decoder.last_hidden, decoder.config['hidden_size']),
        ('canonical_state', decoder.state, decoder.programs[decoder.decode_chunk_size].state_width),
    ):
        digest = hashlib.sha256()
        words = 0
        for start in range(0, width, 131072):
            count = min(131072, width - start)
            chunk = decoder.io.slice(value, start, [batch, count])
            data = decoder.io.finite_f32_bytes(chunk)
            assert len(data) == 4 * batch * count
            digest.update(data)
            words += batch * count
            del chunk, data
        assert words == batch * width
        result[name] = {'sha256': digest.hexdigest(), 'shape': [batch, width], 'finite_fp32_words': words}
    return result


signature_keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                  'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')


def signature(result):
    return {key: result[key] for key in signature_keys}


try:
    for case in request['cases']:
        assert len(case['images']) == len(case['prompts']) == batch
        assert [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in case['images']] == case['input_sha256']
    event('candidate_load_begin')
    model = Model(request['checkpoint'], request['profile'], request['model_id'])
    try:
        event('candidate_load_complete')
        assert model.decoder.compact_kv_outputs
        assert model.speculative.verifier.native_prefix is not None
        assert json.loads(model.speculative.verifier.native_prefix.stats_json())['compact_kv_outputs']
        assert all(program.output_state_width < program.state_width for program in model.decoder.programs.values())
        for trial in range(2):
            for case in request['cases']:
                result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
                assert result['profile'] == request['profile']
                assert result['speculative']['enabled'] and result['decoder_profiles']['compact_kv_outputs']
                assert result['visual_tokens_per_row'] == [60] * batch
                assert result['vision_compile_ms'] == 0
                assert model.decoder.slots == [n + s - 1 for n, s in zip(result['prompt_tokens'], result['sampled_tokens_including_stop'])]
                assert not model.decoder.poisoned
                state = hashes(model)
                row = {'style': case['style'], 'trial': trial, 'result': result, 'state': state}
                evidence['candidate'].append(row)
                save()
                if trial:
                    previous = next(r for r in evidence['candidate'] if r['style'] == case['style'] and r['trial'] == 0)
                    assert signature(previous['result']) == signature(result), 'Reset/replay changed output or stop decisions'
                    assert previous['state'] == state, 'Reset/replay changed complete state'
                    assert previous['result']['speculative_metrics'] == result['speculative_metrics'], 'Reset/replay changed discrete MTP work'
                if case['style'] == 'uniform':
                    assert all(tokens == result['generated_token_ids'][0] for tokens in result['generated_token_ids'])
                print('OPTIMIZED_BATCH_REQUEST', 'candidate', trial, case['style'], flush=True)
        evidence['candidate_memory'] = model.memory_info()
    finally:
        model.close()
        event('candidate_closed')
    # The ordinary engine is loaded only after the candidate and its MTP close.
    ordinary_profile = dict(request['profile'])
    ordinary_profile.update({'speculativeMTP': 0, 'speculativeHeadCandidates': 0,
        'speculativePrefixReplay': False, 'deferPackedStateCommit': False,
        'nativePackedPrefixCommit': False, 'sharePackedOutputStorage': False,
        'reuseVerifiedHead': False, 'internalPackedStores': False, 'compactKVOutputs': False})
    evidence['ordinary_profile'] = ordinary_profile
    event('ordinary_load_begin')
    model = Model(request['checkpoint'], ordinary_profile, request['model_id'])
    try:
        event('ordinary_load_complete')
        for case in request['cases']:
            result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
            assert not result['speculative']['enabled'] and not result['decoder_profiles']['compact_kv_outputs']
            state = hashes(model)
            row = {'style': case['style'], 'result': result, 'state': state}
            evidence['ordinary'].append(row)
            save()
            candidates = [r for r in evidence['candidate'] if r['style'] == case['style']]
            assert len(candidates) == 2
            assert all(signature(r['result']) == signature(result) for r in candidates), 'Optimized/ordinary sequence or stop mismatch'
            assert all(r['state'] == state for r in candidates), 'Optimized/ordinary complete finite state mismatch'
            print('OPTIMIZED_BATCH_REQUEST', 'ordinary', case['style'], flush=True)
        evidence['ordinary_memory'] = model.memory_info()
    finally:
        model.close()
        event('ordinary_closed')
    # Row permutations must preserve each row's prompt/output/stop result.
    for role in ('candidate', 'ordinary'):
        rows = evidence[role]
        for permuted in [r for r in rows if r['style'] == 'ragged_permuted']:
            original = next(r for r in rows if r['style'] == 'ragged' and r.get('trial') == permuted.get('trial'))
            for key in signature_keys:
                assert permuted['result'][key] == list(reversed(original['result'][key])), ('Row permutation', role, key)
    evidence['all_passed'] = True
    event('complete')
except BaseException as error:
    evidence['failure'] = {'type': type(error).__name__, 'message': str(error)}
    save()
    raise
