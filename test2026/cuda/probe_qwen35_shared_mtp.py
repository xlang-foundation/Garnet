"""Sequential real-image control for unwired shared-weight MTP sizing."""
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

request_path, output_path = map(Path, sys.argv[1:3])
assert not output_path.exists(), 'Preserve MTP control evidence'
request = json.loads(request_path.read_text())
evidence = {'request': request, 'stages': [], 'events': [], 'all_passed': False,
            'performance_or_process_memory_measurement': False, 'independent_quality_proven': False}


def save():
    output_path.write_text(json.dumps(evidence, indent=2) + '\n')


def event(name):
    evidence['events'].append({'name': name, 'unix_seconds': time.time(), 'pid': os.getpid()})
    save()
    print('SHARED_MTP_EVENT', name, flush=True)


def signature(result):
    return {k: result[k] for k in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                                  'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')}


def hashes(model, mtp=True):
    d = model.decoder
    b = d.batch
    values = [('last_hidden', d.last_hidden, d.config['hidden_size']),
              ('canonical', d.state, d.programs[d.decode_chunk_size].state_width)]
    if mtp:
        s = model.speculative
        values += [('mtp_key', s.key, s.cache_width), ('mtp_value', s.value, s.cache_width),
                   ('mtp_last_hidden', s.last_hidden, s.hidden_size)]
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


for case in request['cases']:
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']] == case['input_sha256']

saved_constructor = specmod.CUDASpeculativeSession
try:
    for mode in ('baseline', 'pool', 'split', 'ordinary'):
        profile = dict(request['profile'])
        if mode == 'ordinary':
            profile.update({'speculativeMTP': 0, 'speculativeHeadCandidates': 0, 'speculativePrefixReplay': False,
                            'deferPackedStateCommit': False, 'nativePackedPrefixCommit': False,
                            'sharePackedOutputStorage': False, 'reuseVerifiedHead': False,
                            'internalPackedStores': False, 'compactKVOutputs': False})
        stage = {'mode': mode, 'profile': profile, 'records': []}
        evidence['stages'].append(stage)
        event(mode + '_load_begin')
        if mode in ('pool', 'split'):
            def constructor(*args):
                return SharedMTPSession(*args, mode=mode)
            specmod.CUDASpeculativeSession = constructor
        else:
            specmod.CUDASpeculativeSession = saved_constructor
        try:
            model = Model(request['checkpoint'], profile, request['model_id'])
        finally:
            specmod.CUDASpeculativeSession = saved_constructor
        try:
            event(mode + '_load_complete')
            assert model.decoder.decode_chunk_size == 4 and model.decoder.prefill_chunk_size == 64
            for trial in range(2):
                for case in request['cases']:
                    result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
                    state = hashes(model, mode != 'ordinary')
                    row = {'style': case['style'], 'trial': trial, 'result': result, 'state': state}
                    stage['records'].append(row)
                    save()
                    assert model.decoder.slots == [p + s - 1 for p, s in zip(result['prompt_tokens'], result['sampled_tokens_including_stop'])]
                    assert not model.decoder.poisoned
                    if trial:
                        first = next(x for x in stage['records'] if x['style'] == case['style'] and x['trial'] == 0)
                        assert signature(first['result']) == signature(result) and first['state'] == state
                        if mode != 'ordinary':
                            assert first['result']['speculative_metrics'] == result['speculative_metrics']
                    if mode != 'baseline':
                        control = next(x for x in evidence['stages'][0]['records'] if x['style'] == case['style'] and x['trial'] == trial)
                        assert signature(control['result']) == signature(result), 'Changed target token/stop/prompt output'
                        assert all(control['state'][k] == state[k] for k in ('last_hidden', 'canonical')), 'Changed full target state'
                        if mode == 'pool':
                            assert control['state'] == state, 'Shared weights changed MTP cache/hidden'
                            assert control['result']['speculative_metrics'] == result['speculative_metrics'], 'Pool changed discrete work'
                        elif mode == 'split':
                            # Final first proposal adds one prefill call after shifted prompt chunks.
                            expected = 1 + sum((min(64, max(result['prompt_tokens']) - start) + 15) // 16
                                               for start in range(0, max(result['prompt_tokens']), 64))
                            assert result['speculative_metrics']['mtp_prefill_calls'] == expected
                    print('SHARED_MTP_REQUEST', mode, trial, case['style'], flush=True)
            stage['memory'] = model.memory_info()
            if mode in ('pool', 'split'):
                s = stage['memory']['speculative']
                assert s['mtp_shared_projection_bytes_in_parent_io_pool'] > 0
                assert all(g['external_bf16_weight_inputs'] == 8 for g in s['mtp_experiment_graphs'].values())
        finally:
            model.close()
            event(mode + '_closed')
    evidence['all_passed'] = True
    evidence['scope'] = ('Two reset repeats of four real-image cases in each separately loaded baseline, pooled D4, '
                         'split MTP P16/D1/C4 and ordinary non-MTP target. Complete finite target state hashes; '
                         'pool preserves all MTP words/work. Split may change proposals, but target output/state must match. '
                         'No independent quality, full-capacity, performance or process-memory acceptance.')
    save()
    print('SHARED_MTP_CONTROL_COMPLETE', request['model_id'], request['profile']['batch'], flush=True)
except Exception as error:
    evidence['failure'] = str(error)
    save()
    raise
finally:
    specmod.CUDASpeculativeSession = saved_constructor
