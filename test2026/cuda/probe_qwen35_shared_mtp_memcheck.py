"""New pooled/split MTP integration under unfiltered memcheck.

Plain controls are immutable completed evidence. Each sanitizer request must
reproduce its complete target/MTP words and discrete work, plus the separately
loaded ordinary plain target's words/signature. No baseline model is co-resident.
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

plain_path, output_path = map(Path, sys.argv[1:3])
assert not output_path.exists()
plain = json.loads(plain_path.read_text())
assert plain['all_passed'] and [s['mode'] for s in plain['stages']] == ['baseline', 'pool', 'split', 'ordinary']
request = plain['request']
evidence = {'plain_path': str(plain_path), 'plain_sha256': hashlib.sha256(plain_path.read_bytes()).hexdigest(),
            'request': request, 'stages': [], 'events': [], 'all_passed': False,
            'performance_or_process_memory_measurement': False}


def save():
    output_path.write_text(json.dumps(evidence, indent=2) + '\n')


def event(name):
    evidence['events'].append({'name': name, 'unix_seconds': time.time(), 'pid': os.getpid()})
    save()
    print('SHARED_MTP_MEMCHECK_EVENT', name, flush=True)


def signature(result):
    return {k: result[k] for k in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                                  'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')}


def hashes(model):
    d = model.decoder
    s = model.speculative
    b = d.batch
    result = {}
    values = [('last_hidden', d.last_hidden, d.config['hidden_size']),
              ('canonical', d.state, d.programs[d.decode_chunk_size].state_width),
              ('mtp_key', s.key, s.cache_width), ('mtp_value', s.value, s.cache_width),
              ('mtp_last_hidden', s.last_hidden, s.hidden_size)]
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


saved_constructor = specmod.CUDASpeculativeSession
try:
    for mode in ('pool', 'split'):
        control = next(s for s in plain['stages'] if s['mode'] == mode)
        ordinary = next(s for s in plain['stages'] if s['mode'] == 'ordinary')
        stage = {'mode': mode, 'records': []}
        evidence['stages'].append(stage)
        event(mode + '_load_begin')
        def constructor(*args):
            return SharedMTPSession(*args, mode=mode)
        specmod.CUDASpeculativeSession = constructor
        try:
            model = Model(request['checkpoint'], request['profile'], request['model_id'])
        finally:
            specmod.CUDASpeculativeSession = saved_constructor
        try:
            event(mode + '_load_complete')
            for trial in range(2):
                for case in request['cases']:
                    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']] == case['input_sha256']
                    result = model.infer_batch(case['prompts'], case['images'], request['output_limit'])
                    state = hashes(model)
                    stage['records'].append({'style': case['style'], 'trial': trial, 'result': result, 'state': state})
                    save()
                    ref = next(r for r in control['records'] if (r['style'], r['trial']) == (case['style'], trial))
                    target = next(r for r in ordinary['records'] if (r['style'], r['trial']) == (case['style'], trial))
                    assert signature(result) == signature(ref['result']) == signature(target['result'])
                    assert state == ref['state']
                    assert all(state[k] == target['state'][k] for k in ('last_hidden', 'canonical'))
                    assert result['speculative_metrics'] == ref['result']['speculative_metrics']
                    assert not model.decoder.poisoned
                    assert model.decoder.slots == [p + s - 1 for p, s in zip(result['prompt_tokens'], result['sampled_tokens_including_stop'])]
                    print('SHARED_MTP_MEMCHECK_REQUEST', mode, trial, case['style'], flush=True)
            stage['memory'] = model.memory_info()
        finally:
            model.close()
            event(mode + '_closed')
    evidence['all_passed'] = True
    evidence['scope'] = 'Two complete pooled/split reset repeats, full plain target/MTP state and work equality. Sanitizer log audited externally; no time/VRAM/independent-quality claim.'
    save()
    print('SHARED_MTP_MEMCHECK_COMPLETE', request['model_id'], flush=True)
except Exception as error:
    evidence['failure'] = str(error)
    save()
    raise
finally:
    specmod.CUDASpeculativeSession = saved_constructor
