"""B1 native MTP study worker. Concurrent HTTP clients queue at the transport.

This does not admit continuous speculative batching or auxiliary prefix reuse.
All draft/verify/correct work runs inside the native generation transaction.
"""
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import garnet
from xModel.qwen38.vl_27b.metal_mtp_study import MetalMTPStudy


def emit(value):
    raw = (json.dumps(value, ensure_ascii=True, separators=(',', ':'))+'\n').encode()
    while raw:
        count = os.write(1, raw)
        if count < 1:
            raise OSError('Native MTP event pipe closed')
        raw = raw[count:]


def requests():
    pending = b''
    while True:
        block = os.read(0, 65536)
        if not block:
            if pending:
                raise ValueError('Incomplete native MTP request frame')
            return
        pending += block
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            if len(line) > 2*1024*1024:
                raise ValueError('Native MTP request frame too large')
            yield json.loads(line)
        if len(pending) > 2*1024*1024:
            raise ValueError('Native MTP request frame too large')


folder = sys.argv[3]
with open(sys.argv[4]) as stream:
    profile = json.load(stream)
required = dict(batch=1, capacity=4096, target_prefill_chunk=128,
                auxiliary_prefill_chunk=128, auxiliary_draft_chunk=1,
                verification_chunk=4, max_input_tokens=3072, max_output_tokens=1024,
                prefix_cache=False, multimodal=False, native_continuous_batching=False)
for name, expected in required.items():
    if type(profile.get(name)) is not type(expected) or profile[name] != expected:
        raise ValueError('Unsupported MTP study profile: '+name)
if not isinstance(profile.get('profile_id'), str) or not profile['profile_id']:
    raise ValueError('Explicit MTP study identity required')
with open(folder+'/config.json') as stream:
    config = json.load(stream)
study = None
try:
    study = MetalMTPStudy(folder+'/model.safetensors.index.json', config, profile['auxiliary_checkpoint'], profile['capacity'])
    tokenizer = garnet.tokenizer()
    tokenizer.open(folder)
    stops = [tokenizer.token_id('<|im_end|>'), tokenizer.token_id('<|endoftext|>')]
    with open(profile['memory_inventory_path'], 'w') as stream:
        stream.write(study.inventory())
    emit(dict(event='ready', backend=garnet.backend_info(), active_requests=1, capacity=profile['capacity'],
              profile_id=profile['profile_id'], prefix_cache=False, multimodal=False,
              native_continuous_batching=False, queued_concurrency=True,
              target_prefill_chunk=128, auxiliary_prefill_chunk=128, auxiliary_draft_chunk=1,
              verification_chunk=4, max_input_tokens=profile['max_input_tokens'],
              max_output_tokens=profile['max_output_tokens'], memory=json.loads(garnet.memory_stats_json())))
    for request in requests():
        key = request['id']
        body = request['body']
        try:
            if not isinstance(body.get('rendered_prompt'), str):
                raise ValueError('MTP study requires a rendered text prompt')
            ids = list(tokenizer.encode(body['rendered_prompt']))
            limit = body['max_tokens']
            if (type(limit) is not int or not 1 <= limit <= profile['max_output_tokens'] or
                not ids or len(ids) > profile['max_input_tokens'] or len(ids)+limit > profile['capacity']):
                raise ValueError('Native MTP request exceeds input/output/context profile bounds')
        except (ValueError, KeyError) as error:
            emit(dict(event='error', id=key, error=str(error), fatal=False))
            continue
        try:
            study.reset()
            known = study.prefill(ids)
            emit(dict(event='mtp_prefill', id=key, actual_input_tokens=len(ids), stats=json.loads(study.session.mtp_stats_json())))
            generated = []
            text_ids = []
            text = ''
            reason = 'length'

            def token_event(token):
                global text
                if type(token) is not int or not 0 <= token < config['text_config']['vocab_size']:
                    raise ValueError('Native MTP token outside vocabulary')
                generated.append(token)
                if token not in stops:
                    text_ids.append(token)
                text = tokenizer.decode(text_ids).encode('utf8').decode('utf8', 'replace')
                emit(dict(event='token', id=key, token_id=token, sampled_tokens=len(generated),
                          prompt_tokens=len(ids), cached_tokens=0, text=text))

            token_event(known)
            transactions = proposed = accepted = 0
            while len(generated) < limit and generated[-1] not in stops:
                remaining = limit-len(generated)
                result = study.session.mtp_generate([generated[-1]], [remaining], stops, None)[0]
                stats = json.loads(study.session.mtp_generation_stats_json())
                if result[1] != len(result[0]) or not 1 <= result[1] <= min(4, remaining):
                    raise ValueError('Native MTP consumption/budget mismatch')
                if result[2] != (result[0][-1] in stops):
                    raise ValueError('Native MTP stop mismatch')
                emit(dict(event='mtp_transaction', id=key, index=transactions, remaining_before=remaining,
                          emitted=result[0], eos=result[2], stats=stats))
                for token in result[0]:
                    token_event(token)
                transactions += 1
                proposed += stats['proposed_drafts']
                accepted += stats['accepted_drafts']
            if generated[-1] in stops:
                reason = 'stop'
            emit(dict(event='completed', id=key, generated_ids=generated, prompt_ids=ids, text=text,
                      cached_tokens=0, finish_reason=reason, profile_id=profile['profile_id'],
                      generation_mode='native_metal_mtp_single_active',
                      speculative_metrics=dict(native_transactions=transactions, proposed_drafts=proposed, accepted_drafts=accepted),
                      memory=json.loads(garnet.memory_stats_json())))
        except Exception as error:
            emit(dict(event='error', id=key, error=str(error), fatal=True))
            raise
finally:
    if study is not None:
        study.close()
    garnet.shutdown()
    memory = json.loads(garnet.memory_stats_json())
    emit(dict(event='closed', memory=memory))
    for name in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners', 'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        if memory[name] != 0:
            raise ValueError('Native MTP owner leak: '+name)
