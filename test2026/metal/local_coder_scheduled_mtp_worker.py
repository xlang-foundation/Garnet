"""Research B2 HTTP worker: native leases, scheduling and verified generation.

CPU code handles tokenization, embeddings and stream events. The native policy
owns admission, physical rows, known tokens, budgets and committed progress.
Prefix reuse and multimodal inputs are outside this finite serving profile.
"""
import json
import os
import queue
import sys
import threading
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import garnet
from xModel.qwen38.vl_27b.metal_mtp_study import MetalMTPStudy


def emit(event):
    wire = (json.dumps(event, ensure_ascii=True, separators=(',', ':'))+'\n').encode()
    while wire:
        count = os.write(1, wire)
        if count < 1:
            raise OSError('Native scheduled MTP event pipe closed')
        wire = wire[count:]


def main():
    folder = sys.argv[3]
    with open(sys.argv[4]) as stream:
        profile = json.load(stream)
    required = dict(batch=2, capacity=4096, target_prefill_chunk=128,
                    auxiliary_prefill_chunk=64, auxiliary_draft_chunk=0,
                    verification_chunk=2, max_input_tokens=3072,
                    max_output_tokens=1024, prefix_cache=False, multimodal=False,
                    native_continuous_batching=True)
    for name, expected in required.items():
        if type(profile.get(name)) is not type(expected) or profile[name] != expected:
            raise ValueError('Unsupported scheduled MTP profile: '+name)
    if not isinstance(profile.get('profile_id'), str) or not profile['profile_id']:
        raise ValueError('Explicit scheduled MTP identity required')
    for tag in ('native_mtp_continuous_scheduler', 'native_mtp_kv_inplace',
                'native_mtp_deferred_packs', 'native_prefill_hidden_pool', 'linear_chunk_state_inplace'):
        if tag not in garnet.backend_info():
            raise ValueError('Scheduled MTP extension missing: '+tag)
    if 'prefill_alias_test_faults' in garnet.backend_info():
        raise ValueError('Fault-injection builds cannot serve this profile')
    with open(folder+'/config.json') as stream:
        config = json.load(stream)
    incoming = queue.Queue(maxsize=16)
    input_closed = threading.Event()
    input_errors = []
    study = policy = None
    jobs = {}
    live_keys = set()
    frame_index = tick_index = 0

    def read_requests():
        pending = b''
        try:
            while True:
                block = os.read(0, 65536)
                if not block:
                    if pending:
                        raise ValueError('Incomplete scheduled MTP request frame')
                    return
                pending += block
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if len(line) > 2*1024*1024:
                        raise ValueError('Scheduled MTP request frame too large')
                    incoming.put(json.loads(line))
                if len(pending) > 2*1024*1024:
                    raise ValueError('Scheduled MTP request frame too large')
        except Exception as error:
            input_errors.append(repr(error))
        finally:
            input_closed.set()

    try:
        study = MetalMTPStudy(folder+'/model.safetensors.index.json', config,
                              profile['auxiliary_checkpoint'], profile['capacity'], batch=2)
        policy = garnet.continuous_scheduler()
        policy.initialize(study.session, 16)
        if not policy.set_prefill_chunk(128):
            raise ValueError('Native scheduled prefill profile rejected')
        tokenizer = garnet.tokenizer()
        tokenizer.open(folder)
        stops = [tokenizer.token_id('<|im_end|>'), tokenizer.token_id('<|endoftext|>')]
        with open(profile['memory_inventory_path'], 'x') as stream:
            stream.write(study.inventory())
        reader = threading.Thread(target=read_requests, daemon=True)
        reader.start()
        emit(dict(event='ready', backend=garnet.backend_info(), active_requests=2,
                  capacity=4096, profile_id=profile['profile_id'], prefix_cache=False,
                  multimodal=False, native_continuous_batching=True, queued_concurrency=True,
                  target_prefill_chunk=128, auxiliary_prefill_chunk=64, auxiliary_draft_chunk=0,
                  verification_chunk=2, max_input_tokens=3072, max_output_tokens=1024,
                  memory=json.loads(garnet.memory_stats_json())))

        def admit(request):
            if not isinstance(request, dict):
                raise ValueError('Scheduled MTP request must be an object')
            key = request.get('id')
            if not isinstance(key, str) or not key or len(key) > 128 or key in live_keys:
                raise ValueError('Invalid or duplicate scheduled MTP identity')
            try:
                body = request['body']
                prompt, limit = body['rendered_prompt'], body['max_tokens']
                if not isinstance(prompt, str) or type(limit) is not int:
                    raise ValueError('Scheduled MTP requires a rendered text prompt and integer budget')
                ids = list(tokenizer.encode(prompt))
                if (not ids or len(ids) > profile['max_input_tokens'] or
                    not 1 <= limit <= profile['max_output_tokens'] or len(ids)+limit > profile['capacity']):
                    raise ValueError('Native MTP request exceeds input/output/context profile bounds')
            except (ValueError, KeyError, TypeError) as error:
                emit(dict(event='error', id=key, error=str(error), fatal=False))
                return
            identity = policy.submit(len(ids), limit, 0)
            if identity <= 0:
                emit(dict(event='error', id=key, error='Native scheduled MTP pending limit reached', fatal=False))
                return
            jobs[identity] = dict(id=key, prompt=ids, limit=limit, generated=[], text_ids=[],
                                  text='', native_transactions=0, proposed_drafts=0, accepted_drafts=0)
            live_keys.add(key)
            emit(dict(event='mtp_admitted', id=key, native_id=identity,
                      prompt_tokens=len(ids), max_output=limit))

        def token_event(identity, token):
            job = jobs[identity]
            if type(token) is not int or not 0 <= token < config['text_config']['vocab_size']:
                raise ValueError('Scheduled MTP token outside vocabulary')
            if len(job['generated']) >= job['limit']:
                raise ValueError('Scheduled MTP exceeded output budget')
            job['generated'].append(token)
            if token not in stops:
                job['text_ids'].append(token)
            job['text'] = tokenizer.decode(job['text_ids']).encode('utf8').decode('utf8', 'replace')
            emit(dict(event='token', id=job['id'], native_id=identity, token_id=token,
                      sampled_tokens=len(job['generated']), prompt_tokens=len(job['prompt']),
                      cached_tokens=0, text=job['text']))

        def finish(identity):
            job = jobs[identity]
            if not job['generated'] or (job['generated'][-1] not in stops and len(job['generated']) != job['limit']):
                raise ValueError('Scheduled MTP completion is neither EOS nor full budget')
            emit(dict(event='completed', id=job['id'], native_id=identity,
                      generated_ids=job['generated'], prompt_ids=job['prompt'], text=job['text'],
                      cached_tokens=0, finish_reason='stop' if job['generated'][-1] in stops else 'length',
                      profile_id=profile['profile_id'], generation_mode='native_metal_mtp_continuous_batch2',
                      speculative_metrics={k:job[k] for k in ('native_transactions','proposed_drafts','accepted_drafts')},
                      memory=json.loads(garnet.memory_stats_json())))
            live_keys.remove(job['id'])
            del jobs[identity]

        while True:
            if input_errors:
                raise ValueError('Scheduled MTP input failure: '+input_errors[0])
            if not jobs and incoming.empty():
                if input_closed.is_set():
                    break
                try:
                    admit(incoming.get(timeout=.05))
                except queue.Empty:
                    continue
            while not incoming.empty():
                admit(incoming.get_nowait())
            if not jobs:
                continue
            tick = json.loads(policy.next_json())
            if not tick['batches']:
                raise ValueError('Native scheduled MTP stalled with live requests')
            emit(dict(event='mtp_schedule', index=tick_index, plan=tick))
            for batch in tick['batches']:
                if batch['stage'] == 'prefill':
                    predicted = study.prefill_scheduled(batch['requests'], jobs)
                    for request in batch['requests']:
                        identity = request['id']
                        job = jobs[identity]
                        final = request['prefill_offset']+request['prefill_tokens'] == len(job['prompt'])
                        emit(dict(event='mtp_prefill_chunk', id=job['id'], native_id=identity,
                                  tick=tick_index, row=request['row'], epoch=request['epoch'],
                                  offset=request['prefill_offset'], tokens=request['prefill_tokens'], final=final))
                        if final:
                            token = predicted[request['row']]
                            stage = policy.complete_prefill(identity, token, token in stops)
                            if stage not in (2, 3):
                                raise ValueError('Native prefill completion rejected')
                            token_event(identity, token)
                            if stage == 3:
                                finish(identity)
                        elif not policy.complete_prefill_chunk(identity, request['prefill_tokens']):
                            raise ValueError('Native prefill progress rejected')
                elif batch['stage'] == 'decode':
                    result = json.loads(policy.mtp_generate_json(stops))
                    auxiliary = json.loads(study.session.mtp_stats_json())
                    hidden_bindings = study.session.prefill_hidden_pool_bindings
                    linear_updates = study.session.linear_state_inplace_updates
                    if (result['generation']['recursive_draft_calls'] != 0 or
                        result['generation']['correction_profile_chunk'] != 2 or
                        auxiliary['kv_inplace_updates'] != 2 or hidden_bindings != 64 or linear_updates != 96):
                        raise ValueError('Scheduled MTP executed an unqualified native profile')
                    emit(dict(event='mtp_batch_transaction', index=frame_index, tick=tick_index,
                              result=result, auxiliary=auxiliary, hidden_pool_bindings=hidden_bindings,
                              linear_state_inplace_updates=linear_updates))
                    for item in result['requests']:
                        identity = item['id']
                        job = jobs[identity]
                        count = len(item['emitted'])
                        if (item['known'] != job['generated'][-1] or
                            item['remaining_before'] != job['limit']-len(job['generated']) or
                            item['consumed'] != count or not 1 <= count <= 2 or
                            item['eos'] != (item['emitted'][-1] in stops)):
                            raise ValueError('Native scheduled MTP progress/budget mismatch')
                        emit(dict(event='mtp_transaction', id=job['id'], native_id=identity,
                                  frame=frame_index, transaction=item))
                        for token in item['emitted']:
                            token_event(identity, token)
                        job['native_transactions'] += 1
                        job['proposed_drafts'] += result['generation']['submitted_lengths'][item['row']]-1
                        job['accepted_drafts'] += count-1
                        if item['finished']:
                            finish(identity)
                    frame_index += 1
                else:
                    raise ValueError('Unexpected native scheduled stage')
            tick_index += 1
        status = json.loads(policy.status_json())
        if any(status[k] for k in ('active', 'queued', 'tracked', 'inflight')):
            raise ValueError('Scheduled MTP shutdown has live native requests')
    finally:
        if policy is not None:
            policy.close()
        if study is not None:
            study.close()
        garnet.shutdown()
        memory = json.loads(garnet.memory_stats_json())
        emit(dict(event='closed', memory=memory))
        for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners',
                     'garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
            if memory[name] != 0:
                raise ValueError('Native scheduled MTP owner leak: '+name)


main()
