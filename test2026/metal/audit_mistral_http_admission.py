#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit uncached Mistral B4 HTTP admission against frozen inputs and controls.

Checks saved native scheduling/dispatch traces and end-to-end SSE token identity.
These logs do not independently observe GPU KV contents, dynamic arrivals,
peak memory, coding quality, or repeated performance.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path


OWNERS = ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
          'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(events, clients, controls, profile, exit_status, expected_requests=22, control_prefixes=False,
             maximum_chunk=128,prefix_policy=False,required_active_rows=4):
    require(exit_status == dict(returncode=0, expected_close=True), 'unclean worker exit')
    counts = collections.Counter(e['event'] for e in events)
    require(counts['ready'] == counts['closed'] == counts['native_dense_schedule_bound'] == 1,
            'missing/duplicate lifecycle event')
    require(events[-1]['event'] == 'closed', 'events after close')
    require(not counts['error'], 'native error')
    bound = next(e for e in events if e['event'] == 'native_dense_schedule_bound')
    require((bound['physical_rows'], bound['global_token_budget'], bound['maximum_chunk']) == (4, 512, 128)
            and bound['oversized_chunk_rejected'] is True and bound['whole_prompt_rejected'] is True,
            'incorrect compiled scheduler bound')
    ready = next(e for e in events if e['event'] == 'ready')
    require(ready['profile'] == profile and profile['model_family'] == 'mistral_7b'
            and profile['capacity'] == 4096 and profile['batch'] == 4
            and profile['prefill_chunk'] == maximum_chunk and profile['prefix_cache'] is prefix_policy
            and ready['prefix_cache'] is prefix_policy and ready['multimodal'] is False,
            'incorrect serving profile')
    backend = {p.strip() for p in ready['backend'].split(';')}
    require({'prepared_dense_prefill', 'dense_prefill_max_tokens=512',
             'indexed_token_projections', 'active_decode_projections',
             'affine_coefficient_dtype=fp16'} <= backend, 'incorrect native backend')
    matrix_attention = profile.get('chunk_attention_mps', False)
    active_requests = profile.get('chunk_attention_active_rows', False)
    require(type(active_requests) is bool and active_requests == ('chunk_attention_active_rows' in backend)
            and (not active_requests or matrix_attention), 'physical request schedule differs from backend')
    reuse64 = profile.get('require_reuse64', False)
    require(type(reuse64) is bool and reuse64 == ('mpp_affine_reuse64_optional' in backend),
            'cooperative64 policy differs from backend')
    if reuse64:
        threshold = profile.get('reuse64_min_outputs')
        require(type(threshold) is int and threshold in (4096, 5120)
                and f'mpp_affine_reuse64_min_outputs={threshold}' in backend
                and profile.get('reuse64_relaxed') is False and 'mpp_affine_reuse64_relaxed=0' in backend
                and profile.get('expected_body_reuse64_projections') == (160 if threshold == 4096 else 64),
                'incorrect compiled cooperative64 profile')
    if prefix_policy:
        require(profile.get('native_cached_chunks') is True and 'native_cached_prefill_scheduler' in backend,
                'native cached-progress extension/profile missing')
    require(matrix_attention == ('chunk_attention_mps' in backend), 'attention policy differs from backend')
    if matrix_attention:
        require(profile.get('chunk_attention_buckets') is True and profile.get('chunk_attention_index32') is True
                and profile['expected_chunk_attention_dispatches'] == 32, 'incorrect attention profile')
        require(counts['native_chunk_attention_mps_prepared'] == counts['native_chunk_attention_buckets_prepared'] == 1,
                'missing prepared attention inventory')
    leases, rows, epochs, progress = {}, {}, [0]*4, {}
    chunks, tokens, completed = collections.defaultdict(list), collections.defaultdict(list), {}
    cached_prefixes,token_cached = {},{}
    stage_counts = collections.Counter()
    profile_counts = collections.Counter()
    pending = None
    pending_fresh = False
    fresh_prefill_batches = 0
    attention_checks, attention_expected = set(), None
    maximum_active = 0
    reuse64_expected = None
    for e in events:
        kind = e['event']
        if 'memory' in e:
            m = e['memory']
            require(0 <= m['metal_allocated_bytes'] <= m['metal_recommended_working_set_bytes']-256*1024*1024,
                    'sampled working-set bound exceeded')
        if kind == 'native_batch':
            require(pending is None, 'batch missing affine coverage')
            require(reuse64_expected is None, 'batch missing cooperative64 coverage')
            stage = e['stage']
            require(stage in ('prefill', 'decode'), 'unexpected schedule stage')
            requests = e['requests']
            physical = [r['row'] for r in requests]
            require(requests and len(set(physical)) == len(physical)
                    and all(type(r) is int and 0 <= r < 4 for r in physical)
                    and e['active_rows'] == [int(r in physical) for r in range(4)], 'invalid active rows')
            maximum_active = max(maximum_active, len(requests))
            if stage == 'prefill':
                require(sum(r['prefill_tokens'] for r in requests) <= 512, 'global prefill budget exceeded')
            for r in requests:
                identity, row, epoch = r['id'], r['row'], r['epoch']
                cached = r.get('cached_tokens',0)
                require(type(cached) is int and 0 <= cached <= r['prompt_tokens'] and (prefix_policy or cached == 0),
                        'invalid native restored prefix')
                if identity not in leases:
                    require(stage == 'prefill' and r['prefill_offset'] == cached
                            and row not in rows and epoch > epochs[row], 'stale/overlapping row lease')
                    leases[identity] = (row, epoch)
                    rows[row], epochs[row], progress[identity] = identity, epoch, cached
                    cached_prefixes[identity] = cached
                require(cached_prefixes[identity] == cached, 'native reused prefix changed')
                require(leases[identity] == (row, epoch) and rows.get(row) == identity,
                        'row identity/epoch changed')
                if stage == 'prefill':
                    cached_head = prefix_policy and r['prefill_tokens'] == 0 and cached == r['prompt_tokens']
                    require((0 < r['prefill_tokens'] <= maximum_chunk or cached_head) and r['prefill_offset'] == progress[identity]
                            and r['slot'] == r['prefill_offset'] and r['generated_tokens'] == 0
                            and r['prefill_offset']+r['prefill_tokens'] <= r['prompt_tokens'],
                            'invalid prefill position/chunk')
                    progress[identity] += r['prefill_tokens']
                    chunks[identity].append((r['prefill_offset'], r['prefill_tokens'],
                                             progress[identity] == r['prompt_tokens']))
                else:
                    require(progress[identity] >= r['prompt_tokens']
                            and r['slot'] == progress[identity]
                            and r['generated_tokens'] == r['slot']-r['prompt_tokens']+1,
                            'invalid decode position')
                    progress[identity] += 1
            pending = stage
            pending_fresh = stage != 'prefill' or any(r['prefill_tokens'] for r in requests)
            if stage == 'prefill' and pending_fresh:fresh_prefill_batches += 1
            if matrix_attention and pending_fresh:
                active = sum(r['prefill_tokens'] for r in requests) if stage == 'prefill' else len(requests)
                tail = stage == 'prefill' and active <= 8
                extent = 0
                if stage == 'prefill' and not tail:
                    visible = max(r['prefill_offset']+r['prefill_tokens'] for r in requests)
                    extent = 128
                    while extent < visible:
                        extent *= 2
                active_owners = sum(r['prefill_tokens'] > 0 for r in requests) if stage == 'prefill' else len(requests)
                attention_expected = (stage, active, tail, extent, active_owners)
                attention_checks = set()
            stage_counts[stage] += 1
        elif kind == 'native_chunk_attention_mps_prepared':
            require(matrix_attention and e['plan_count'] == profile['expected_attention_mps_plan_count']
                    and e['scratch_bytes'] == profile['expected_attention_mps_scratch_bytes'],
                    'changed prepared attention storage')
        elif kind == 'native_chunk_attention_buckets_prepared':
            require(matrix_attention and e['bucket_count'] == profile['expected_attention_mps_bucket_count'],
                    'changed prepared attention buckets')
        elif kind in ('native_chunk_attention_mps', 'native_chunk_attention_index32', 'native_chunk_attention_bucket',
                      'native_chunk_attention_request_products'):
            require(matrix_attention and pending is not None and pending_fresh and kind not in attention_checks,
                    'unexpected/duplicate attention dispatch event')
            attention_checks.add(kind)
            stage, active, tail, extent, active_owners = attention_expected
            require(e['stage'] == stage, 'attention stage/count mismatch')
            if kind != 'native_chunk_attention_request_products':
                require(e['active_tokens'] == active, 'attention stage/count mismatch')
            expected = 32 if stage == 'prefill' else 0
            if kind == 'native_chunk_attention_request_products':
                products = (0 if tail else expected)*active_owners*8*4
                skipped = (0 if tail else expected)*(4-active_owners)*8*4
                require(active_requests and e['matrices'] == products and e['skipped_matrices'] == skipped
                        and e['active_request_rows'] == active_owners, 'incorrect physical request matrix ranges')
            elif kind == 'native_chunk_attention_mps':
                require(e['dispatches'] == (0 if tail else expected)
                        and e['plan_count'] == profile['expected_attention_mps_plan_count']
                        and e['scratch_bytes'] == profile['expected_attention_mps_scratch_bytes'],
                        'incorrect attention matrix route/storage')
            elif kind == 'native_chunk_attention_index32':
                require(e['dispatches'] == (32 if tail else 0), 'incorrect attention scalar-tail route')
            else:
                require(e['dispatches'] == expected and e['tail_dispatches'] == (32 if tail else 0)
                        and e['capacity'] == extent and e['unknown_controls'] == 0
                        and e['bucket_count'] == profile['expected_attention_mps_bucket_count'],
                        'incorrect prepared attention context/tail decision')
        elif kind == 'native_affine_profile':
            stage = e['stage']
            require(stage in ('prefill', 'decode', 'head'), 'unknown affine stage')
            if stage != 'head':
                require(pending == stage and pending_fresh, 'affine stage mismatch')
                require(not matrix_attention or len(attention_checks) == 3+int(active_requests), 'missing attention route coverage')
                pending = None
            elif pending == 'prefill' and not pending_fresh:
                pending = None # Restored final hidden state; no prefill body ran.
            require(e['affine_dispatches'] == dict(prefill=224, decode=225, head=1)[stage]
                    and e['misses'] == 0 and 0 < e['pipeline_count'] <= 512
                    and e['index_bits'] == 32, 'incomplete affine dispatch coverage')
            profile_counts[stage] += 1
        elif kind in ('native_mpp', 'native_indexed'):
            expected = 224 if kind == 'native_mpp' or e['active_tokens'] < 4*maximum_chunk else 0
            require(e['affine_dispatches'] == expected, 'incorrect prefill projection route')
            if kind == 'native_indexed':
                require(0 < e['active_tokens'] <= 4*maximum_chunk and e['physical_rows'] == 4*maximum_chunk,
                        'invalid indexed prefill extent')
                if reuse64:
                    require(reuse64_expected is None, 'missing prior cooperative64 event')
                    reuse64_expected = (e['active_tokens'], e['physical_rows'])
        elif kind == 'native_reuse64':
            require(reuse64 and reuse64_expected == (e['logical_rows'], e['physical_rows']),
                    'cooperative64 extent differs from active native rows')
            require(e['affine_dispatches'] == (profile['expected_body_reuse64_projections']
                                             if e['logical_rows'] >= 64 else 0),
                    'incorrect cooperative64 projection coverage')
            reuse64_expected = None
        elif kind == 'native_indexed_decode':
            require(e['stage'] in ('decode', 'head') and 0 < e['active_rows'] <= 4
                    and e['physical_rows'] == 4, 'invalid indexed decode extent')
            expected = (225 if e['stage'] == 'decode' else 1) if e['active_rows'] < 4 else 0
            require(e['affine_dispatches'] == expected, 'incorrect indexed decode route')
        elif kind == 'native_prefill_chunk':
            identity = e['native_request']
            require(identity in leases and leases[identity] == (e['row'], e['epoch'])
                    and chunks[identity], 'unknown prefill acknowledgement')
            require(chunks[identity].pop(0) == (e['offset'], e['tokens'], e['final']),
                    'incorrect prefill acknowledgement')
        elif kind == 'token':
            require(e['id'] not in completed, 'token after completion')
            ts = tokens[e['id']]
            cached = e['cached_tokens']
            require(type(cached) is int and 0 <= cached <= e['prompt_tokens'] and (prefix_policy or cached == 0),
                    'invalid token cache count')
            require(e['id'] not in token_cached or token_cached[e['id']] == cached, 'token cache count changed')
            token_cached[e['id']] = cached
            require(e['sampled_tokens'] == len(ts)+1,
                    'incorrect incremental token count/cache policy')
            ts.append(e['token_id'])
        elif kind == 'completed':
            key, identity = e['id'], e['native_request']
            require(key not in completed and identity in leases
                    and leases[identity] == (e['native_row'], e['native_epoch'])
                    and rows.get(e['native_row']) == identity, 'invalid completion lease')
            ids = e['generated_ids']
            require(ids == tokens[key] and ids and all(type(i) is int and 0 <= i < 32768 for i in ids)
                    and not any(i == 2 for i in ids[:-1]), 'invalid output IDs/EOS')
            require(e['finish_reason'] == ('stop' if ids[-1] == 2 else 'length')
                    and e['cached_tokens'] == cached_prefixes[identity] == token_cached[key] and e['profile_id'] == profile['profile_id']
                    and e['generation_mode'] == 'native_metal_continuous_batch'
                    and progress[identity] == len(e['prompt_ids'])+len(ids)-1,
                    'invalid completion state')
            completed[key] = e
            del rows[e['native_row']]
    require(not rows and pending is None and reuse64_expected is None and all(not c for c in chunks.values()), 'unfinished native work')
    require(counts['native_reuse64'] == (fresh_prefill_batches if reuse64 else 0),
            'missing/unexpected cooperative64 event')
    require(profile_counts['prefill'] == fresh_prefill_batches == counts['native_mpp'] == counts['native_indexed']
            and profile_counts['decode'] == stage_counts['decode']
            and counts['native_indexed_decode'] == profile_counts['decode']+profile_counts['head'],
            'missing projection event')
    require(maximum_active == required_active_rows and len(completed) == len(clients) == expected_requests, 'missing B4/request coverage')
    require(set(completed) == {c['id'] for c in clients}, 'HTTP/native identity mismatch')
    require(set(tokens) == set(completed), 'orphan native token stream')
    details = []
    for c in clients:
        e, result, body = completed[c['id']], c['result'], c['body']
        require(c['sse_done'] == 1 and c['sse_ids'] == e['generated_ids']
                and c['final_prompt_ids'] == e['prompt_ids'] == c['expected_prompt_ids']
                and c['final_response_ids'] == e['generated_ids'], 'HTTP/frozen/native token mismatch')
        require(result['status'] == 'completed' and result['text'] == c['sse_text'] == e['text']
                and result['generated_tokens'] == len(e['generated_ids'])
                and result['prompt_tokens'] == len(e['prompt_ids']) and result['cached_prompt_tokens'] == e['cached_tokens']
                and c['stream_cached_counts'] == [e['cached_tokens']]
                and result['finish_reason'] == c['sse_finish'] == e['finish_reason'],
                'HTTP result mismatch')
        require(len(e['generated_ids']) <= body['max_tokens']
                and (e['finish_reason'] == 'stop' or len(e['generated_ids']) == body['max_tokens'])
                and len(e['prompt_ids'])+body['max_tokens'] <= 4096, 'generation budget mismatch')
        control = controls.get(c['plan'])
        if control:
            expected = control['generated_ids'][:body['max_tokens']] if control_prefixes else control['generated_ids']
            require(control['status'] == 'completed' and control['prompt_ids'] == e['prompt_ids']
                    and expected == e['generated_ids']
                    and (control_prefixes or control['text'] == e['text']),
                    'frozen native coding control divergence')
        details.append(dict(plan=c['plan'], request=c['id'], native_request=e['native_request'],
                            row=e['native_row'], epoch=e['native_epoch'], output_tokens=len(e['generated_ids']),
                            exact_native_coding_control=True if control else None))
    close = events[-1]['memory']
    require(all(close[k] == 0 for k in OWNERS) and 0 <= close['metal_allocated_bytes'] <= 4*1024*1024,
            'native cleanup failed')
    return dict(requests=details, maximum_active_rows=maximum_active, event_counts=dict(counts),
                head_only_prefill_batches=stage_counts['prefill']-fresh_prefill_batches,
                close_allocation_bytes=close['metal_allocated_bytes'])


def load_client(prefix, plan, body, expected_prompt_ids, read, raw):
    """Bind a saved request/result to its raw stream; caller records file hashes."""
    require(read(prefix.with_suffix('.request.json')) == body, 'changed HTTP request')
    wire = [json.loads(line) for line in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
    require(wire and wire[-1]['data'] == '[DONE]', 'unterminated SSE')
    data = [json.loads(w['data']) for w in wire if w['data'] != '[DONE]']
    identities = {d['id'] for d in data}
    require(len(identities) == 1, 'invalid stream identity')
    finals = [d for d in data if 'response_token_ids' in d]
    require(len(finals) == 1, 'missing/duplicate final token arrays')
    return dict(plan=plan, id=next(iter(identities)), body=body,
        result=read(prefix.with_suffix('.result.json')), expected_prompt_ids=expected_prompt_ids,
        sse_ids=[d['token_id'] for d in data if 'token_id' in d],
        sse_done=sum(w['data'] == '[DONE]' for w in wire),
        sse_text=''.join(c.get('delta', {}).get('content', '') for d in data for c in d.get('choices', [])),
        sse_finish=finals[0]['choices'][0]['finish_reason'],
        stream_cached_counts=sorted({(d['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) for d in data if d.get('usage')}),
        final_prompt_ids=finals[0]['prompt_token_ids'], final_response_ids=finals[0]['response_token_ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--coding-control', type=Path, required=True)
    parser.add_argument('--control-provenance', type=Path, required=True)
    parser.add_argument('--control-provenance-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = {}

    def raw(path):
        data = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def read(path):
        return json.loads(raw(path))

    provenance = read(args.control_provenance)
    require(files[str(args.control_provenance.resolve())] == args.control_provenance_sha256,
            'changed control provenance')
    fixture = read(args.fixture)
    controls = {c['id']: read(args.coding_control/c['id']/'request-0.result.json')
                for c in fixture['cases'] if c['kind'] == 'coding'}
    for path, digest in files.items():
        if path != str(args.control_provenance.resolve()):
            require(provenance['files'].get(path) == digest, 'unfrozen control/fixture: '+path)
    root = args.admission
    require(read(root/'campaign/status.json')['status'] == 'completed_collection_not_automatic_qualification',
            'incomplete campaign')
    require(read(root/'accounting-audit.json')['issues'] == [], 'failed accounting audit')
    manifest, profile = read(root/'manifest.json'), read(root/'profile.json')
    events = [json.loads(json.loads(line)['line']) for line in raw(root/'server-1/worker-stdout.jsonl').splitlines()]
    exit_status = read(root/'server-1/worker-exit.json')
    clients = []
    for plan in manifest['plans']:
        bodies, workload = read(Path(plan['cases'])), read(Path(plan['workload']))
        require(files[str(Path(plan['cases']).resolve())] == plan['cases_sha256'] == workload['cases_sha256'],
                'changed frozen request bodies')
        require(len(bodies) == len(workload['requests']), 'invalid wave length')
        for i, (body, expected) in enumerate(zip(bodies, workload['requests'])):
            prefix = root/'campaign/cohort-1'/(plan['id']+'-measured-0')/('request-'+str(i))
            clients.append(load_client(prefix, plan['id'], body, expected['token_ids'], read, raw))
    report = validate(events, clients, controls, profile, exit_status)
    rejected = []
    corruptions = ['wrong-bound', 'stale-lease', 'wrong-slot', 'oversized-chunk', 'missing-projection',
                   'wrong-token', 'changed-prompt', 'wrong-stream', 'wrong-control', 'cache-hit', 'live-owner']
    if profile.get('chunk_attention_mps'):
        corruptions += ['changed-attention-storage', 'wrong-context', 'wrong-tail', 'missing-attention-route']
    if profile.get('require_reuse64'):
        corruptions += ['wrong-reuse-threshold', 'wrong-reuse-rows', 'wrong-reuse-count', 'missing-reuse-event']
    if profile.get('chunk_attention_active_rows'):
        corruptions += ['wrong-request-matrices', 'wrong-request-owners', 'missing-request-schedule']
    for label in corruptions:
        es, cs, ct = copy.deepcopy(events), copy.deepcopy(clients), copy.deepcopy(controls)
        if label == 'wrong-bound': es[0]['global_token_budget'] = 1024
        elif label == 'stale-lease': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['epoch'] = 0
        elif label == 'wrong-slot': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['slot'] += 1
        elif label == 'oversized-chunk': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['prefill_tokens'] = 129
        elif label == 'missing-projection': next(e for e in es if e['event'] == 'native_affine_profile')['affine_dispatches'] = 0
        elif label == 'wrong-token': next(e for e in es if e['event'] == 'completed')['generated_ids'][0] ^= 1
        elif label == 'changed-prompt': cs[0]['expected_prompt_ids'][0] ^= 1
        elif label == 'wrong-stream': cs[0]['final_response_ids'][0] ^= 1
        elif label == 'wrong-control': next(iter(ct.values()))['generated_ids'][0] ^= 1
        elif label == 'cache-hit': next(e for e in es if e['event'] == 'completed')['cached_tokens'] = 1
        elif label == 'changed-attention-storage': next(e for e in es if e['event'] == 'native_chunk_attention_mps')['scratch_bytes'] += 1
        elif label == 'wrong-context': next(e for e in es if e['event'] == 'native_chunk_attention_bucket')['capacity'] = 4096
        elif label == 'wrong-tail': next(e for e in es if e['event'] == 'native_chunk_attention_bucket')['tail_dispatches'] = 32
        elif label == 'missing-attention-route': es.remove(next(e for e in es if e['event'] == 'native_chunk_attention_mps'))
        elif label == 'wrong-reuse-threshold':
            ready = next(e for e in es if e['event'] == 'ready')
            threshold = str(profile['reuse64_min_outputs'])
            ready['backend'] = ready['backend'].replace('mpp_affine_reuse64_min_outputs='+threshold,
                                                       'mpp_affine_reuse64_min_outputs=0')
        elif label == 'wrong-reuse-rows': next(e for e in es if e['event'] == 'native_reuse64')['logical_rows'] += 1
        elif label == 'wrong-reuse-count': next(e for e in es if e['event'] == 'native_reuse64' and e['affine_dispatches'])['affine_dispatches'] = 0
        elif label == 'missing-reuse-event': es.remove(next(e for e in es if e['event'] == 'native_reuse64'))
        elif label == 'wrong-request-matrices': next(e for e in es if e['event'] == 'native_chunk_attention_request_products')['matrices'] += 1
        elif label == 'wrong-request-owners': next(e for e in es if e['event'] == 'native_chunk_attention_request_products')['active_request_rows'] += 1
        elif label == 'missing-request-schedule': es.remove(next(e for e in es if e['event'] == 'native_chunk_attention_request_products'))
        else: es[-1]['memory'][OWNERS[0]] = 1
        try:
            validate(es, cs, ct, profile, exit_status)
        except (ValueError, KeyError, IndexError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    raw(Path(__file__))
    report.update(collection_integrity_passed=True, exact_native_coding_control_cases=len(controls),
                  corruptions_rejected=rejected, files=files, scope=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('collection_integrity_passed', 'exact_native_coding_control_cases',
                                          'maximum_active_rows', 'close_allocation_bytes')}))


if __name__ == '__main__':
    main()
