"""Independent native B2 schedule/lease/progress and HTTP control audit.

Args: collected root, qualified native B2 root, frozen B1 HTTP control, report.
Finite admission only; this cannot qualify a repeated performance campaign.
"""
import hashlib
import json
from pathlib import Path
import sys
from audit_local_coder_mtp_http import digest, native_events


def native_accounting(events, require):
    groups, identities, owners, last_epoch = {}, {}, {}, {}
    pending = {}
    rejected = []
    tick = -1
    frames = combined = refreshes = proposed = accepted = 0
    ready, closed = [], []
    for event in events:
        kind = event.get('event')
        if kind == 'ready':
            ready.append(event)
            require(len(ready) == 1 and not groups, 'One ready native engine before admission')
            require(event['active_requests'] == 2 and event['native_continuous_batching'] is True and
                    event['prefix_cache'] is False and event['multimodal'] is False, 'Honest native B2 uncached text profile')
            require(event['target_prefill_chunk'] == 128 and event['auxiliary_prefill_chunk'] == 64 and
                    event['auxiliary_draft_chunk'] == 0 and event['verification_chunk'] == 2 and
                    event['capacity'] == 4096, 'Qualified B2 prepared profile')
        elif kind == 'mtp_admitted':
            key, identity = event['id'], event['native_id']
            require(len(ready) == 1 and key not in groups and identity not in identities and
                    type(identity) is int and identity > 0, 'Distinct admitted native and HTTP identities')
            require(0 < event['prompt_tokens'] <= 3072 and 0 < event['max_output'] <= 1024 and
                    event['prompt_tokens']+event['max_output'] <= 4096, 'Admitted native bounds')
            groups[key] = dict(admitted=event, native_id=identity, tokens=[], pending=[], transactions=[],
                               prefills=[], offset=0, first_pending=False, awaiting=None,
                               placement=None, proposed=0, accepted=0, terminal=False)
            identities[identity] = groups[key]
        elif kind == 'mtp_schedule':
            require(not pending and all(not g['pending'] and g['awaiting'] is None and not g['first_pending']
                                        for g in groups.values()), 'Prior native tick fully streamed')
            tick += 1
            require(event['index'] == tick and bool(event['plan']['batches']), 'Sequential nonempty native schedule')
            for batch in event['plan']['batches']:
                require(batch['stage'] in ('prefill', 'decode') and batch['bucket_size'] == 2 and
                        1 <= len(batch['requests']) <= 2, 'Bounded native scheduled batch')
                mask, epochs = [0, 0], [0, 0]
                for request in batch['requests']:
                    identity, row, epoch = request['id'], request['row'], request['epoch']
                    require(identity in identities and identity not in pending and type(row) is int and
                            0 <= row < 2 and not mask[row] and type(epoch) is int and epoch > 0, 'Native schedule identity and physical lease')
                    group = identities[identity]
                    require('completed' not in group and not group['terminal'], 'Completed native request cannot be rescheduled')
                    if group['placement'] is None:
                        require(batch['stage'] == 'prefill' and row not in owners and epoch > last_epoch.get(row, 0), 'Fresh row lease or safe completed-row reuse')
                        group['placement'] = (row, epoch)
                        owners[row] = identity
                        last_epoch[row] = epoch
                    require(group['placement'] == (row, epoch) and owners.get(row) == identity, 'Stable exclusive native row ownership')
                    require(request['prompt_tokens'] == group['admitted']['prompt_tokens'] and
                            request['generated_tokens'] == len(group['tokens']), 'Native plan tracks actual prompt/output progress')
                    if batch['stage'] == 'prefill':
                        require(not group['tokens'] and request['prefill_offset'] == group['offset'] and
                                request['prefill_tokens'] == min(128, group['admitted']['prompt_tokens']-group['offset']), 'Actual bounded prefill offset and size')
                    else:
                        require(group['offset'] == group['admitted']['prompt_tokens'] and bool(group['tokens']) and
                                request['token_id'] == group['tokens'][-1] and
                                request['slot'] == group['offset']+len(group['tokens'])-1, 'Native decode known token and unconsumed position')
                    pending[identity] = (batch['stage'], request)
                    mask[row], epochs[row] = 1, epoch
                require(batch['active_rows'] == mask and batch['epochs'] == epochs, 'Native active mask and epochs')
        elif kind == 'mtp_prefill_chunk':
            identity = event['native_id']
            require(identity in pending and pending[identity][0] == 'prefill', 'Scheduled native prefill execution')
            group = identities[identity]
            request = pending.pop(identity)[1]
            require(event['id'] == group['admitted']['id'] and event['tick'] == tick and
                    (event['row'], event['epoch']) == group['placement'] and
                    event['offset'] == request['prefill_offset'] and event['tokens'] == request['prefill_tokens'], 'Prefill event matches native lease/plan')
            group['offset'] += event['tokens']
            final = group['offset'] == group['admitted']['prompt_tokens']
            require(event['final'] is final, 'Prefill completion uses actual length')
            group['first_pending'] = final
            group['prefills'].append(event)
        elif kind == 'mtp_batch_transaction':
            require(event['index'] == frames and event['tick'] == tick, 'Sequential native generation frames')
            result, aux = event['result'], event['auxiliary']
            items, stats = result['requests'], result['generation']
            decode = {identity for identity, (stage, _) in pending.items() if stage == 'decode'}
            require(1 <= len(items) <= 2 and {item['id'] for item in items} == decode and
                    len({item['row'] for item in items}) == len(items), 'Exactly the scheduled native decode requests')
            require(stats['recursive_draft_calls'] == 0 and stats['draft_profile_chunk'] == 2 and
                    stats['correction_profile_chunk'] == aux['profile_chunk'] == 2 and
                    aux['kv_inplace_updates'] == 2 and event['hidden_pool_bindings'] == 64 and
                    event['linear_state_inplace_updates'] == 96, 'Qualified native generation and workspace paths')
            require(len(stats['proposals']) == 4 and len(stats['submitted_lengths']) == len(stats['consumed']) == 2, 'B2 native statistics dimensions')
            rows = {item['row'] for item in items}
            for row in set(range(2))-rows:
                require(stats['submitted_lengths'][row] == stats['consumed'][row] == 0, 'Inactive native row consumes nothing')
            frame_proposed = frame_accepted = 0
            for item in items:
                identity, row = item['id'], item['row']
                group = identities[identity]
                require(pending.pop(identity)[0] == 'decode' and group['awaiting'] is None and not group['pending'], 'One native decode result per scheduled request')
                require((row, item['epoch']) == group['placement'] and
                        item['known'] == group['tokens'][-1] and stats['proposals'][row*2] == item['known'], 'Authoritative native known token and lease')
                remaining = group['admitted']['max_output']-len(group['tokens'])
                count = len(item['emitted'])
                require(item['remaining_before'] == remaining and item['consumed'] == stats['consumed'][row] == count and
                        1 <= count <= min(2, remaining) and stats['submitted_lengths'][row] == min(2, remaining), 'Native request budget and consumed block')
                require(type(item['eos']) is bool and type(item['finished']) is bool and
                        item['finished'] is (item['eos'] or count == remaining), 'Native EOS or exact budget completion')
                group['awaiting'] = (frames, item)
                group['terminal'] = item['finished']
                group['proposed'] += stats['submitted_lengths'][row]-1
                group['accepted'] += count-1
                frame_proposed += stats['submitted_lengths'][row]-1
                frame_accepted += count-1
            require(stats['proposed_drafts'] == frame_proposed and stats['accepted_drafts'] == frame_accepted, 'Whole-frame draft acceptance accounting')
            proposed += frame_proposed
            accepted += frame_accepted
            frames += 1
            combined += int(len(items) == 2)
            refreshes += int(stats['target_prediction_refreshed'])
        elif kind == 'mtp_transaction':
            require(event['id'] in groups, 'Known HTTP transaction identity')
            group = groups[event['id']]
            require(event['native_id'] == group['native_id'] and
                    group['awaiting'] == (event['frame'], event['transaction']) and not group['pending'], 'Per-request transaction equals native batch result')
            group['awaiting'] = None
            group['pending'] = list(event['transaction']['emitted'])
            group['transactions'].append(event)
        elif kind == 'token':
            require(event['id'] in groups, 'Known HTTP token identity')
            group = groups[event['id']]
            require(event['native_id'] == group['native_id'] and 'completed' not in group and group['awaiting'] is None, 'Native stream routing')
            if not group['tokens']:
                require(group['first_pending'] and group['offset'] == group['admitted']['prompt_tokens'], 'First token follows full native prefill')
                group['first_pending'] = False
            else:
                require(bool(group['pending']) and event['token_id'] == group['pending'].pop(0), 'Stream exactly native emitted IDs')
            group['tokens'].append(event['token_id'])
            require(event['sampled_tokens'] == len(group['tokens']) <= group['admitted']['max_output'] and
                    event['prompt_tokens'] == group['admitted']['prompt_tokens'] and event['cached_tokens'] == 0, 'Native token/usage sequence')
        elif kind == 'completed':
            require(event['id'] in groups, 'Known native completion')
            group = groups[event['id']]
            require(event['native_id'] == group['native_id'] and 'completed' not in group and not group['pending'] and
                    group['awaiting'] is None and not group['first_pending'] and
                    event['generated_ids'] == group['tokens'] and bool(group['tokens']), 'Complete native request and streamed IDs')
            require(len(event['prompt_ids']) == group['offset'] == group['admitted']['prompt_tokens'] and
                    event['speculative_metrics'] == dict(native_transactions=len(group['transactions']),
                        proposed_drafts=group['proposed'], accepted_drafts=group['accepted']), 'Actual prefill and every native transaction accounted')
            require(not group['transactions'] or group['terminal'] is True, 'Native terminal progress')
            group['completed'] = event
            row, _ = group['placement']
            require(owners.pop(row) == group['native_id'], 'Release completed native row')
        elif kind == 'error':
            require(event.get('fatal') is False and event['id'] not in groups and not owners and not pending, 'Isolated nonfatal preflight rejection')
            rejected.append(event)
        elif kind == 'closed':
            require(not owners and not pending and all('completed' in g for g in groups.values()), 'Native shutdown drains every request')
            closed.append(event)
        elif kind is not None:
            raise ValueError('Unknown native serving event: '+str(kind))
    require(len(ready) == len(closed) == 1 and len(groups) == 23 and all('completed' in g for g in groups.values()), 'All23 complete requests and clean native lifecycle')
    require(combined > 0 and refreshes > 0, 'Actual native continuous batching and interleaved refresh exercised')
    require(len(rejected) == 1 and 'exceeds input/output/context profile bounds' in rejected[0]['error'], 'Profile rejection retained')
    for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
        require(closed[0]['memory'][name] == 0, 'Closed owner '+name)
    return groups, closed[0]['memory'], dict(native_frames=frames, combined_frames=combined,
        prediction_refreshes=refreshes, proposed_drafts=proposed, accepted_drafts=accepted)


def audit(root, qualified, reference):
    root, qualified, reference = map(Path, (root, qualified, reference))
    read = lambda path: json.loads(Path(path).read_text())
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(label)
        checks += 1

    artifacts = read(root/'artifacts.json')
    for name, wanted in artifacts['files'].items():
        require(digest(root/'sources'/name) == wanted, 'Frozen implementation/profile/workload changed: '+name)
    q = read(qualified/'qualification-provenance.json')
    require(artifacts['qualified_source_manifest_sha256'] == q['source_manifest_sha256'] and
            artifacts['qualified_library_sha256'] == q['native_library_sha256'] and
            artifacts['qualified_native_audit_sha256'] == digest(qualified/'accounting-audit1.json'), 'Qualified native source/library/evidence identities')
    require(artifacts['asset_manifest_sha256'] == 'db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198' and
            artifacts['auxiliary_sha256'] == 'dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b', 'Pinned target and auxiliary assets')
    control_log = reference/'server/worker-stdout.jsonl'
    require(digest(control_log) == artifacts['reference_worker_log_sha256'] and
            digest(reference/'accounting-audit1.json') == artifacts['reference_admission_audit_sha256'], 'Frozen HTTP reference evidence')
    controls = {}
    for event in native_events(control_log):
        if event.get('event') == 'completed':
            key = tuple(event['prompt_ids'])
            if key in controls:
                require(controls[key]['generated_ids'] == event['generated_ids'], 'Deterministic full canonical reference')
            controls[key] = event
    collection = read(root/'collection.json')
    require(collection['status'] == 'collected_pending_evaluation_and_audit' and not collection['issues'], 'Complete finite HTTP collection')
    require(collection['shutdown']['graceful_native_closed'] is True and collection['shutdown']['forced_kill'] is False, 'Graceful owned native shutdown')
    worker_exit = read(root/'server/worker-exit.json')
    require(worker_exit['returncode'] == 0 and worker_exit['expected_close'] is True, 'Native worker clean exit')
    groups, closed, native_totals = native_accounting(native_events(root/'server/worker-stdout.jsonl'), require)
    require('exceeds input/output/context profile bounds' in (root/'rejected-profile.sse').read_text(), 'HTTP rejection retained')
    coding = {}
    for plan in read(qualified/'tasks.json')['plans']:
        if plan['coding']:
            control = controls[tuple(plan['prompt'])]
            require(plan['expected'] == control['generated_ids'], 'Native qualified coding reference agrees with frozen HTTP control')
            coding[plan['id']] = dict(prompt_ids=plan['prompt'], reference_ids=plan['expected'], reference_text=control['text'])
    # The HTTP checks below independently bind every saved SSE event and wire
    # request to the full frozen prompt/output/text, after native replay above.
    quality_manifest = read(root/'responses/frozen/manifest.json')
    require(len(quality_manifest['plans']) == len(coding) == 8, 'Eight frozen coding tasks')
    evaluation = read(root/'evaluation.json')
    require(evaluation['passed'] == evaluation['total'] == 8 and all(r['status'] == 'passed' for r in evaluation['results']), 'Eight sandbox passes')
    require(evaluation['suite_sha256'] == '675ea667c357f361ce39e3863b61193dc6b32f4a229eb7d29a17803b75408786', 'Frozen coding suite')
    planned = []
    for plan in quality_manifest['plans']:
        folder = root/'responses/frozen'/plan['id']
        workload, bodies = read(folder/'workload.json'), read(folder/'cases.json')
        require(digest(folder/'cases.json') == plan['cases_sha256'] == workload['cases_sha256'], 'Frozen coding request bytes')
        require(len(bodies) == len(workload['requests']) == 1, 'One request per coding task')
        expected = coding[plan['id']]
        require(expected['prompt_ids'] == workload['requests'][0]['token_ids'], 'Frozen coding input IDs')
        planned.append(dict(label=plan['id'], path=root/'responses'/plan['id']/'request-0', body=bodies[0],
            prompt=expected['prompt_ids'], ids=expected['reference_ids'], text=expected['reference_text'], coding=True))
    canonical = read(root/'sources/canonical.json')
    require({p['id'] for p in canonical} == {f'p{p}-c{c}-o128' for p in (256,2048) for c in (1,2,4)}, 'Six canonical cells')
    for plan in canonical + [dict(canonical[0], id='after-rejection')]:
        folder = Path(plan['cases']).parent
        bodies, workload = read(folder/'cases.json'), read(folder/'workload.json')
        require(digest(folder/'cases.json') == workload['cases_sha256'], 'Frozen canonical request bytes')
        require(len(bodies) == len(workload['requests']) == len(plan['expected_ids']), 'Canonical request count')
        for row, body in enumerate(bodies):
            prompt = workload['requests'][row]['token_ids']
            control = controls[tuple(prompt)]
            require(plan['expected_ids'][row] == control['generated_ids'], 'Canonical frozen output IDs')
            planned.append(dict(label=plan['id']+'-r'+str(row), path=root/plan['id']/('request-'+str(row)), body=body,
                prompt=prompt, ids=control['generated_ids'], text=control['text'], coding=False))
    seen = set()
    rows = []
    for plan in planned:
        path = plan['path']
        response = read(path.with_suffix('.result.json'))
        stream = [json.loads(line) for line in path.with_suffix('.events.jsonl').read_text().splitlines()]
        require(bool(stream) and stream[-1]['data'] == '[DONE]' and sum(s['data'] == '[DONE]' for s in stream) == 1, 'One terminal HTTP DONE')
        data = [json.loads(s['data']) for s in stream[:-1]]
        require(all('error' not in e for e in data), 'No HTTP generation errors')
        key = data[-1]['id']
        require(key in groups and key not in seen and all(e['id'] == key for e in data), 'HTTP request identity')
        seen.add(key)
        group = groups[key]
        native = group['completed']
        ids = [e['token_id'] for e in data if 'token_id' in e]
        require(ids == plan['ids'] == native['generated_ids'] == data[-1]['response_token_ids'], 'Full frozen HTTP/native output IDs')
        require(plan['prompt'] == native['prompt_ids'] == data[-1]['prompt_token_ids'], 'Full frozen HTTP/native prompt IDs')
        require(''.join(e['choices'][0]['delta'].get('content', '') for e in data) == plan['text'] == native['text'] == response['text'], 'Lossless frozen HTTP text')
        require(response['status'] == 'completed' and response['cached_prompt_tokens'] == 0 and response['prompt_tokens'] == len(plan['prompt']) and response['generated_tokens'] == len(ids), 'HTTP client result accounting')
        wire = json.dumps(plan['body'], separators=(',', ':'), ensure_ascii=False).encode()
        require(response['request_sha256'] == hashlib.sha256(wire).hexdigest() and path.with_suffix('.request.json').read_bytes() == wire, 'Actual submitted request bytes')
        require(response['finish_reason'] == native['finish_reason'] == data[-1]['choices'][0]['finish_reason'], 'HTTP/native finish reason')
        require(1 <= len(ids) <= plan['body']['max_tokens'], 'Complete-code/output budget')
        for count, event in enumerate([e for e in data if 'token_id' in e], 1):
            usage = event['usage']
            require(usage['completion_tokens'] == count and usage['prompt_tokens'] == len(plan['prompt']) and usage['total_tokens'] == count+len(plan['prompt']) and usage['prompt_tokens_details']['cached_tokens'] == 0, 'Monotone per-token HTTP usage')
        require(native['generation_mode'] == 'native_metal_mtp_continuous_batch2' and
                native['cached_tokens'] == 0, 'Native B2 serving identity and cache policy')
        require(group['admitted']['max_output'] == plan['body']['max_tokens'], 'Native budget equals submitted HTTP budget')
        memory = native['memory']
        require(memory['metal_allocated_bytes'] <= memory['metal_recommended_working_set_bytes']-256*1024*1024, 'Native allocation snapshot guard')
        if plan['coding']:
            evaluated = [r for r in evaluation['results'] if r['task'] == plan['label']]
            require(len(evaluated) == 1 and evaluated[0]['response_sha256'] == digest(path.with_suffix('.result.json')), 'Evaluated exact HTTP coding response')
        rows.append(dict(label=plan['label'], http_id=key, prompt_tokens=len(plan['prompt']),
                         output_tokens=len(ids), transactions=len(group['transactions']),
                         accepted_drafts=group['accepted'], proposed_drafts=group['proposed']))
    require(seen == set(groups), 'No unaccounted native request')
    inventory = read(root/'memory-inventory.json')
    pools = [r for r in inventory['references'] if r['role'] == 'prefill_hidden_pingpong']
    require(len(pools) == len({r['buffer_id'] for r in pools}) == 4 and
            sorted(tuple(r['shape']) for r in pools) == [(2,10240)]*2+[(2,655360)]*2, 'Exactly two hidden buffers per compiled target profile')
    weights = {}
    for ref in inventory['references']:
        if ref['tag'].startswith('auxiliary_') and ref['role'] == 'weight_record':
            weights.setdefault(ref['tag'], {}).setdefault(ref['name'], []).append(ref['buffer_id'])
    require(set(weights) == {'auxiliary_0','auxiliary_1'} and len(weights['auxiliary_0']) == 15 and
            weights['auxiliary_0'] == weights['auxiliary_1'], 'One raw auxiliary weight set across corrector/teacher')
    return dict(status='passed', checks=checks, tasks=rows, coding_passes=8, canonical_cells=6,
                native_totals=native_totals, closed=closed, auditor_sha256=digest(__file__),
                audit_helper_sha256=digest(Path(__file__).with_name('audit_local_coder_mtp_http.py')),
                worker_log_sha256=digest(root/'server/worker-stdout.jsonl'),
                scope='Native B2 uncached MTP HTTP full IDs, native schedules/leases, streaming and coding admission; no repeated speed or prefix claim')


if __name__ == '__main__':
    report = audit(*sys.argv[1:4])
    with Path(sys.argv[4]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
