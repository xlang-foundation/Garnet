"""Audit native MTP transactions, HTTP streams and frozen reference outputs.

This finite admission audit establishes accounting and output parity, not speed.
"""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def native_events(path):
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(json.loads(line)['line'])
        except (ValueError, KeyError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def audit(root, qualified, reference):
    root, qualified, reference = Path(root), Path(qualified), Path(reference)
    read = lambda p: json.loads(Path(p).read_text())
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
    require(artifacts['qualified_source_manifest_sha256'] == q['source_manifest_sha256'], 'Qualified source identity')
    require(artifacts['qualified_library_sha256'] == q['builds']['native-mtp-profiles-build']['libgarnet.dylib'], 'Qualified library identity')
    require(artifacts['asset_manifest_sha256'] == 'db7aecc1a6bd8911e5737c0dc3f13089f048638669177f8c8dc331df0a6ad198', 'Pinned target assets')
    require(artifacts['auxiliary_sha256'] == 'dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b', 'Pinned auxiliary assets')
    control_log = reference/'server/cohort-1/worker-stdout.jsonl'
    require(digest(control_log) == artifacts['reference_worker_log_sha256'], 'Frozen canonical reference log')
    controls = {}
    for event in native_events(control_log):
        if event.get('event') == 'completed':
            key = tuple(event['prompt_ids'])
            if key in controls:
                require(controls[key]['generated_ids'] == event['generated_ids'], 'Deterministic canonical control')
            controls[key] = event
    collection = read(root/'collection.json')
    require(collection['status'] == 'collected_pending_evaluation_and_audit' and not collection['issues'], 'Completed finite collection')
    require(collection['shutdown']['graceful_native_closed'] is True and collection['shutdown']['forced_kill'] is False, 'Graceful owned shutdown')
    worker_exit = read(root/'server/worker-exit.json')
    require(worker_exit['returncode'] == 0 and worker_exit['expected_close'] is True, 'Native worker clean exit')
    profile = read(root/'sources/profile.json')
    events = native_events(root/'server/worker-stdout.jsonl')
    ready = [e for e in events if e.get('event') == 'ready']
    require(len(ready) == 1, 'One prepared native engine')
    require(ready[0]['active_requests'] == 1 and ready[0]['prefix_cache'] is False and ready[0]['multimodal'] is False and ready[0]['native_continuous_batching'] is False, 'Honest B1 uncached text profile')
    require(ready[0]['auxiliary_draft_chunk'] == 1 and ready[0]['verification_chunk'] == 4 and ready[0]['target_prefill_chunk'] == ready[0]['auxiliary_prefill_chunk'] == 128, 'Prepared profile family')
    closed = [e for e in events if e.get('event') == 'closed']
    require(len(closed) == 1, 'One closed native engine')
    for name in ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners', 'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets'):
        require(closed[0]['memory'][name] == 0, 'Closed owner '+name)
    groups = {}
    current = None
    rejected = []
    for event in events:
        kind = event.get('event')
        if kind == 'error':
            require(event.get('fatal') is False and current is None, 'Only isolated preflight rejection allowed')
            rejected.append(event)
        elif kind == 'mtp_prefill':
            require(current is None and event['id'] not in groups, 'Single-active native request identity')
            current = event['id']
            groups[current] = dict(prefill=event, tokens=[], pending=[], transactions=[])
        elif kind in ('mtp_transaction', 'token', 'completed'):
            require(current is not None and event['id'] == current, 'Native event routing')
            group = groups[current]
            if kind == 'mtp_transaction':
                require(not group['pending'] and bool(group['tokens']), 'Ordered native transaction')
                require(event['index'] == len(group['transactions']), 'Sequential native transaction index')
                group['pending'] = list(event['emitted'])
                group['transactions'].append(event)
            elif kind == 'token':
                if group['tokens']:
                    require(bool(group['pending']) and event['token_id'] == group['pending'].pop(0), 'Stream exactly native emitted IDs')
                group['tokens'].append(event['token_id'])
                require(event['sampled_tokens'] == len(group['tokens']) and event.get('cached_tokens', 0) == 0, 'Native token/usage sequence')
            else:
                require(not group['pending'] and event['generated_ids'] == group['tokens'], 'Final native/stream IDs')
                require(event['cached_tokens'] == 0 and event['generation_mode'] == 'native_metal_mtp_single_active' and event['profile_id'] == profile['profile_id'], 'Completed native profile')
                group['completed'] = event
                current = None
    require(current is None and len(groups) == 23 and all('completed' in g for g in groups.values()), 'All23 complete requests')
    require(len(rejected) == 1 and 'exceeds input/output/context profile bounds' in rejected[0]['error'], 'Context rejection retained')
    require('exceeds input/output/context profile bounds' in (root/'rejected-profile.sse').read_text(), 'HTTP rejection retained')
    coding = {p['id']:p for p in read(qualified/'coding.json')['plans']}
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
        prefill = group['prefill']
        require(prefill['actual_input_tokens'] == len(plan['prompt']) and prefill['stats']['committed'] == [len(plan['prompt'])], 'Actual prefill length, not physical chunk length')
        require(prefill['stats']['profile_chunk'] == 128 and prefill['stats']['head_calls'] == 1 and prefill['stats']['teacher_calls'] == (len(plan['prompt'])+127)//128, 'Wide teacher prefill accounting')
        generated = ids[:1]
        accepted = proposed = 0
        for index, transaction in enumerate(group['transactions']):
            stats = transaction['stats']
            count = len(transaction['emitted'])
            require(transaction['remaining_before'] == plan['body']['max_tokens']-len(generated) and 1 <= count <= min(4, transaction['remaining_before']), 'Native transaction budget')
            require(stats['consumed'] == [count] and count <= stats['submitted_lengths'][0] <= min(4, transaction['remaining_before']), 'Native target consumption')
            require(stats['proposals'][0] == generated[-1] and stats['accepted_drafts'] == count-1 and stats['proposed_drafts'] == stats['submitted_lengths'][0]-1, 'Authoritative known/draft accounting')
            require(stats['draft_profile_chunk'] == 1 and stats['correction_profile_chunk'] == 4, 'Prepared T1/T4 native selection')
            generated.extend(transaction['emitted'])
            final = len(generated) == len(ids)
            require(transaction['eos'] == (final and native['finish_reason'] == 'stop') and stats['auxiliary_validation_calls'] == int(final), 'Native EOS/final correction')
            accepted += stats['accepted_drafts']
            proposed += stats['proposed_drafts']
        require(generated == ids and native['speculative_metrics'] == dict(native_transactions=len(group['transactions']), proposed_drafts=proposed, accepted_drafts=accepted), 'Every native emitted token reconstructed')
        require(native['memory']['metal_allocated_bytes'] <= native['memory']['metal_recommended_working_set_bytes']-256*1024*1024, 'Native allocation snapshot guard')
        if plan['coding']:
            evaluated = [r for r in evaluation['results'] if r['task'] == plan['label']]
            require(len(evaluated) == 1 and evaluated[0]['response_sha256'] == digest(path.with_suffix('.result.json')), 'Evaluated exact HTTP coding response')
        rows.append(dict(label=plan['label'], http_id=key, prompt_tokens=len(plan['prompt']), output_tokens=len(ids), transactions=len(group['transactions']), accepted_drafts=accepted, proposed_drafts=proposed))
    require(seen == set(groups), 'No unaccounted native request')
    return dict(status='passed', checks=checks, tasks=rows, coding_passes=8, canonical_cells=6,
        closed=closed[0]['memory'], auditor_sha256=digest(__file__), worker_log_sha256=digest(root/'server/worker-stdout.jsonl'),
        scope='B1 uncached MTP HTTP full-token, routing, budget, long-prefill and coding admission; no speed or native continuous-batch/prefix claim')


if __name__ == '__main__':
    report = audit(sys.argv[1], sys.argv[2], sys.argv[3])
    with Path(sys.argv[4]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
