"""Audit every frozen MTP campaign request before descriptive speed comparison."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from audit_local_coder_mtp_http import digest, native_events
from summarize_local_coder_baseline import audit as baseline_audit


def audit(root):
    root = Path(root)
    read = lambda p: json.loads(Path(p).read_text())
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(label)
        checks += 1

    spec = read(root/'engine.json')
    for name, wanted in spec['sources'].items():
        require(digest(root/'sources'/name) == wanted, 'Frozen measured source changed: '+name)
    require(spec['max_running_requests'] == 1 and spec['native_continuous_batching'] is False and spec['prefix_caching'] is False and spec['enable_multimodal'] is False, 'Honest B1 uncached target policy')
    require(read(root/'post-run-source-verification.json')['status'] == 'passed' and read(root/'post-run-source-verification.json')['engine_spec_sha256'] == digest(root/'engine.json'), 'Post-run source verification')
    admission = Path(spec['admitted_http_archive'])
    require(digest(admission/'accounting-audit1.json') == spec['admission_audit_sha256'], 'Qualified HTTP admission')
    admitted = read(admission/'accounting-audit1.json')
    require(admitted['status'] == 'passed' and admitted['coding_passes'] == 8 and admitted['canonical_cells'] == 6, 'Complete coding/canonical admission')
    require(digest(root/'sources/libgarnet.dylib') == spec['qualified_library_sha256'], 'Qualified native library')
    base = baseline_audit(root/'campaign')
    require(not base['issues'] and len(base['cohorts']) == 3, 'Complete baseline accounting/memory bracketing: '+repr(base['issues']))
    manifest = read(root/'sources/benchmark-manifest.json')
    require(manifest['required_fresh_process_cohorts'] == 3 and manifest['warmup_waves_per_cell'] == 2 and manifest['measured_waves_per_cell'] == 5, 'Frozen repeats and fresh cohorts')
    require({p['id'] for p in manifest['plans']} == {f'p{p}-c{c}-o128' for p in (256,2048) for c in (1,2,4)}, 'All six frozen cells')
    expected = {p['id']:p for p in read(admission/'sources/canonical.json')}
    requests = []
    summaries = []
    for cohort in range(1, 4):
        server = root/spec['native_cohort_log_roots'][str(cohort)]
        profile = read(root/'sources'/spec['native_cohort_profiles'][str(cohort)])
        original = read(admission/'sources/profile.json')
        require({k:v for k,v in profile.items() if k != 'memory_inventory_path'} == {k:v for k,v in original.items() if k != 'memory_inventory_path'}, 'Identical admitted runtime profile')
        require(read(server/'worker-exit.json') == dict(returncode=0, expected_close=True), 'Clean native worker exit')
        events = native_events(server/'worker-stdout.jsonl')
        ready = [e for e in events if e.get('event') == 'ready']
        closed = [e for e in events if e.get('event') == 'closed']
        require(len(ready) == len(closed) == 1 and not any(e.get('event') == 'error' for e in events), 'One error-free prepared/closed engine per cohort')
        require(ready[0]['active_requests'] == 1 and ready[0]['capacity'] == 4096 and ready[0]['prefix_cache'] is False and ready[0]['native_continuous_batching'] is False, 'Measured engine profile')
        for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
            require(closed[0]['memory'][name] == 0, 'Closed owner '+name)
        groups = {}
        current = None
        for event in events:
            kind = event.get('event')
            if kind == 'mtp_prefill':
                require(current is None and event['id'] not in groups, 'One active native request')
                current = event['id']
                groups[current] = dict(prefill=event, tokens=[], transactions=[])
            elif kind in ('token', 'mtp_transaction', 'completed'):
                require(current is not None and event['id'] == current, 'Native event identity')
                group = groups[current]
                if kind == 'token':
                    group['tokens'].append(event['token_id'])
                    require(event['sampled_tokens'] == len(group['tokens']) and event['cached_tokens'] == 0, 'Native per-token usage')
                elif kind == 'mtp_transaction':
                    require(event['index'] == len(group['transactions']), 'Ordered native generation transactions')
                    group['transactions'].append(event)
                else:
                    require(group['tokens'] == event['generated_ids'] and event['cached_tokens'] == 0 and event['generation_mode'] == 'native_metal_mtp_single_active', 'Native complete output IDs')
                    group['completed'] = event
                    current = None
        require(current is None and len(groups) == 98 and all('completed' in g for g in groups.values()), 'All98 native cohort requests')
        matched = set()
        cohort_cells = []
        for plan in manifest['plans']:
            workload = read(root/'campaign/frozen'/plan['id']/'workload.json')
            bodies = read(root/'campaign/frozen'/plan['id']/'cases.json')
            measured = []
            for phase, repeats in [('warmup', 2), ('measured', 5)]:
                for wave in range(repeats):
                    folder = root/'campaign'/f'cohort-{cohort}'/f"{plan['id']}-{phase}-{wave}"
                    results = []
                    for row, body in enumerate(bodies):
                        path = folder/('request-'+str(row))
                        result = read(path.with_suffix('.result.json'))
                        stream = [json.loads(line) for line in path.with_suffix('.events.jsonl').read_text().splitlines()]
                        require(stream[-1]['data'] == '[DONE]' and sum(e['data'] == '[DONE]' for e in stream) == 1, 'One terminal HTTP DONE')
                        data = [json.loads(e['data']) for e in stream[:-1]]
                        require(data and not any('error' in e for e in data), 'Successful HTTP token stream')
                        key = data[-1]['id']
                        require(key in groups and key not in matched and all(e['id'] == key for e in data), 'HTTP/native request identity')
                        matched.add(key)
                        group = groups[key]
                        native = group['completed']
                        ids = [e['token_id'] for e in data if 'token_id' in e]
                        prompt = workload['requests'][row]['token_ids']
                        require(ids == native['generated_ids'] == data[-1]['response_token_ids'] == expected[plan['id']]['expected_ids'][row], 'Complete frozen output IDs')
                        require(native['prompt_ids'] == data[-1]['prompt_token_ids'] == prompt and group['prefill']['actual_input_tokens'] == len(prompt), 'Actual frozen input IDs and length')
                        require(group['prefill']['stats']['committed'] == [len(prompt)] and group['prefill']['stats']['head_calls'] == 1 and group['prefill']['stats']['teacher_calls'] == (len(prompt)+127)//128, 'Wide teacher state accounting')
                        require(result['text'] == native['text'] == ''.join(e['choices'][0]['delta'].get('content', '') for e in data), 'Lossless native/HTTP text')
                        require(result['finish_reason'] == native['finish_reason'] == data[-1]['choices'][0]['finish_reason'], 'Finish reason')
                        require(result['status'] == 'completed' and result['prompt_tokens'] == len(prompt) and result['generated_tokens'] == len(ids) and result['cached_prompt_tokens'] == 0, 'Client/native usage')
                        wire = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode()
                        require(path.with_suffix('.request.json').read_bytes() == wire and result['request_sha256'] == hashlib.sha256(wire).hexdigest(), 'Actual frozen request bytes')
                        token_times = [s['monotonic'] for s in stream[:-1] if 'token_id' in json.loads(s['data'])]
                        content_times = [s['monotonic'] for s in stream[:-1] if any(c.get('delta', {}).get('content') for c in json.loads(s['data']).get('choices', []))]
                        times = [s['monotonic'] for s in stream]
                        require(times == sorted(times) and result['submitted'] <= times[0] <= times[-1] <= result['completed'], 'Monotone client arrival interval')
                        require(bool(content_times) and len(token_times) == len(ids) and len(ids) > 1 and token_times[-1] > token_times[0], 'Incremental content/usage interval')
                        require(result['first_content'] == content_times[0] and result['last_content'] == content_times[-1] and result['first_usage_arrival'] == token_times[0] and result['last_usage_arrival'] == token_times[-1] and result['first_usage_batch_tokens'] == 1, 'Raw HTTP timing boundaries')
                        require(math.isclose(result['ttft_content_seconds'], content_times[0]-result['submitted'], rel_tol=1e-12, abs_tol=1e-12) and math.isclose(result['decode_tokens_per_second'], (len(ids)-1)/(token_times[-1]-token_times[0]), rel_tol=1e-12, abs_tol=1e-12) and math.isclose(result['end_to_end_seconds'], result['completed']-result['submitted'], rel_tol=1e-12, abs_tol=1e-12), 'Raw-stream TTFT/decode/end-to-end arithmetic')
                        for n, event in enumerate([e for e in data if 'token_id' in e], 1):
                            require(event['usage']['completion_tokens'] == n and event['usage']['prompt_tokens'] == len(prompt) and event['usage']['total_tokens'] == n+len(prompt) and event['usage']['prompt_tokens_details']['cached_tokens'] == 0, 'Monotone HTTP usage')
                        generated = ids[:1]
                        proposed = accepted = 0
                        for transaction in group['transactions']:
                            stats = transaction['stats']
                            count = len(transaction['emitted'])
                            remaining = body['max_tokens']-len(generated)
                            require(transaction['remaining_before'] == remaining and stats['consumed'] == [count] and 1 <= count <= stats['submitted_lengths'][0] <= min(4, remaining), 'Native transaction budget/consumption')
                            require(stats['proposals'][0] == generated[-1] and stats['accepted_drafts'] == count-1 and stats['proposed_drafts'] == stats['submitted_lengths'][0]-1, 'Known token/draft accounting')
                            require(stats['draft_profile_chunk'] == 1 and stats['correction_profile_chunk'] == 4, 'Native prepared profile selection')
                            generated.extend(transaction['emitted'])
                            final = len(generated) == len(ids)
                            require(transaction['eos'] == (final and native['finish_reason'] == 'stop') and stats['auxiliary_validation_calls'] == int(final), 'EOS/final correction')
                            proposed += stats['proposed_drafts']
                            accepted += stats['accepted_drafts']
                        require(generated == ids and native['speculative_metrics'] == dict(native_transactions=len(group['transactions']), proposed_drafts=proposed, accepted_drafts=accepted), 'Every emitted token reconstructed')
                        require(native['memory']['metal_allocated_bytes'] <= native['memory']['metal_recommended_working_set_bytes']-256*1024*1024, 'Native memory snapshot guard')
                        for field in ('ttft_content_seconds', 'decode_tokens_per_second', 'end_to_end_seconds'):
                            require(type(result[field]) in (int,float) and math.isfinite(result[field]) and result[field] > 0, 'Finite positive timing')
                        requests.append(dict(cohort=cohort, cell=plan['id'], phase=phase, wave=wave, row=row, http_id=key,
                            generated_tokens=len(ids), proposed_drafts=proposed, accepted_drafts=accepted))
                        results.append(result)
                    summary = read(folder/'summary.json')
                    seconds = max(r['completed'] for r in results)-min(r['submitted'] for r in results)
                    tps = sum(r['generated_tokens'] for r in results)/seconds
                    require(math.isclose(summary['aggregate_output_tokens_per_second'], tps, rel_tol=1e-12, abs_tol=1e-12), 'Whole-wave throughput including queueing')
                    if phase == 'measured':
                        measured.append(dict(overall_output_tps=tps, ttft_seconds=statistics.median(r['ttft_content_seconds'] for r in results), decode_tps=statistics.median(r['decode_tokens_per_second'] for r in results)))
            cohort_cells.append(dict(id=plan['id'], **{field:statistics.median(w[field] for w in measured) for field in ('overall_output_tps','ttft_seconds','decode_tps')}))
        require(matched == set(groups), 'No unaccounted native request')
        summaries.append(dict(number=cohort, cells=cohort_cells, worker_log_sha256=digest(server/'worker-stdout.jsonl')))
    require(len(requests) == 294 and sum(r['phase'] == 'measured' for r in requests) == 210, 'Complete measured and warmup request counts')
    cells = []
    for plan in manifest['plans']:
        values = [next(c for c in cohort['cells'] if c['id'] == plan['id']) for cohort in summaries]
        cells.append(dict(id=plan['id'], **{field:statistics.median(c[field] for c in values) for field in ('overall_output_tps','ttft_seconds','decode_tps')}))
    return dict(status='passed', checks=checks, requests=requests, cohorts=summaries, cells=cells,
        baseline_accounting=base, auditor_sha256=digest(__file__), reference_manifest_sha256=spec['reference_manifest_sha256'],
        scope='All294 native MTP requests,90 measured waves/210 measured requests; median of three cohort medians; no automatic2x admission')


if __name__ == '__main__':
    report = audit(sys.argv[1])
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks'], cells=report['cells'])))
