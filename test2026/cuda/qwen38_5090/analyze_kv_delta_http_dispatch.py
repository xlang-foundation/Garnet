"""CPU-only reconstruction of the closed compact pilot's measured dispatches.

Requires a replayed, hash-bound archive. Timestamp differences across processes
are observations of clocks on one Windows host, not causal GPU measurements.
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

source, destination = map(Path, sys.argv[1:3])
pilot = 'kv-delta-http-pilot-attempt1'
proof_path = source / (pilot + '-verified.json')
proof = json.loads(proof_path.read_bytes())
archive = source / (pilot + '-evidence.zip')
sha = lambda data: hashlib.sha256(data).hexdigest()
assert proof['status'] == 'compact_kv_delta_http_pilot_replayed'
assert sha(archive.read_bytes()) == proof['archive_sha256']
base = 'results/' + pilot + '/native/'
report = {}
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    dispatches = [json.loads(line) for line in z.read(base + 'worker/batch-dispatch.jsonl').splitlines()]
    worker = {}
    for line in z.read(base + 'worker/worker-stdout.jsonl').splitlines():
        wrapper = json.loads(line)
        try:
            event = json.loads(wrapper['line'])
        except ValueError:
            continue
        if isinstance(event, dict) and event.get('event') in ('token', 'completed'):
            worker.setdefault(event['id'], []).append((wrapper['received'], event))
    membership = {}
    for number, group in enumerate(dispatches):
        for request_id in group['request_ids']:
            assert request_id not in membership
            membership[request_id] = (number, group)
    for case in ('text', 'image'):
        for c in (1, 2, 4, 8):
            cohort = case + '-c' + str(c)
            paths = sorted(n for n in names if n.startswith(base + 'benchmark/') and
                           '-s3-' + cohort + '-r' in n and n.endswith('-result.json'))
            assert len(paths) == 8
            rows = []
            origin = min(json.loads(z.read(path))['submitted'] for path in paths)
            for path in paths:
                result = json.loads(z.read(path))
                assert result['status'] == 'completed'
                events_path = path.removesuffix('-result.json') + '-events.jsonl'
                sse = [json.loads(line) for line in z.read(events_path).splitlines()]
                finals = [json.loads(e['data']) for e in sse if e['data'] != '[DONE]' and
                          'response_token_ids' in json.loads(e['data'])]
                assert len(finals) == 1
                request_id = finals[0]['id']
                number, group = membership[request_id]
                tokens = [(received, event) for received, event in worker[request_id] if event['event'] == 'token']
                completed = [event for _, event in worker[request_id] if event['event'] == 'completed']
                assert len(tokens) == 128 and len(completed) == 1
                assert completed[0]['generated_ids'] == finals[0]['response_token_ids']
                rows.append({
                    'key': result['key'], 'request_id': request_id,
                    'submitted_relative_seconds': result['submitted'] - origin,
                    'dispatch_relative_seconds': group['monotonic'] - origin,
                    'first_token_relative_seconds': result['first_token'] - origin,
                    'completed_relative_seconds': result['completed'] - origin,
                    'observed_submit_to_dispatch_seconds': group['monotonic'] - result['submitted'],
                    'observed_dispatch_to_first_worker_receipt_seconds': tokens[0][0] - group['monotonic'],
                    'client_ttft_seconds': result['first_token'] - result['submitted'],
                    'group_number': number, 'real_rows': group['real_rows'],
                    'generation_mode': completed[0]['generation_mode'],
                })
            groups = sorted({row['group_number'] for row in rows})
            assert len(groups) == proof['actual_dispatch_by_cohort'][cohort]['groups']
            report[cohort] = {'requests': rows, 'groups': [dict(number=n, **dispatches[n]) for n in groups]}

output = {
    'status': 'closed_compact_http_dispatch_reconstructed',
    'archive_sha256': proof['archive_sha256'],
    'proof_sha256': sha(proof_path.read_bytes()),
    'analyzer_sha256': sha(Path(__file__).read_bytes()),
    'cohorts': report,
    'scope': 'All64measured requests retained. Submit/dispatch/receipt timestamp differences are observations across Windows processes, not a calibrated clock or GPU trace. Missing HTTP accept/enqueue timestamps prevent separating connection, handler, queue and batching delay. Occupancy differences do not establish the cause of a regression.',
    'next_experiment': 'Instrument accept, complete-body read, enqueue, grouping start and dispatch; CPU fixture and unchanged model token gates before another GPU timing. Keep client workloads unchanged; no assumption that a larger grouping window or backlog fixes the observed behavior.',
    'two_x_established': False,
}
destination.write_text(json.dumps(output, indent=2), encoding='utf8')
for cohort, values in report.items():
    print(json.dumps({'cohort': cohort, 'groups': [g['real_rows'] for g in values['groups']],
                      'submit_to_dispatch_seconds': [round(r['observed_submit_to_dispatch_seconds'], 4) for r in values['requests']]}))
