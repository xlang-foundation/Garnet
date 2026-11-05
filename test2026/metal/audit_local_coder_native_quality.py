"""Audit a serial native quality replay against its frozen prompts and events.

This checks transport/accounting, not model correctness or memory admission.
Run after collection and worker shutdown; sandbox evaluation is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(responses, worker_log):
    manifest = json.loads((responses/'frozen/manifest.json').read_text())
    status = json.loads((responses/'status.json').read_text())
    issues, results, groups, ready = [], [], [], []
    current, ids, seen = None, [], set()
    for number, line in enumerate(worker_log.read_text().splitlines(), 1):
        event = json.loads(line)
        try:
            event = json.loads(event['line'])
        except json.JSONDecodeError:
            continue  # Native compiler diagnostics share stdout.
        if not isinstance(event, dict):
            continue
        kind = event.get('event')
        if kind == 'ready':
            ready.append(event)
        elif kind == 'error':
            issues.append(f'Native error at line {number}')
        elif kind in ('token', 'completed'):
            key = event.get('id')
            if current is None:
                if key is None or key in seen:
                    issues.append(f'Missing/reused request ID at line {number}')
                current, ids = key, []
                seen.add(key)
            if key != current:
                issues.append(f'Interleaved native requests at line {number}')
            if kind == 'token':
                ids.append(event.get('token_id'))
                if event.get('sampled_tokens') != len(ids):
                    issues.append(f'Nonsequential native token usage at line {number}')
            else:
                if ids != event.get('generated_ids'):
                    issues.append(f'Native streamed/final token IDs differ at line {number}')
                if not ids or any(type(token) is not int or token < 0 for token in ids):
                    issues.append(f'Invalid native token IDs at line {number}')
                groups.append(event)
                current, ids = None, []
    if current is not None:
        issues.append('Unterminated native request')
    if len(ready) != 1:
        issues.append('Expected exactly one ready event')
    elif ready[0].get('active_requests') != 1 or ready[0].get('prefix_cache') is not False:
        issues.append('Ready event does not describe the uncached single-active study')
    plans = manifest['plans']
    if status.get('status') != 'collected_pending_sandbox_evaluation':
        issues.append('Collection is not complete')
    if len(groups) != len(plans):
        issues.append('Native completion count differs from frozen tasks')
    if status.get('tasks') != [dict(id=p['id'], client_exit_code=0) for p in plans]:
        issues.append('Client task status/order differs from frozen plan')
    for plan, event in zip(plans, groups):
        name = plan['id']
        frozen = responses/'frozen'/name
        workload = json.loads((frozen/'workload.json').read_text())
        cases = json.loads((frozen/'cases.json').read_text())
        row = json.loads((responses/name/'request-0.result.json').read_text())
        expected = workload['requests'][0]['token_ids']
        if digest(frozen/'cases.json') != plan['cases_sha256'] or workload['cases_sha256'] != plan['cases_sha256']:
            issues.append(f'{name}: frozen cases hash mismatch')
        if len(cases) != 1 or len(workload['requests']) != 1:
            issues.append(f'{name}: expected one frozen request')
        if event.get('prompt_ids') != expected:
            issues.append(f'{name}: native prompt IDs differ')
        generated = event.get('generated_ids', [])
        if row.get('status') != 'completed' or row.get('cached_prompt_tokens') != 0:
            issues.append(f'{name}: incomplete or cached client result')
        if row.get('prompt_tokens') != len(expected) or row.get('generated_tokens') != len(generated):
            issues.append(f'{name}: client/native usage differs')
        if not 1 <= len(generated) <= cases[0]['max_tokens']:
            issues.append(f'{name}: output budget exceeded')
        if row.get('text') != event.get('text') or row.get('finish_reason') != event.get('finish_reason'):
            issues.append(f'{name}: client/native text or finish reason differs')
        results.append(dict(id=name, prompt_tokens=len(expected), generated_tokens=len(generated),
                            generated_ids_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest()))
    return dict(status='passed' if not issues else 'failed', issues=issues, tasks=results,
                scope='Frozen prompt IDs, native stream IDs and HTTP result accounting; not coding correctness',
                manifest_sha256=digest(responses/'frozen/manifest.json'), worker_log_sha256=digest(worker_log),
                ready=ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--worker-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.responses, args.worker_log)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], issues=report['issues'])))
    if report['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
