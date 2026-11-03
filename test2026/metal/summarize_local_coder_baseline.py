"""Audit all frozen baseline cells; retain paging and incomplete cohorts in output.

This produces descriptive results, not an automatic performance admission or
speedup claim. Warmups are retained on disk and excluded only by the frozen plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return dict(count=0, minimum=None, median=None, maximum=None)
    return dict(count=len(values), minimum=values[0], median=statistics.median(values),
                maximum=values[-1])


def paging(rows):
    counters = []
    for row in rows:
        counters.append({key: int(value) for key, value in
                         re.findall(r'^([^:\n]+):\s+(\d+)\.', row['vm_stat'], re.M)})
    result = {}
    for key in ('Pageins', 'Pageouts', 'Swapins', 'Swapouts', 'Compressions', 'Decompressions'):
        values = [row[key] for row in counters if key in row]
        result[key] = values[-1]-values[0] if len(values) >= 2 else None
    return result


def audit(root):
    manifest = read(root/'frozen'/'manifest.json')
    status = read(root/'status.json')
    report = dict(status=status, scope=manifest['scope'], cohorts=[], issues=[],
                  admission='Pending explicit review; this report makes no speedup claim')
    required = manifest['required_fresh_process_cohorts']
    if status['status'] != 'completed_collection_not_automatic_qualification':
        report['issues'].append('Campaign is not complete')
    for number in range(1, required+1):
        folder = root/f'cohort-{number}'
        if not (folder/'cohort.json').exists():
            report['issues'].append(f'Missing cohort {number}')
            continue
        cohort = dict(number=number, process=read(folder/'cohort.json'), cells=[])
        log_path = folder/'server.log'
        if log_path.exists():
            log = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
            cohort['cache_allocation_log_evidence'] = [line for line in log.splitlines()
                if re.search(r'KV cache sized down|reporting .* KV budget|KV cache size:', line)]
            running = [int(value) for value in re.findall(r'Running: (\d+) reqs', log)]
            waiting = [int(value) for value in re.findall(r'Waiting: (\d+) reqs', log)]
            cohort['periodic_log_max_running_requests'] = max(running, default=None)
            cohort['periodic_log_max_waiting_requests'] = max(waiting, default=None)
            cohort['scheduler_observation_scope'] = 'Periodic log samples can miss transitions; client concurrency is not proof of simultaneous decode'
            preemptions = [line for line in log.splitlines() if re.search(r'preempted|recomputing', line, re.I)]
            cohort['preemption_log_lines'] = preemptions[:200]
            cohort['preemption_log_lines_omitted'] = max(0, len(preemptions)-200)
        if cohort['process'].get('shutdown', {}).get('forced_kill') is not False:
            report['issues'].append(f'Cohort {number} lacks clean shutdown evidence')
        samples = []
        if (folder/'memory.jsonl').exists():
            # A live final partial line is not a corrupt completed measurement.
            for line in (folder/'memory.jsonl').read_text().splitlines():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    report['issues'].append(f'Cohort {number} has a partial memory sample')
        for plan in manifest['plans']:
            cell = dict(id=plan['id'], waves=[], issues=[])
            frozen = root/'frozen'/plan['id']
            bodies = read(frozen/'cases.json')
            workload = read(frozen/'workload.json')
            cases_hash = hashlib.sha256((frozen/'cases.json').read_bytes()).hexdigest()
            if cases_hash != plan['cases_sha256'] or cases_hash != workload['cases_sha256']:
                cell['issues'].append('Frozen request body hash mismatch')
            for phase, count in [('warmup', manifest['warmup_waves_per_cell']),
                                 ('measured', manifest['measured_waves_per_cell'])]:
                for wave in range(count):
                    path = folder/f"{plan['id']}-{phase}-{wave}"
                    if not (path/'summary.json').exists():
                        cell['issues'].append(f'Missing {phase} wave {wave}')
                        continue
                    summary = read(path/'summary.json')
                    results = [read(p) for p in sorted(path.glob('request-*.result.json'))]
                    if summary['status'] != 'completed' or len(results) != summary['requests']:
                        cell['issues'].append(f'Failed/incomplete {phase} wave {wave}')
                    if len(results) != len(bodies):
                        cell['issues'].append(f'Wrong concurrency in {phase} wave {wave}')
                    for index, body in enumerate(bodies):
                        result_path = path/f'request-{index}.result.json'
                        if not result_path.exists():
                            continue
                        result = read(result_path)
                        actual = read(path/f'request-{index}.request.json')
                        if actual != body:
                            cell['issues'].append(f'Changed request {index} in {phase} wave {wave}')
                        if (result['status'] != 'completed' or
                            result.get('prompt_tokens') != workload['requests'][index]['prompt_tokens'] or
                            not 0 < result.get('generated_tokens', 0) <= body['max_tokens'] or
                            result.get('cached_prompt_tokens', 0) != 0):
                            cell['issues'].append(f'Invalid request accounting for {index} in {phase} wave {wave}')
                    if phase == 'warmup':
                        continue
                    first = min((r['submitted'] for r in results), default=0)
                    last = max((r['completed'] for r in results), default=0)
                    # Bracket the request interval to avoid silently missing paging
                    # between the final request event and the next memory sample.
                    before = [r for r in samples if r['monotonic'] <= first]
                    after = [r for r in samples if r['monotonic'] >= last]
                    window = ([before[-1]] if before else []) + [
                        r for r in samples if first < r['monotonic'] < last] + ([after[0]] if after else [])
                    if not before or not after:
                        cell['issues'].append(f'Incomplete memory bracketing for measured wave {wave}')
                    cell['waves'].append(dict(wave=wave, summary=summary,
                        ttft_seconds=distribution([r.get('ttft_content_seconds') for r in results]),
                        decode_tokens_per_second=distribution([r.get('decode_tokens_per_second') for r in results]),
                        generated_tokens=[r.get('generated_tokens') for r in results],
                        finish_reasons=[r.get('finish_reason') for r in results],
                        system_paging_pages=paging(window),
                        max_process_tree_rss_kib=max((sum(p['rss_kib'] for p in r['processes']) for r in window), default=None),
                        raw_directory=str(path.resolve())))
            cell['aggregate_output_tokens_per_second'] = distribution([
                w['summary'].get('aggregate_output_tokens_per_second') for w in cell['waves']])
            report['issues'].extend(f"Cohort {number}, {plan['id']}: {issue}" for issue in cell['issues'])
            cohort['cells'].append(cell)
        report['cohorts'].append(cohort)
    report['memory_scope'] = 'System paging includes other apps; summed RSS may share pages and is not unique GPU allocation'
    report['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(issues=len(result['issues']), cohorts=len(result['cohorts']),
                          status=result['status'])))


if __name__ == '__main__':
    main()
