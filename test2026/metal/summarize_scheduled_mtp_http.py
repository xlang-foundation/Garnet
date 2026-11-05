"""Summarize a fully audited finite B2 admission; not repeated performance."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize(root):
    root = Path(root)
    read = lambda path: json.loads(Path(path).read_text())
    collection = read(root/'collection.json')
    audit = read(root/'accounting-audit1.json')
    if (audit['status'] != 'passed' or audit['coding_passes'] != 8 or audit['canonical_cells'] != 6 or
        digest(root/'server/worker-stdout.jsonl') != audit['worker_log_sha256'] or
        collection['status'] != 'collected_pending_evaluation_and_audit' or
        collection['memory_sampler']['returncode'] != 0 or
        collection['final_memory_sample']['status'] != 'bracketed'):
        raise ValueError('Complete audited collection and orderly bracketed sampler required')
    samples = [json.loads(line) for line in (root/'memory.jsonl').read_text().splitlines()]
    times = [row['monotonic'] for row in samples]
    if (len(times) < 2 or any(a >= b for a, b in zip(times, times[1:])) or
        any(row['root_pid'] != collection['memory_sampler']['root_pid'] for row in samples)):
        raise ValueError('Invalid owned memory sample sequence')

    def counter(row, key):
        match = re.search(r'^'+re.escape(key)+r':\s+(\d+)\.', row['vm_stat'], re.M)
        if match is None:
            raise ValueError('Missing system counter: '+key)
        return int(match.group(1))

    def memory_window(interval):
        before = [row for row in samples if row['monotonic'] <= interval['begin']]
        after = [row for row in samples if row['monotonic'] >= interval['end']]
        if not before or not after:
            raise ValueError('Memory samples do not bracket the complete interval')
        a, b = before[-1], after[0]
        page = re.search(r'page size of (\d+) bytes', a['vm_stat'])
        if page is None or page.group(1) not in b['vm_stat'].split('bytes')[0]:
            raise ValueError('System page size changed')
        delta = {key: counter(b, key)-counter(a, key) for key in ('Swapins', 'Swapouts')}
        if min(delta.values()) < 0:
            raise ValueError('Nonmonotone system counters')
        return dict(begin_sample=a['monotonic'], end_sample=b['monotonic'],
                    swapin_bytes=delta['Swapins']*int(page.group(1)),
                    swapout_bytes=delta['Swapouts']*int(page.group(1)))

    cells = []
    for prompt in (256, 2048):
        for clients in (1, 2, 4):
            name = f'p{prompt}-c{clients}-o128'
            summary = read(root/name/'summary.json')
            results = [read(root/name/f'request-{i}.result.json') for i in range(clients)]
            if (summary['status'] != 'completed' or summary['successful_requests'] != clients or
                any(row['status'] != 'completed' for row in results)):
                raise ValueError('Incomplete canonical cell: '+name)
            cells.append(dict(id=name, ttft_seconds=statistics.median(row['ttft_content_seconds'] for row in results),
                request_decode_tps=statistics.median(row['decode_tokens_per_second'] for row in results),
                overall_output_tps=summary['aggregate_output_tokens_per_second'], wave_seconds=summary['wave_seconds'],
                system_paging=memory_window(collection['canonical_intervals'][name])))
    return dict(scope='One fresh B2 process after eight coding tasks, each canonical cell once; finite admission observations, not matched repeated performance qualification',
        cells=cells, coding_system_paging=memory_window(collection['coding_interval']),
        memory_scope='System paging includes other applications; bracketed sample intervals include at most one sample edge on each side. RSS is not unique GPU memory. No paging-free or peak-allocation claim.',
        native_totals=audit['native_totals'], output_tokens=sum(task['output_tokens'] for task in audit['tasks']),
        evidence_sha256={name:digest(root/name) for name in ('accounting-audit1.json', 'evaluation.json', 'corruption-tests1.json', 'memory.jsonl', 'collection.json')},
        summarizer_sha256=digest(__file__))


if __name__ == '__main__':
    report = summarize(sys.argv[1])
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output_tokens=report['output_tokens'], cells=report['cells'])))
