"""Replay frozen OpenAI streaming requests; preserve raw events and honest timing.

Runs against one already-started engine. Model/backend admission and memory
measurement are separate. SSE messages are never counted as tokens.
"""
import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def run_request(url, body, output, barrier, timeout, expected_prompt=None, allow_cached=False):
    wire = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode()
    output.with_suffix('.request.json').write_bytes(wire)
    barrier.wait()
    start = time.perf_counter()
    result = dict(status='failed', submitted=start,
                  request_sha256=hashlib.sha256(wire).hexdigest())
    count, first_count, first_usage, last_usage = 0, None, None, None
    first_content, last_content, finish, done, prompt, cached = None, None, None, False, None, 0
    content, reasoning = [], []
    try:
        req = Request(url.rstrip('/') + '/v1/chat/completions', data=wire,
                      headers={'Content-Type': 'application/json'})
        with output.with_suffix('.events.jsonl').open('x') as log:
            with urlopen(req, timeout=timeout) as response:
                event = []
                for line in response:
                    now = time.perf_counter()
                    line = line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        event.append(line[5:].lstrip(' '))
                        continue
                    if line or not event:
                        continue
                    raw, event = '\n'.join(event), []
                    log.write(json.dumps(dict(monotonic=now, data=raw)) + '\n')
                    log.flush()
                    if raw == '[DONE]':
                        done = True
                        continue
                    if done:
                        raise ValueError('Data after DONE')
                    data = json.loads(raw)
                    if data.get('error'):
                        raise ValueError(str(data['error']))
                    choices = data.get('choices', [])
                    if len(choices) > 1:
                        raise ValueError('Only n=1 is admitted')
                    for choice in choices:
                        delta = choice.get('delta', {})
                        if delta.get('content'):
                            content.append(delta['content'])
                            first_content = now if first_content is None else first_content
                            last_content = now
                        if delta.get('reasoning_content'):
                            reasoning.append(delta['reasoning_content'])
                        finish = choice.get('finish_reason') or finish
                    usage = data.get('usage')
                    if usage:
                        n, p = usage.get('completion_tokens'), usage.get('prompt_tokens')
                        if type(n) is not int or n < count or type(p) is not int or p < 1:
                            raise ValueError('Invalid/nonmonotonic usage')
                        if prompt is not None and p != prompt:
                            raise ValueError('Prompt count changed')
                        prompt = p
                        if expected_prompt is not None and p != expected_prompt:
                            raise ValueError('Server prompt count differs from frozen tokenized workload')
                        if 'max_tokens' in body and n > body['max_tokens']:
                            raise ValueError('Server exceeded output budget')
                        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
                        if cached and not allow_cached:
                            raise ValueError('Cached tokens invalidate uncached cohort')
                        if n > count:
                            if first_usage is None:
                                first_usage, first_count = now, n
                            last_usage, count = now, n
                if event:
                    raise ValueError('Unterminated SSE event')
        if not done or finish not in ('stop', 'length') or not count:
            raise ValueError('Incomplete stream or missing token usage')
        end = time.perf_counter()
        result.update(status='completed', completed=end, prompt_tokens=prompt,
                      generated_tokens=count, cached_prompt_tokens=cached,
                      finish_reason=finish, text=''.join(content), reasoning=''.join(reasoning),
                      first_content=first_content, last_content=last_content,
                      ttft_content_seconds=None if first_content is None else first_content-start,
                      first_usage_arrival=first_usage, last_usage_arrival=last_usage,
                      first_usage_batch_tokens=first_count, end_to_end_seconds=end-start,
                      decode_tokens_per_second=None)
        if first_usage is not None and last_usage > first_usage and count > first_count:
            result['decode_tokens_per_second'] = (count-first_count)/(last_usage-first_usage)
        else:
            result['decode_timing_limitation'] = 'No incremental usage interval; no decode-rate claim'
    except Exception as error:
        result.update(completed=time.perf_counter(), error=repr(error))
        if isinstance(error, HTTPError):
            result.update(http_status=error.code, http_body=error.read().decode(errors='replace'))
    output.with_suffix('.result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--cases', type=Path, required=True,
                        help='JSON array of exact OpenAI request bodies, already token-budgeted')
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--allow-cached', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    workload = json.loads(args.workload.read_text())
    cases_sha = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    if workload['cases_sha256'] != cases_sha or len(workload['requests']) != len(cases):
        raise ValueError('Frozen workload does not match request bodies')
    if not isinstance(cases, list) or not cases:
        raise ValueError('A nonempty request wave is required')
    for case in cases:
        if case.get('stream') is not True or case.get('n', 1) != 1:
            raise ValueError('Every request must stream exactly one choice')
        if not (case.get('stream_options') or {}).get('include_usage'):
            raise ValueError('Usage is required')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', cases_sha256=cases_sha,
                    metadata=json.loads(args.metadata.read_text()), url=args.url,
                    metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                    workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                    benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    allow_cached=args.allow_cached,
                    timing_scope='Client arrival timestamps; chunks are not tokens; memory measured separately')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    barrier = threading.Barrier(len(cases))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        jobs = [pool.submit(run_request, args.url, body, args.output/('request-'+str(i)),
                            barrier, args.timeout, workload['requests'][i]['prompt_tokens'],
                            args.allow_cached) for i, body in enumerate(cases)]
        rows = [job.result() for job in jobs]
    completed = [row for row in rows if row['status'] == 'completed']
    summary = dict(status='completed' if len(completed) == len(rows) else 'failed',
                   requests=len(rows), successful_requests=len(completed),
                   aggregate_output_tokens_per_second=None)
    if len(completed) == len(rows):
        wall = max(r['completed'] for r in rows)-min(r['submitted'] for r in rows)
        summary.update(wave_seconds=wall,
                       aggregate_output_tokens_per_second=sum(r['generated_tokens'] for r in rows)/wall)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    manifest['status'] = summary['status']
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summary))
    if summary['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
