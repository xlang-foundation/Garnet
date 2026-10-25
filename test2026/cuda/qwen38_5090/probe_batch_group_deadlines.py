"""CPU-only Windows grouping deadline experiment; never an inference benchmark."""
from pathlib import Path
import hashlib
import json
import platform
import queue
import statistics
import sys
import threading
import time

output = Path(sys.argv[1])
assert not output.exists()
wait_seconds = .010
trials = 32
rows = []
for mode in ('legacy_monotonic_break', 'precise_perf_counter_retry'):
    clock = time.monotonic if mode.startswith('legacy') else time.perf_counter
    for delay in (.001, .003, .008, .012):
        for trial in range(trials):
            jobs = queue.Queue()
            release = threading.Event()
            observed = {}
            def feed():
                release.wait()
                time.sleep(delay)
                observed['put_started'] = time.perf_counter()
                jobs.put('second')
                observed['put_finished'] = time.perf_counter()
            feeder = threading.Thread(target=feed)
            feeder.start()
            started = time.perf_counter()
            deadline = clock() + wait_seconds
            release.set()
            group = ['first']
            empty_returns = 0
            while len(group) < 2:
                remaining = deadline - clock()
                if remaining <= 0:
                    break
                try:
                    ticket = jobs.get(timeout=remaining)
                except queue.Empty:
                    empty_returns += 1
                    if mode.startswith('legacy'):
                        break
                    continue
                group.append(ticket)
            dispatched = time.perf_counter()
            feeder.join(timeout=5)
            assert not feeder.is_alive() and 'put_finished' in observed
            rows.append({'mode': mode, 'requested_second_delay_seconds': delay,
                         'trial': trial, 'rows': len(group), 'started': started,
                         'dispatched': dispatched, 'elapsed_seconds': dispatched - started,
                         'second_put_started': observed['put_started'],
                         'second_put_finished': observed['put_finished'],
                         'empty_returns': empty_returns})

summaries = []
for mode in ('legacy_monotonic_break', 'precise_perf_counter_retry'):
    for delay in (.001, .003, .008, .012):
        selected = [r for r in rows if r['mode'] == mode and r['requested_second_delay_seconds'] == delay]
        summaries.append({'mode': mode, 'requested_second_delay_seconds': delay,
                          'trials': len(selected), 'paired': sum(r['rows'] == 2 for r in selected),
                          'closed_single_before_9ms': sum(r['rows'] == 1 and r['elapsed_seconds'] < .009 for r in selected),
                          'missed_put_finished_before_true_10ms': sum(r['rows'] == 1 and r['second_put_finished'] - r['started'] < .010 for r in selected),
                          'minimum_elapsed_seconds': min(r['elapsed_seconds'] for r in selected),
                          'median_elapsed_seconds': statistics.median(r['elapsed_seconds'] for r in selected),
                          'maximum_elapsed_seconds': max(r['elapsed_seconds'] for r in selected)})
result = {'status': 'cpu_grouping_deadline_observed', 'python': sys.version,
          'platform': platform.platform(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'clocks': {name: vars(time.get_clock_info(name)) for name in ('monotonic', 'perf_counter')},
          'wait_seconds': wait_seconds, 'summaries': summaries, 'raw_trials': rows,
          'scope': 'CPU queue/clock experiment reproducing the legacy loop and proposed QPC/retry loop. Producer scheduling and queue timeout behavior retained. No HTTP/model/GPU timing, no attribution of an earlier HTTP regression and no speed claim.'}
output.write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in result.items() if k != 'raw_trials'}, indent=2))
