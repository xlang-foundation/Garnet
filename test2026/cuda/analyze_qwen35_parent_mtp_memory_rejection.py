"""Analyze native/counter evidence while preserving failed isolation acceptance. Original two observation gates remain failed and are reported explicitly."""
import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = root / 'qwen35-parent-mtp-memory-v1-rejected-analysis.json'
assert not output.exists()
sources = {}


def read(path, lines=False):
    raw = path.read_bytes()
    sources[path.name] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8-sig')
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


def epoch(value):
    return datetime.datetime.fromisoformat(value).timestamp()


identity_path = root / 'qwen35-parent-mtp-memory-v1-before-run-identity.json'
identity = read(identity_path)
identity_sha = sources[identity_path.name]
plain = read(root / 'qwen35-parent-mtp-v1-summary.json')
timing = read(root / 'qwen35-shared-mtp-study-v1-summary.json')
assert plain['native_plain_controls_passed'] and timing['all_completed_raw_controls_passed']
observations = read(root / 'qwen35-parent-mtp-memory-v1-processes.jsonl', True)
started = next(r for r in observations if r['event'] == 'observer_started')
stopped = next(r for r in observations if r['event'] == 'stopped')
window = (epoch(started['utc']), epoch(stopped['utc']))
errors = [r for r in observations if r['event'] in ('observer_failure', 'process_metadata_unavailable')]
observation_rejections = list(errors)
processes = [r for r in observations if r['event'] == 'process_observed']
compilers = {'cl', 'nvcc', 'ptxas', 'ninja', 'cmake', 'msbuild', 'link'}
families = compilers | {'xlang3', 'llama-server', 'compute-sanitizer', 'nsys'}
fields = ('DedicatedUsage', 'SharedUsage', 'TotalCommitted')
data, intervals, pids = [], [], set()
for size, batch in [('2b', 1), ('4b', 8)]:
    expected_parent = next(r for r in plain['profiles'] if r['model'] == size)
    expected_timing = next(r for r in timing['models'] if r['model'] == size)
    cohort_name = f'matched512-garnet-{size}-b{batch}-study-compact-kv-' + ('retry-v2-b1.json' if size == '2b' else 'b1.json')
    cohort = read(root / cohort_name)
    modes = {}
    for mode in ('d4', 'split', 'parent'):
        tag = f'memory-parent-mtp-{size}-b{batch}-{mode}-v1'
        part = lambda suffix: read(root / (tag + suffix))
        owner, terminal = part('-owner.json'), part('-exit.json')
        assert terminal['model_exit_code'] == terminal['monitor_exit_code'] == 0
        assert terminal['execution_identity_sha256'].lower() == owner['execution_identity_sha256'].lower() == identity_sha
        assert terminal['garnet_dll_sha256'].lower() == owner['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
        events, summary = part('-session.json'), part('-summary.json')
        assert events[-1]['event'] == 'complete'
        pid = owner['process_id']
        assert {e['process_id'] for e in events} == {pid} == {summary['process_id']}
        assert pid not in pids
        pids.add(pid)
        assert not summary['ambiguous_nonzero_duplicate_samples']
        assert summary['adapter']['uuid'] == 'GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81'
        assert all(summary['phases'][p]['samples'] >= 3 for p in ('steady_before', 'steady_after', 'after_model_close'))
        selected = [e for e in events if 'input_sha256' in e]
        assert len(selected) == 6
        for index, event in enumerate(selected):
            ref = cohort[index]
            assert event['source_record_index'] == index and event['profile'] == ref['profile']
            assert event['input_sha256'] == ref['input_sha256']
            assert event['prompt_token_ids'] == ref['prompt_token_ids']
            assert event['visual_tokens_per_row'] == ref['visual_tokens_per_row']
            assert event['output_signature'] == {k: ref[k] for k in event['output_signature']}
        audit = part('-workload-control.json')
        assert audit['all_six_input_hash_groups_exact'] and audit['all_six_garnet_output_stop_signatures_match_timing'] and audit['same_garnet_profile']
        for source in audit['sources']:
            assert hashlib.sha256((root / source['file']).read_bytes()).hexdigest() == source['sha256']
        steady = next(e for e in events if e['event'] == 'steady_after_begin')
        memory = steady['owned_memory']
        expected = (expected_parent['parent_backed_owned_model_bytes'] if mode == 'parent' else
                    expected_timing['per_run']['a1' if mode == 'd4' else 'b1']['owned_model_bytes_range'][0])
        assert memory['owned_model_bytes'] == expected
        assert summary['owned_bytes_after_warmup'] == [expected]
        assert all(e['mtp_strategy']['mode'] == mode for e in events)
        assert steady['native_build_policy'] == cohort[0]['native_build_policy']
        if mode != 'd4':
            spec = memory['speculative']
            assert spec['mtp_parent_storage'] == (mode == 'parent')
            if mode == 'parent':
                assert spec['mtp_owned_activation_pool_bytes'] == spec['mtp_owned_output_pool_bytes'] == 0
        interval = (epoch(owner['started']), epoch(terminal['ended_utc']))
        assert window[0] <= interval[0] < interval[1] <= window[1]
        assert any(p['pid'] == pid and interval[0] <= epoch(p['observed_utc']) <= interval[1] for p in processes)
        overlaps = []
        for process in processes:
            family = process['name'].lower().removesuffix('.exe')
            if family not in families or (family == 'xlang3' and process['pid'] == pid):
                continue
            presence = (epoch(process['started_utc']), epoch(process['observed_utc']))
            if max(interval[0], presence[0]) <= min(interval[1], presence[1]):
                overlaps.append(process)
        observation_rejections.extend({'event': 'known_external_presence_overlap', 'model': size, 'mode': mode, 'process': p} for p in overlaps)
        intervals.append({'size': size, 'mode': mode, 'process_id': pid, 'interval_unix': interval})
        packets = read(root / (tag + '-counters.jsonl'), True)
        assert packets[-1]['event'] == 'complete'
        lo = next(e['unix_seconds'] for e in events if e['event'] == 'baseline_begin') + 1
        hi = next(e['unix_seconds'] for e in events if e['event'] == 'baseline_end') - 1
        baseline = []
        for packet in packets:
            if packet['event'] != 'sample' or packet['query_begin_unix'] < lo or packet['query_end_unix'] > hi:
                continue
            assert packet['target_process_id'] == pid
            rows = packet['target_rows']
            assert all(r['process_id'] == pid and r['cuda_node_selected'] for r in rows)
            nonzero = [r for r in rows if any(int(r[f]) for f in fields)]
            assert len(nonzero) <= 1
            if rows:
                baseline.append(nonzero[0] if nonzero else rows[0])
        modes[mode] = {'process_id': pid, 'profile': steady['profile'], 'native_build_policy': steady['native_build_policy'],
            'retained_owned_model_bytes': expected, 'all_phase_sampled_statistics': summary['phases'],
            'warm_process_counter_bytes': {f: summary['phases']['steady_after']['bytes'][f]['median'] for f in fields},
            'context_baseline': {'samples': len(baseline), 'median_bytes': {f: statistics.median(int(r[f]) for r in baseline) for f in fields} if baseline else None},
            'largest_observed_bytes_not_true_peak': summary['largest_observed_bytes'],
            'missing_counter_samples': summary['target_counter_missing_samples']}
    assert modes['d4']['profile'] == modes['split']['profile'] == modes['parent']['profile']
    assert modes['d4']['native_build_policy'] == modes['split']['native_build_policy'] == modes['parent']['native_build_policy']
    assert modes['split']['retained_owned_model_bytes'] - modes['parent']['retained_owned_model_bytes'] == expected_parent['removed_owned_pool_bytes_vs_validated_split']
    deltas = {control: {f: modes[control]['warm_process_counter_bytes'][f] - modes['parent']['warm_process_counter_bytes'][f] for f in fields} for control in ('d4', 'split')}
    data.append({'model': size, 'batch': batch, 'modes': modes, 'control_minus_parent_counter_bytes': deltas,
                 'parent_committed_not_greater_than_d4': deltas['d4']['TotalCommitted'] >= 0,
                 'parent_committed_not_greater_than_split': deltas['split']['TotalCommitted'] >= 0})
assert all(a['interval_unix'][1] < b['interval_unix'][0] for a, b in zip(intervals, intervals[1:]))
result = {'all_six_memory_controls_passed': False, 'all_six_native_counter_workload_owner_controls_passed': True, 'isolation_accepted': False, 'observation_rejections': observation_rejections, 'models': data, 'intervals': intervals,
    'execution_identity_sha256': identity_sha, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
    'sources_sha256': sources, 'observer_maximum_snapshot_gap_seconds': stopped['maximum_snapshot_gap_seconds'],
    'known_named_process_overlap_count': sum(r['event'] == 'known_external_presence_overlap' for r in observation_rejections), 'latency_accepted': False, 'production_promoted': False,
    'scope': 'ISOLATION REJECTED: original controller remains terminal1. Native/counter evidence only; not eligible as isolated memory acceptance. Three fresh sequential processes per model; exact six workload groups and prior owner ledgers. '
             'WDDM committed process counters are sampled, not physical residency or true peaks. '
             'Context baseline is explicitly initialized then held; missing instances are unobserved. '
             'Sampled named-process coverage cannot exclude short-lived or unobserved activity. '
             'Polling and10-second holds invalidate primary timing; no independent quality or all-batch/capacity acceptance.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print('PARENT_MTP_MEMORY_REJECTED_DIAGNOSTIC_ONLY', len(intervals))
for model in data:
    print(json.dumps({'model': model['model'], 'control_minus_parent_counter_bytes': model['control_minus_parent_counter_bytes']}))
