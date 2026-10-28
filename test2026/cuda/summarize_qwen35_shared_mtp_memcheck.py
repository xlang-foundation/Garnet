"""Inspect all raw sanitizer outputs and plain/memcheck full-state controls."""
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity_path = root / 'qwen35-shared-mtp-memcheck-v1-before-run-identity.json'
identity = read(identity_path)
epoch = digest(identity_path)
result = {'all_memcheck_controls_passed': False, 'cases': [], 'execution_identity_sha256': epoch,
          'independent_quality_proven': False, 'performance_or_process_memory_measurement': False}
keys = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row')
for model, batch in [('2b', 1), ('4b', 8)]:
    tag = f'qwen35-shared-mtp-{model}-b{batch}-memcheck-v1'
    raw_path = root / (tag + '.json')
    terminal = read(root / (tag + '-exit.json'))
    assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == epoch.lower()
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    log = (root / (tag + '.log')).read_text(errors='replace')
    assert re.findall(r'ERROR SUMMARY:\s*(\d+)\s+errors?', log) == ['0']
    assert '========= ERROR: ' not in log
    raw = read(raw_path)
    assert raw['all_passed'] and digest(Path(raw['plain_path'])) == raw['plain_sha256']
    plain = read(Path(raw['plain_path']))
    assert [s['mode'] for s in raw['stages']] == ['pool', 'split']
    matches = 0
    for stage in raw['stages']:
        assert len(stage['records']) == 8
        control = next(s for s in plain['stages'] if s['mode'] == stage['mode'])
        for row in stage['records']:
            ref = next(x for x in control['records'] if (x['style'], x['trial']) == (row['style'], row['trial']))
            assert all(row['result'][k] == ref['result'][k] for k in keys)
            assert row['state'] == ref['state'] and row['result']['speculative_metrics'] == ref['result']['speculative_metrics']
            matches += 1
    result['cases'].append({'model': model, 'batch': batch, 'error_summary': 0,
                            'complete_state_signature_work_matches': matches,
                            'raw_sha256': digest(raw_path), 'log_sha256': digest(root / (tag + '.log'))})
result['all_memcheck_controls_passed'] = True
result['scope'] = 'Unfiltered integration memcheck for both experimental pooled/split profiles. No racecheck/synccheck, all-batch, independent quality, latency or process-memory acceptance.'
output = root / 'qwen35-shared-mtp-memcheck-v1-summary.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print('SHARED_MTP_MEMCHECK_AUDIT_PASS', len(result['cases']))
