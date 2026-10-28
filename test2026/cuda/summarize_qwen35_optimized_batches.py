"""Audit complete real-model batch admission; never converts diagnostics to speed."""
import hashlib
import json
import re
import sys
from pathlib import Path

root, identity_path = map(Path, sys.argv[1:3])
output = root / 'optimized-batches-v1-summary.json'
assert not output.exists()
identity_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
identity = json.loads(identity_path.read_text())
design = json.loads((root / 'optimized-batches-v1-prospective-design.json').read_text())
records = []
for profile in design['profiles']:
    for mode in ('plain', 'memcheck'):
        tag = f"optimized-batches-{profile['size']}-b{profile['batch']}-{mode}-v1"
        data_path = root / (tag + '.json')
        terminal_path = root / (tag + '-exit.json')
        log_path = root / (tag + '.log')
        data = json.loads(data_path.read_text())
        terminal = json.loads(terminal_path.read_text(encoding='utf-8-sig'))
        assert terminal['exit_code'] == 0 and terminal['execution_identity_sha256'].lower() == identity_sha
        assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
        request_path = root / profile['request']
        assert hashlib.sha256(request_path.read_bytes()).hexdigest() == profile['request_sha256']
        assert data['request'] == json.loads(request_path.read_text())
        assert data['all_passed'] and len(data['candidate']) == 8 and len(data['ordinary']) == 4
        events = [event['event'] for event in data['events']]
        assert events == ['candidate_load_begin', 'candidate_load_complete', 'candidate_closed',
                          'ordinary_load_begin', 'ordinary_load_complete', 'ordinary_closed', 'complete']
        if mode == 'memcheck':
            assert re.search(r'ERROR SUMMARY:\s+0 errors', log_path.read_text(encoding='utf-8-sig'))
        for row in data['candidate']:
            ordinary = next(r for r in data['ordinary'] if r['style'] == row['style'])
            assert row['state'] == ordinary['state']
            for key in ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
                        'finish_reasons', 'stop_token_ids', 'visual_tokens_per_row'):
                assert row['result'][key] == ordinary['result'][key]
            for state in row['state'].values():
                assert state['finite_fp32_words'] == state['shape'][0] * state['shape'][1]
        records.append({'size': profile['size'], 'batch': profile['batch'], 'mode': mode,
                        'candidate_requests': 8, 'ordinary_requests': 4, 'passed': True,
                        'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in (data_path, terminal_path, log_path)}})
result = {'records': records, 'all_six_new_profiles_plain_and_memcheck_passed': len(records) == 12,
          'garnet_dll_sha256': identity['garnet_binary_sha256'], 'execution_identity_sha256': identity_sha,
          'independent_quality_proven': False, 'full_capacity_boundary_proven': False,
          'performance_or_process_memory_measured': False,
          'scope': design['scope']}
output.write_text(json.dumps(result, indent=2))
print('OPTIMIZED_BATCH_ADMISSION_COMPLETE', len(records))
