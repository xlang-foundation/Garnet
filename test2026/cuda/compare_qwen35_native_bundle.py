"""Strict completed real-state controls before any native-bundle timing."""
import hashlib
import json
from pathlib import Path
import sys

control_path, candidate_path, historical_path, output = map(Path, sys.argv[1:])
assert not output.exists()
read = lambda p: json.loads(p.read_text())
control, candidate, historical = map(read, (control_path, candidate_path, historical_path))
assert control['all_controls_passed'] and candidate['all_controls_passed']
assert control['mode'] == 'control' and candidate['mode'] == 'bundle'
assert control['request'] == candidate['request'] == historical['request']
old_rows = next(s['records'] for s in historical['stages'] if s['mode'] == 'baseline')
assert len(control['records']) == len(candidate['records']) == len(old_rows) == 8
fields = ('prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop', 'finish_reasons',
          'stop_token_ids', 'visual_tokens_per_row', 'speculative_metrics', 'profile', 'cache_policy', 'arithmetic')
for a, b, old in zip(control['records'], candidate['records'], old_rows):
    assert (a['trial'], a['style']) == (b['trial'], b['style']) == (old['trial'], old['style'])
    for key in fields:
        assert a['result'][key] == b['result'][key] == old['result'][key], (a['trial'], a['style'], key)
    assert a['state'] == b['state'], 'Complete target/MTP state changed'
    assert all(a['state'][key] == value for key, value in old['state'].items()), 'New DLL changed saved native state'
    assert a['mtp_slots'] == b['mtp_slots'] and a['target_slots'] == b['target_slots']
    assert a['owned_memory']['owned_model_bytes'] == b['owned_memory']['owned_model_bytes']
    assert a['owned_memory']['owned_model_bytes'] > 0
result = {'all_controls_passed': True, 'all8_complete_target_mtp_state_and_work_exact': True,
          'saved5ead_real_baseline_state_outputs_work_preserved': True, 'owned_tensor_and_trt_bytes_equal': True,
          'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (control_path, candidate_path, historical_path)},
          'scope': 'Exact state/output/work and owned-byte preservation only. CUDA driver graph overhead is absent from owned ledger; no process-memory, independent-quality or speed acceptance.'}
with output.open('x') as stream:
    json.dump(result, stream, indent=2)
print('NATIVE_BUNDLE_COMPLETE_REAL_STATE_COMPARISON_PASS')
