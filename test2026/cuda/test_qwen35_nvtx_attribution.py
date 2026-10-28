"""Independent positive/negative attribution fixtures; no CUDA imports."""
import json
import sys
from pathlib import Path
from qwen35_nvtx_attribution import validate_ranges, correlated_path, phase

ranges = validate_ranges([{'start': 0, 'end': 100, 'tid': 7, 'label': 'request'},
    {'start': 10, 'end': 50, 'tid': 7, 'label': 'target_prefill'},
    {'start': 20, 'end': 40, 'tid': 7, 'label': 'mtp_prompt_cache'}])
api = {'start': 25, 'end': 30, 'tid': 7}
# GPU ends after the CPU range: actual correlation, not time containment.
event = {'start': 200, 'end': 900, 'correlation': 8}
path, error = correlated_path(event, {8: [api]}, ranges)
assert path == ['request', 'target_prefill', 'mtp_prompt_cache'] and error is None
assert phase(path) == 'mtp_prompt_cache'
assert correlated_path(event, {}, ranges)[1] == 'no_correlated_runtime_api'
assert correlated_path(event, {8: [api, api]}, ranges)[1] == 'ambiguous_correlated_runtime_api'
assert correlated_path(event, {8: [dict(api, tid=9)]}, ranges)[1] == 'correlated_api_outside_request_thread_or_range'
assert correlated_path(event, {8: [dict(api, start=99, end=110)]}, ranges)[1] == 'correlated_api_outside_request_thread_or_range'
try:
    validate_ranges([{'start': 0, 'end': 40, 'tid': 7, 'label': 'a'},
                     {'start': 20, 'end': 50, 'tid': 7, 'label': 'b'}])
    raise AssertionError('Crossing NVTX ranges admitted')
except AssertionError as e:
    assert str(e) == 'Crossing/ambiguous NVTX ranges'
out = Path(sys.argv[1]);assert not out.exists()
out.write_text(json.dumps({'all_six_attribution_controls_passed': True,
    'controls': ['late GPU completion correlated correctly', 'missing correlation unassigned', 'duplicate API ambiguous',
                 'foreign thread unassigned', 'crossing API unassigned', 'crossing range rejected'],
    'scope': 'CPU synthetic attribution protocol only; actual trace schema/ranges/correlations still require audit.'}, indent=2)+'\n')
print('QWEN35_NVTX_ATTRIBUTION_CPU_CONTROLS_PASS')
