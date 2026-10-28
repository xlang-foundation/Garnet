# SPDX-License-Identifier: Apache-2.0
"""Summarize the internal-root API probe; never infer real-model savings."""
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
stem = 'internal-packed-store-v3'

def read(name):
    return json.loads((root / name).read_text(encoding='utf-8-sig'))

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

identity = read(stem + '-before-execution-identity.json')
exe_sha = identity['standalone_trt_packed_store_alias_probe_sha256']
expected = {(batch, chunk, alias) for batch in (1, 2, 4, 8)
            for chunk in (4, 64) for alias in (False, True)}
diagnostics = []
reference = None
for mode in ('plain', 'memcheck', 'racecheck', 'synccheck'):
    prefix = stem + '-' + mode
    result, status = read(prefix + '.json'), read(prefix + '-exit.json')
    assert status['exit_code'] == 0, prefix
    assert status['executable_sha256'].lower() == exe_sha.lower(), prefix
    assert result['all_passed'] and result['negative_contract_checks'] == 16, prefix
    cases = result['cases']
    assert len(cases) == 16 and {(c['batch'], c['chunk'], c['alias_root']) for c in cases} == expected
    assert all(c['checks'] == 22 and c['all_bits_and_guards_passed']
               and c['internal_fill_root'] and not c['network_root_alias_confirmed'] for c in cases)
    if reference is None:
        reference = cases
    log = (root / (prefix + '.log')).read_text(encoding='utf-8-sig')
    assert 'TRT_PACKED_STORE_ALIAS_PROBE_COMPLETE' in log
    summary = [line for line in log.splitlines() if 'SUMMARY:' in line]
    if mode != 'plain':
        assert status['kernel_filter'] == 'all'
        expression = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                      if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
        assert re.search(expression, log), prefix
    diagnostics.append({'mode': mode, 'exit_code': 0,
                        'completed_cases': len(cases),
                        'complete_buffer_and_guard_checks': sum(c['checks'] for c in cases),
                        'negative_contract_checks': result['negative_contract_checks'],
                        'context_requirements_and_plan_sizes_equal_to_plain': cases == reference,
                        'context_upper_bounds_bytes': [c['context_upper_bound_bytes_not_model_savings'] for c in cases],
                        'sanitizer_summary_lines': summary, 'raw_result': prefix + '.json',
                        'raw_log': prefix + '.log'})

pairs = []
for batch in (1, 2, 4, 8):
    for chunk in (4, 64):
        a, b = [next(c for c in reference if c['batch'] == batch and
                     c['chunk'] == chunk and c['alias_root'] == alias) for alias in (False, True)]
        field = 'context_upper_bound_bytes_not_model_savings'
        assert b[field] < a[field]
        # Neither serialized plan embeds the complete multi-megabyte root.
        assert max(a['serialized_bytes'], b['serialized_bytes']) < 65536
        pairs.append({'batch': batch, 'chunk': chunk,
                      'copy_context_upper_bound_bytes': a[field],
                      'internal_alias_context_upper_bound_bytes': b[field],
                      'control_serialized_bytes': a['serialized_bytes'],
                      'candidate_serialized_bytes': b['serialized_bytes']})

assert read('packed-store-external-alias-rejection.json')['status'] == 'REJECTED_PREDECLARED_NETWORK_IO_ALIAS_GATE'
native = root / 'build-vs18/bin/garnet.dll'
assert sha(native) == identity['garnet_binary_sha256'], 'Production DLL changed during probe'
output = {'status': 'COMPONENT_PASSED_PRODUCTION_UNCHANGED',
          'method': 'Internal TensorRT Fill root and chained documented V3 aliases; no network input/output alias',
          'diagnostics': diagnostics, 'synthetic_context_pairs': pairs,
          'executable_sha256': exe_sha, 'unchanged_garnet_dll_sha256': sha(native),
          'external_root_method': 'Still rejected; not reclassified by this distinct method',
          'limitations': [
              'Two synthetic stores, not real decoder arithmetic, layer count, state or payload shapes.',
              'Copy operations check bit preservation; racecheck does not prove all possible concurrency behavior.',
              'Component workspace bounds exclude public outputs, parameters and other runtime allocations.',
              'No model memory, latency, correctness or 2x acceptance follows from these synthetic results.',
              'Production integration and original full-state, real-image and regression gates remain required.']}
destination = root / 'internal-packed-store-component-summary.json'
destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(destination)
