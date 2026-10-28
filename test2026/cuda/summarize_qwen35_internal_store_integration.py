# SPDX-License-Identifier: Apache-2.0
"""Summarize completed production plugin/fixture gates without model claims."""
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
stem = 'internal-store-integration-v1'

def read(name):
    return json.loads((root / name).read_text(encoding='utf-8-sig'))

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

identity = read(stem + '-before-execution-identity.json')
native_sha = identity['garnet_binary_sha256']
assert read(stem + '-resume-controller-exit.json')['exit_code'] == 0
assert read(stem + '-device-preservation.json')['device_fatbin_equal']
assert sha(root / 'build-vs18/bin/garnet.dll') == native_sha
assert sha(root / 'trt-packed-store-alias-probe.exe') == identity['standalone_trt_packed_store_alias_probe_sha256']
expected = {(b, w, n, a) for b in (1, 2, 4, 8) for w in (4, 64)
            for n in (1, 2, 6) for a in (False, True)}
diagnostics = []
for mode in ('plain', 'memcheck', 'racecheck', 'synccheck'):
    tag = stem + '-production-probe-' + mode
    data, status = read(tag + '.json'), read(tag + '-exit.json')
    assert status['exit_code'] == 0 and status['garnet_dll_sha256'].lower() == native_sha
    assert data['all_passed'] and data['negative_contract_checks'] == 16
    cases = data['cases']
    assert len(cases) == 48 and {(c['batch'], c['chunk'], c['parts_per_store'], c['alias_root']) for c in cases} == expected
    assert all(c['production_plugin'] and c['internal_fill_root'] and not c['network_root_alias_confirmed']
               and c['all_bits_and_guards_passed'] and c['checks'] == 22 for c in cases)
    log = (root / (tag + '.log')).read_text(encoding='utf-8-sig')
    assert 'TRT_PACKED_STORE_ALIAS_PROBE_COMPLETE' in log
    if mode != 'plain':
        assert status['kernel_filter'] == 'all'
        pattern = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                   if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
        assert re.search(pattern, log)
    diagnostics.append({'mode': mode, 'cases': len(cases), 'full_buffer_guard_checks': sum(c['checks'] for c in cases),
                        'negative_contract_checks': data['negative_contract_checks'],
                        'summary_lines': [s for s in log.splitlines() if 'SUMMARY:' in s],
                        'result': tag + '.json', 'log': tag + '.log'})

tag = stem + '-decoder-equivalence-resumed-plain'
fixture, status = read(tag + '.json'), read(tag + '-exit.json')
assert status['exit_code'] == 0 and status['garnet_dll_sha256'].lower() == native_sha
assert fixture['all_passed'] and fixture['comparison_kind'] == 'internal packed stores'
assert [r['batch'] for r in fixture['records']] == [1, 2, 4, 8]
original = read(stem + '-decoder-equivalence-plain.json')
assert original['current_batch'] == 4 and original['baseline_snapshots'] == original['candidate_snapshots']
assert fixture['records'][2]['snapshots'] == original['candidate_snapshots']
for case in fixture['records']:
    assert case['same_all_full_finite_states_and_ids'] and len(case['snapshots']) == 9
    assert sum(s.get('packed_sha256') is not None for s in case['snapshots']) == 8

output = {'status': 'PRODUCTION_PLUGIN_DIAGNOSTICS_AND_FIXTURE_PASSED',
          'garnet_dll_sha256': native_sha, 'device_fatbin_equal_to_prior': True,
          'production_plugin_diagnostics': diagnostics, 'fixture_records': fixture['records'],
          'fixture_result': tag + '.json',
          'failure_and_resume': 'internal-store-payload-diagnostic-correction.json',
          'limits': ['Small two-layer fixture, not Qwen3.5 pretrained validation or performance acceptance.',
                     'B4 full nine snapshots recovered from original exact comparison; its memory/native counter metadata was not persisted.',
                     'Fixture workspace/owned allocation reductions are not process memory or real-model savings.',
                     'Production plugin copy/Fill diagnostics do not prove all decoder arithmetic or arbitrary concurrency safety.',
                     'Original broader numerical, pretrained, full28, Qwen3VL, timing and isolated memory gates remain open.']}
destination = root / 'internal-packed-store-integration-summary.json'
destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(destination)
