"""Current-build default regressions and prior Qwen3-VL smoke preservation."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('identity', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-original-regressions-v3-summary.json'
assert not output.exists(), 'Preserve original regression proof'
sources, cases = {}, []


def read(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


identity = read(a.identity)
epoch = sources[str(a.identity)]
stem = 'compact-kv-original-regressions-v2'
assert identity.get('qwen3_vl_checkpoint_sha256'), 'Pin legacy checkpoint before the regression run'
suite = read(a.root / 'compact-kv-original-regressions-v3-suite-exit.json')
assert suite['exit_code'] == 0 and suite['execution_identity_sha256'].lower() == epoch
assert suite['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
prior_identity = read(a.root / identity['resume_prior_identity_file'])
prior_sha = identity['resume_prior_identity_sha256']
assert sources[str(a.root / identity['resume_prior_identity_file'])] == prior_sha
assert all(identity['source_sha256'][name] == sha for name, sha in prior_identity['source_sha256'].items())
assert all(identity['orchestration_sha256'][name] == sha for name, sha in prior_identity['orchestration_sha256'].items())
origin_rows = read(a.root / 'compact-kv-original-regressions-v3-case-origins.json')
origins = {row['tag']: row for row in origin_rows}
assert len(origin_rows) == len(origins) == 53
assert sum(row['reused'] for row in origin_rows) == 6


def result(tag, mode='plain'):
    data, terminal = read(a.root / (tag + '.json')), read(a.root / (tag + '-exit.json'))
    assert data['all_passed'] and terminal['exit_code'] == 0
    assert terminal['garnet_dll_sha256'].lower() == identity['garnet_binary_sha256'].lower()
    origin = origins[tag]
    assert terminal['execution_identity_sha256'].lower() == origin['execution_identity_sha256'].lower()
    assert origin['execution_identity_sha256'].lower() == (prior_sha if origin['reused'] else epoch)
    assert sources[str(a.root / (tag + '.json'))] == origin['data_sha256'].lower()
    assert sources[str(a.root / (tag + '-exit.json'))] == origin['exit_sha256'].lower()
    log_raw = (a.root / (tag + '.log')).read_bytes()
    assert hashlib.sha256(log_raw).hexdigest() == origin['log_sha256'].lower()
    if mode != 'plain':
        raw = (a.root / (tag + '.log')).read_bytes()
        sources[tag + '.log'] = hashlib.sha256(raw).hexdigest()
        pattern = (r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)'
                   if mode == 'racecheck' else r'ERROR SUMMARY:\s+0 errors')
        assert re.search(pattern, raw.decode('utf-8-sig'))
    cases.append({'tag': tag, 'mode': mode, 'all_original_checks_passed': True})
    return data


for mode in ('plain', 'memcheck', 'racecheck', 'synccheck'):
    data = result(stem + '-sdk-' + mode, mode)
    assert data['output_checks'] == 50 and len(data['invalid_binding_rejections']) == 12
    assert data['threaded_cloned_outputs'] == 24
    assert data['caller_pool_drop_partial_graph_close_io_close_and_last_output_owner']
    assert len(data['managed_graphs']) == 2
    assert all(row['contexts'] == 1 and row['graph_replays'] > 0 and row['graph_capture_failures'] == 0
               for row in data['managed_graphs'])
for mode in ('default', 'joined', 'shared', 'internal'):
    profiles = ((8, 4), (4, 4), (8, 16), (4, 16)) if mode == 'default' else ((b, 4) for b in (1, 2, 4, 8))
    for batch, chunk in profiles:
        data = result(f'{stem}-{mode}-verifier-b{batch}-c{chunk}')
        assert data['batch'] == batch and data['chunk'] == chunk and len(data['checks']) == 6
        assert all(row['states_bitwise_equal'] and row['predictions_equal'] for row in data['checks'])
        native = data['memory']['native_prefix_commit']
        assert native['ready'] and not native['compact_kv_outputs'] and not native['copy_only']
        assert data['deferred_checks'] and all(data['deferred_checks'].values())
        assert data['invalid_prefix_token_control_capacity_and_close_rollback']
    for batch, prefill in itertools.product((1, 2, 4, 8), (32, 64)):
        data = result(f'{stem}-{mode}-session-b{batch}-p{prefill}-d4')
        assert data['batch'] == batch and data['prefill_chunk'] == prefill and data['chunk'] == 4
        assert len(data['records']) == 24 and all(row['all_equal'] for row in data['records'])
        assert not data['target_memory']['compact_kv_outputs']
        native = data['memory']['target_verifier']['native_prefix_commit']
        assert native['ready'] and not native['compact_kv_outputs'] and not native['copy_only']
fullstate = result(stem + '-fullstate-equivalence-memcheck', 'memcheck')
assert {row['batch'] for row in fullstate['records']} == {1, 2, 4, 8}
assert all(row['same_all_full_finite_states_and_ids'] and len(row['snapshots']) == 9
           and row['old_weight_bytes'] == row['joined_weight_bytes'] for row in fullstate['records'])
assert len(cases) == 53
legacy = 'qwen3-vl-promptpack-dll-regression-compact-kv-v3'
current = read(a.root / (legacy + '.json'))
terminal = read(a.root / (legacy + '-exit.json'))
native = read(a.root / (legacy + '-identity.json'))
prior = read(a.root / 'qwen3-vl-promptpack-dll-regression-internal-stores.json')
prior_native = read(a.root / 'qwen3-vl-promptpack-dll-regression-internal-stores-identity.json')
assert terminal['exit_code'] == 0 and current['status']['ready']
assert native['garnet_sha256'].lower() == identity['garnet_binary_sha256'].lower()
assert native['xlang_sha256'].lower() == identity['xlang_binary_sha256'].lower()
for field in ('image_sha256', 'xmodel_prefill_sha256', 'checkpoint'):
    assert native[field] == prior_native[field]
assert current['profile'] == prior['profile'] and current['preprocess'] == prior['preprocess']
assert len(current['rows']) == len(prior['rows']) == 2
for new, old in zip(current['rows'], prior['rows']):
    assert new['status'] == old['status'] == 'ok'
    assert new['text'] == old['text']
    assert new['output_tokens'] == old['output_tokens'] == 16
    assert new['visual_tokens'] == old['visual_tokens'] == current['preprocess']['visual_token_count']
summary = {'sources_sha256': sources, 'garnet_dll_sha256': identity['garnet_binary_sha256'],
           'all_53_original_default_regressions_passed': True, 'cases': cases,
           'preserved_v2_cases': 6, 'fresh_v3_cases': 47, 'all_prior_registered_sources_inputs_runtime_unchanged': True,
           'original_sdk_50_checks_each_four_tools_passed': True,
           'original_verifier16_session32_passed': True, 'fullstate_four_batches_memcheck_passed': True,
           'qwen3_vl_two_16_token_smokes_match_saved_prior': True,
           'legacy_checkpoint_pinned_before_dispatch': True,
           'pretrained_qwen35_qualified': False, 'independent_quality_qualified': False,
           'primary_performance_eligible': False, 'process_memory_measured': False,
           'scope': 'Six verified v2 cases preserved;47 fresh cases, with the complete prior source/input/runtime snapshot unchanged. Unmodified default-off tests on the actual native build: SDK ownership/workspace '
                    '50 checks in each of4 unfiltered modes;16 verifiers,32 sessions and full finite '
                    'state/hidden/prediction equivalence across4 batches under memcheck. Legacy '
                    'Qwen3-VL smoke preserves saved image/preprocessing/profile/text/counts in2 '
                    '16-token requests using the current DLL and original model programs. '
                    'No broad heldout quality, fresh legacy engine compilation, Metal execution, '
                    'Qwen3.5 pretrained admission, real-model compact memory or latency claim.'}
output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('QWEN35_COMPACT_ORIGINAL_REGRESSIONS_AUDIT_PASS', len(cases))
