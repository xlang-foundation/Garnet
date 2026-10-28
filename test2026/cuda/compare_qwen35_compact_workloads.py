"""Full28 request controls and explicit compact owner accounting; no speed claim."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('control', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--checkpoint-config', required=True, type=Path)
p.add_argument('--reference-cohort', type=Path)
a = p.parse_args()
assert not a.output.exists(), 'Preserve compact workload evidence'
sources, differences, ownership = [], [], []


def read(path):
    raw = path.read_bytes()
    sources.append({'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw.decode('utf-8-sig'))


def cohort(path):
    rows = read(path)
    indexed = {(row['trial'], row['workload_group']): row for row in rows}
    assert len(rows) == len(indexed) == 28
    assert sum(not row['warmup'] for row in rows) == 20
    return indexed


config = read(a.checkpoint_config)['text_config']
layers, full = len(config['layer_types']), config['layer_types'].count('full_attention')
left, right = cohort(a.control), cohort(a.candidate)
prior = cohort(a.reference_cohort) if a.reference_cohort else None
assert left.keys() == right.keys() and (prior is None or prior.keys() == left.keys())
fields = ('model', 'batch', 'input_sha256', 'prompt_token_ids', 'generated_token_ids',
          'sampled_tokens_including_stop', 'finish_reasons', 'stop_token_ids', 'cache_policy',
          'speculative', 'speculative_metrics', 'warmup', 'native_build_policy', 'request_output_limit',
          'visual_tokens_per_row', 'matched_batch_inputs')
fixed_owners = ('state_bytes', 'request_buffer_bytes', 'embedding_head_shared_bytes', 'head_input_bytes',
                'head_output_bytes', 'proposal_io_bytes', 'projection_weight_pool_bytes',
                'projection_weight_pool_entries', 'joined_projection_group_count', 'joined_projection_group_bytes')
decoder_sum = ('state_bytes', 'request_buffer_bytes', 'engine_output_bytes', 'shared_activation_pool_bytes',
               'embedding_head_shared_bytes', 'head_input_bytes', 'head_output_bytes', 'proposal_io_bytes',
               'projection_weight_pool_bytes', 'trt_runtime_allocator_live_bytes', 'compact_state_commit_bytes')
graph_fields = ('contexts', 'aux_streams', 'joined_projection_inputs', 'internal_packed_stores',
                'internal_packed_root_fills', 'external_bf16_weight_bytes', 'network_output_aliases_input')


def check(condition, key, field):
    if not condition:
        differences.append({'key': key, 'field': field})


for key, old in left.items():
    new = right[key]
    for field in fields:
        check(old[field] == new[field], key, field)
        if prior is not None:
            check(old[field] == prior[key][field], key, 'prior control ' + field)
    cp, np = dict(old['profile']), dict(new['profile'])
    check(cp.pop('compactKVOutputs', False) is False and np.pop('compactKVOutputs', False) is True,
          key, 'compact flag transition')
    check(cp == np, key, 'other profile parameters')
    if prior is not None:
        rp = dict(prior[key]['profile'])
        check(rp.pop('compactKVOutputs', False) is False and cp == rp, key, 'prior profile')
    dp, dn = dict(old['decoder_profiles']), dict(new['decoder_profiles'])
    check(dp.pop('compact_kv_outputs') is False and dn.pop('compact_kv_outputs') is True and dp == dn,
          key, 'actual decoder profile transition')
    om, nm = old['owned_memory'], new['owned_memory']
    od, nd = om['decoder'], nm['decoder']
    for field in fixed_owners:
        check(od[field] == nd[field], key, 'canonical/weight/io owner ' + field)
    check(not od['compact_kv_outputs'] and nd['compact_kv_outputs'], key, 'actual compact decoder')
    check(od['compact_state_commit_bytes'] == 0 and not od['compact_state_commit_helpers'],
          key, 'ordinary helper allocation')
    helpers = nd['compact_state_commit_helpers']
    check(len(helpers) == 2 and nd['compact_state_commit_bytes'] == 2 * layers * 16,
          key, 'compact copy helper allocation')
    for helper in helpers:
        check(helper['ready'] and helper['compact_kv_outputs'] and helper['copy_only'] and
              helper['workspace_bytes'] == helper['parameter_bytes'] == 0 and
              helper['owned_bytes'] == helper['layout_bytes'] == layers * 16,
              key, 'copy helper native metadata')
    for decoder in (od, nd):
        check(decoder['owned_decoder_bytes'] == sum(decoder[field] for field in decoder_sum),
              key, 'decoder component sum without alias double count')
        check(all(g['graph_capture_failures'] == 0 for g in decoder['graphs']), key, 'capture failures')
    check(len(od['graphs']) == len(nd['graphs']) == 3, key, 'graph count')
    for og, ng in zip(od['graphs'], nd['graphs']):
        check(all(og[field] == ng[field] for field in graph_fields), key, 'unchanged graph math/policy counts')
    check(od['packed_profiles'] == nd['packed_profiles'], key, 'packed profile')
    check(nd['engine_output_bytes'] < od['engine_output_bytes'] and
          nd['shared_output_pool_bytes'] < od['shared_output_pool_bytes'], key, 'compact output allocation')
    for field in ('vision_input_buffer_bytes',):
        check(om[field] == nm[field], key, field)
    for field in ('trt_allocator_live_bytes', 'owned_device_output_bytes', 'context_workspace_required_bytes',
                  'contexts', 'cuda_graph_requested', 'graph_captures', 'graph_replays', 'graph_capture_failures'):
        check(om['vision'][field] == nm['vision'][field], key, 'vision ' + field)
    os, ns = om['speculative'], nm['speculative']
    on, nn = os['target_verifier']['native_prefix_commit'], ns['target_verifier']['native_prefix_commit']
    check(on['ready'] and nn['ready'] and not on['compact_kv_outputs'] and nn['compact_kv_outputs'] and
          not on['copy_only'] and not nn['copy_only'], key, 'native partial-prefix metadata')
    check(on['workspace_bytes'] == nn['workspace_bytes'] and on['parameter_bytes'] == nn['parameter_bytes'] and
          nn['layout_bytes'] - on['layout_bytes'] == full * 8 and
          nn['owned_bytes'] - on['owned_bytes'] == full * 8, key, 'prefix helper owner difference')
    check(os['mtp_buffers_bytes'] == ns['mtp_buffers_bytes'] and
          ns['owned_speculative_bytes'] - os['owned_speculative_bytes'] == full * 8,
          key, 'speculative owner difference')
    delta = nd['owned_decoder_bytes'] - od['owned_decoder_bytes'] + full * 8
    check(nm['owned_model_bytes'] - om['owned_model_bytes'] == delta, key, 'model owner delta')
    check(nm['owned_model_bytes'] < om['owned_model_bytes'], key, 'retained model owner nonregression')
    ownership.append({'key': key, 'old_owned_model_bytes': om['owned_model_bytes'],
                      'compact_owned_model_bytes': nm['owned_model_bytes'],
                      'output_bytes_saved': od['engine_output_bytes'] - nd['engine_output_bytes'],
                      'copy_helper_bytes': nd['compact_state_commit_bytes'],
                      'prefix_metadata_extra_bytes': full * 8})
result = {'all_passed': not differences, 'all_28_checked': True, 'checked_fields': fields,
          'fixed_decoder_owners': fixed_owners, 'ownership': ownership, 'differences': differences,
          'prior_control_cohort_checked': prior is not None, 'sources': sources,
          'primary_performance_eligible': False, 'process_memory_measured': False,
          'scope': 'Only compactKVOutputs changes. Full28 workload/stop/discrete-MTP controls, '
                   'actual compact helper metadata and complete retained-owner arithmetic. '
                   'Shared pool and joined views are counted once. Original profile control is '
                   'compared with its saved prior cohort when supplied. This does not establish '
                   'independent quality, all28 complete canonical-state hashes, latency, '
                   'process VRAM/physical residency or the2x goal.'}
a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_passed': result['all_passed'], 'differences': differences}, indent=2))
assert result['all_passed'], 'Compact outputs changed workload, protected owners or explicit allocation budgets'
