"""All28 real-request workload and native-owner audit for vision replay."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('control', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--reference-cohort', type=Path, help='Prior unchanged-profile control; records default-path preservation')
a = p.parse_args()
assert not a.output.exists(), 'Preserve vision workload evidence'
load = lambda path: {(row['trial'], row['workload_group']): row
                     for row in json.loads(path.read_text(encoding='utf-8'))}
left, right = load(a.control), load(a.candidate)
reference = load(a.reference_cohort) if a.reference_cohort else None
assert len(left) == len(right) == 28 and left.keys() == right.keys()
assert reference is None or reference.keys() == left.keys()
differences = []
fields = ('input_sha256', 'prompt_token_ids', 'generated_token_ids', 'sampled_tokens_including_stop',
          'finish_reasons', 'stop_token_ids', 'cache_policy', 'speculative', 'speculative_metrics', 'warmup')
owners = ('state_bytes', 'request_buffer_bytes', 'engine_output_bytes', 'shared_output_pool_bytes',
          'shared_activation_pool_bytes', 'embedding_head_shared_bytes', 'head_input_bytes',
          'head_output_bytes', 'proposal_io_bytes', 'projection_weight_pool_bytes',
          'projection_weight_pool_entries', 'joined_projection_group_count', 'joined_projection_group_bytes',
          'owned_decoder_bytes')
for key, control in left.items():
    candidate = right[key]
    if reference is not None:
        prior = reference[key]
        for field in fields:
            if control[field] != prior[field]:
                differences.append({'key': key, 'field': 'prior control ' + field})
        current_profile, prior_profile = dict(control['profile']), dict(prior['profile'])
        if current_profile.pop('visionCudaGraph', False) or prior_profile.pop('visionCudaGraph', False) or current_profile != prior_profile:
            differences.append({'key': key, 'field': 'prior control profile'})
    for field in fields:
        if control[field] != candidate[field]:
            differences.append({'key': key, 'field': field})
    cp, rp = dict(control['profile']), dict(candidate['profile'])
    if cp.pop('visionCudaGraph', False) is not False or rp.pop('visionCudaGraph', False) is not True:
        differences.append({'key': key, 'field': 'vision flag transition'})
    if cp != rp:
        differences.append({'key': key, 'field': 'other profile parameters'})
    old, new = control['owned_memory'], candidate['owned_memory']
    for field in owners:
        if old['decoder'][field] != new['decoder'][field]:
            differences.append({'key': key, 'field': 'decoder owner ' + field})
    if old['speculative']['owned_speculative_bytes'] != new['speculative']['owned_speculative_bytes']:
        differences.append({'key': key, 'field': 'speculative owner'})
    vision = new['vision']
    valid = (vision['cuda_graph_requested'] and vision['serial_graph_reuse_requested']
             and vision['contexts'] == 1 and vision['graph_capture_failures'] == 0
             and new['vision_input_buffer_bytes'] == 1474560
             and old['vision_input_buffer_bytes'] == 0
             and not old['vision']['cuda_graph_requested'] and old['vision']['graph_captures'] == 0)
    if not candidate['warmup']:
        valid = valid and vision['graph_captures'] == 1 and vision['graph_replays'] > 0
    if not valid:
        differences.append({'key': key, 'field': 'native vision staging/capture/context counters'})
result = {
    'all_passed': not differences, 'all_28_checked': True, 'checked_fields': fields,
    'checked_decoder_owners': owners, 'differences': differences,
    'prior_control_cohort_checked': reference is not None,
    'sources': [{'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (a.control, a.candidate, *([a.reference_cohort] if a.reference_cohort else []))],
    'scope': 'Only visionCudaGraph changes. All28 inputs/prompts/outputs/stops/cache/discrete-MTP controls, '
             'unchanged decoder/speculative owners, explicit staging bytes and actual one-context/one-capture '
             'warm replay are checked. Not independent vision quality, complete canonical-state hashes for '
             'every request, primary latency acceptance, process VRAM or physical residency.',
}
a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_passed': result['all_passed'], 'differences': differences}, indent=2))
assert result['all_passed'], 'Vision replay changed workload/owners or failed to replay'
