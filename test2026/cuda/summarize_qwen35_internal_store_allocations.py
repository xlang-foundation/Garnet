# SPDX-License-Identifier: Apache-2.0
"""Compare fresh post-load owned counters; these are not process-memory peaks."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
stem = 'internal-store-real-allocation-v1'

def read(name):
    return json.loads((root / name).read_text(encoding='utf-8-sig'))

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

identity = read(stem + '-before-execution-identity.json')
native_sha = identity['garnet_binary_sha256']
records = []
sources = []
for size, batch in (('2b', 1), ('4b', 8)):
    modes = []
    for mode in ('control', 'candidate'):
        name = f'{stem}-{size}-b{batch}-{mode}'
        value, status = read(name + '.json'), read(name + '-exit.json')
        assert status['exit_code'] == 0 and status['garnet_dll_sha256'].lower() == native_sha
        assert value['all_native_profile_assertions_passed']
        assert value['batch'] == batch and value['candidate'] == (mode == 'candidate')
        assert value['model'] == ('Qwen3.5-2B' if size == '2b' else 'Qwen3.5-4B')
        modes.append(value)
        sources.append({'file': name + '.json', 'sha256': digest(root / (name + '.json'))})
    control, candidate = modes
    pa, pb = dict(control['profile']), dict(candidate['profile'])
    assert pa.pop('internalPackedStores') is False and pb.pop('internalPackedStores') is True
    assert pa == pb and control['source_profile'] == candidate['source_profile']
    assert control['vision_initialized'] == candidate['vision_initialized']
    assert control['speculative_initialized'] and candidate['speculative_initialized']
    a, b = control['memory']['decoder'], candidate['memory']['decoder']
    stable = ('state_bytes', 'request_buffer_bytes', 'engine_output_bytes', 'shared_output_pool_bytes',
              'embedding_head_shared_bytes', 'head_input_bytes', 'head_output_bytes', 'proposal_io_bytes',
              'projection_weight_pool_bytes', 'projection_weight_pool_entries',
              'joined_projection_group_count', 'joined_projection_group_bytes')
    assert all(a[field] == b[field] for field in stable), 'Extra owner/weight/layout changed'
    layers = control['decoder_layer_count']
    assert layers == candidate['decoder_layer_count']
    for value in modes:
        bodies = value['memory']['decoder']['graphs'][:-1]
        assert len(bodies) == 2 and all(not g['network_output_aliases_input'] for g in bodies)
        assert sum(g['internal_packed_stores'] for g in bodies) == (2 * layers if value['candidate'] else 0)
    records.append({'model': control['model'], 'batch': batch, 'layers': layers,
                    'vision_initialized':control['vision_initialized'],'speculative_initialized':True,
                    'control_body_context_bytes': [g['context_workspace_required_bytes'] for g in a['graphs'][:-1]],
                    'candidate_body_context_bytes': [g['context_workspace_required_bytes'] for g in b['graphs'][:-1]],
                    'control_shared_activation_bytes': a['shared_activation_pool_bytes'],
                    'candidate_shared_activation_bytes': b['shared_activation_pool_bytes'],
                    'shared_activation_reduction_bytes': a['shared_activation_pool_bytes'] - b['shared_activation_pool_bytes'],
                    'control_owned_decoder_bytes': a['owned_decoder_bytes'],
                    'candidate_owned_decoder_bytes': b['owned_decoder_bytes'],
                    'control_owned_model_bytes': control['memory']['owned_model_bytes'],
                    'candidate_owned_model_bytes': candidate['memory']['owned_model_bytes'],
                    'owned_model_reduction_bytes': control['memory']['owned_model_bytes'] - candidate['memory']['owned_model_bytes'],
                    'unchanged_owner_categories': {field: a[field] for field in stable}})
useful = all(r['shared_activation_reduction_bytes'] > 0 and r['owned_model_reduction_bytes'] > 0 for r in records)
output = {'status': 'REAL_POST_LOAD_ALLOCATION_REDUCTION_OBSERVED' if useful else 'REAL_ALLOCATION_ADMISSION_REJECTED',
          'garnet_dll_sha256': native_sha, 'records': records, 'sources': sources,
          'both_profiles_have_context_and_owned_reduction': useful,
          'scope': 'Four fresh sequential native processes, exact captured serving profiles and source weights; post-load engine requirements and owned counters only.',
          'limits': ['No inference, independent pretrained correctness or latency result.',
                     'Not warmed process memory, physical residency, true lifetime peak or a replacement for prior memory chart.',
                     'Driver/library/transient/build/graph executable allocations may be absent from owned counters.',
                     'Original numeric/SDK, real-image ordinary/prior/repeated, full28, Qwen3VL, timing and isolated memory gates remain open.']}
destination = root / 'internal-packed-store-real-allocation-summary.json'
destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(destination)
if not useful:
    raise SystemExit('Actual model allocation admission rejected; preserve raw evidence')
