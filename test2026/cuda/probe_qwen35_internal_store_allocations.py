# SPDX-License-Identifier: Apache-2.0
"""Sequential fresh-process post-load allocation admission; no generation/timing."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model

checkpoint, source, destination, selected = sys.argv[1:5]
assert selected in ('0', '1')
candidate = selected == '1'
record = json.loads(Path(source).read_text())[8]
profile = dict(record['profile'])
profile['internalPackedStores'] = candidate
assert profile['sharePackedActivationStorage'] and profile['sharePackedOutputStorage']
assert profile['joinDecodeInputProjections'] and profile['decodeChunk'] == 4
assert profile['prefillChunk'] == 64 and profile['speculativeMTP'] == 3
print('INTERNAL_STORE_REAL_ALLOCATION_LOAD_BEGIN', record['model'], record['batch'], candidate, flush=True)
began = time.time()
model = Model(checkpoint, profile, record['model'])
try:
    memory = model.memory_info()
    layers = len(model.config['text_config']['layer_types'])
    graphs = memory['decoder']['graphs']
    bodies = graphs[:-1]
    assert len(bodies) == 2 and all(not g['network_output_aliases_input'] for g in graphs)
    assert sum(g['internal_packed_stores'] for g in bodies) == (2 * layers if candidate else 0)
    assert sum(g['internal_packed_root_fills'] for g in bodies) == (2 if candidate else 0)
    assert all(g['user_managed_workspace'] and g['external_workspace_bound'] for g in bodies)
    assert memory['decoder']['shared_activation_pool_bytes'] >= max(g['context_workspace_required_bytes'] for g in bodies)
    assert memory['owned_model_bytes'] is not None
    assert model.speculative is not None
    result = {'model': record['model'], 'batch': record['batch'], 'candidate': candidate,'decoder_layer_count':layers,
              'source_profile': source, 'profile': profile, 'memory': memory,
              'vision_initialized':model.vision is not None,'speculative_initialized':model.speculative is not None,
              'load_seconds_not_request_latency': time.time() - began,
              'all_native_profile_assertions_passed': True,
              'scope': 'Post-load owned allocation and engine context diagnostic only. No inference, quality, full-request latency, warmed process measurement or peak/residency claim.'}
    Path(destination).write_text(json.dumps(result, indent=2))
finally:
    model.close()
print('INTERNAL_STORE_REAL_ALLOCATION_DIAGNOSTIC_COMPLETE', flush=True)
