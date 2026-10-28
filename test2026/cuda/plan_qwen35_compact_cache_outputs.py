"""CPU-only prospective output-layout sizes; no implemented allocation saving."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'compact-kv-output-layout-plan.json'
assert not output.exists(), 'Preserve prospective layout evidence'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(a.root))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


measured = read(a.root / 'internal-packed-store-real-allocation-summary.json')
cases = []
for size in ('2B', '4B'):
    config = read(a.root / f'models/Qwen3.5-{size}/config.json')['text_config']
    channels = 2 * config['linear_num_key_heads'] * config['linear_key_head_dim']
    channels += config['linear_num_value_heads'] * config['linear_value_head_dim']
    linear = channels * config['linear_conv_kernel_dim']
    linear += config['linear_num_value_heads'] * config['linear_key_head_dim'] * config['linear_value_head_dim']
    payload_per_token = channels + config['linear_num_value_heads'] * config['linear_value_head_dim']
    payload_per_token += 2 * config['linear_num_value_heads']
    for batch in (1, 2, 4, 8):
        for capacity in (384, 512):
            canonical = []
            offset = 0
            for index, kind in enumerate(config['layer_types']):
                assert kind in ('linear_attention', 'full_attention')
                width = linear if kind == 'linear_attention' else 2 * config['num_key_value_heads'] * capacity * config['head_dim']
                canonical.append({'layer': index, 'kind': kind, 'offset': offset, 'width': width})
                offset += width
            programs = []
            for tokens in (4, 64):
                hidden = tokens * config['hidden_size']
                payload = sum(tokens * payload_per_token for row in canonical if row['kind'] == 'linear_attention') if tokens == 4 else 0
                compact = []
                cursor = hidden
                for row in canonical:
                    width = row['width'] if row['kind'] == 'linear_attention' else 2 * config['num_key_value_heads'] * tokens * config['head_dim']
                    compact.append({'layer': row['layer'], 'offset': cursor, 'width': width})
                    cursor += width
                programs.append({'tokens': tokens, 'original_width_words_per_row': hidden + offset + payload,
                                 'prospective_compact_width_words_per_row': cursor + payload,
                                 'prospective_payload_offset': cursor, 'payload_width': payload,
                                 'prospective_output_intervals': compact})
            before = max(row['original_width_words_per_row'] for row in programs) * batch * 4
            after = max(row['prospective_compact_width_words_per_row'] for row in programs) * batch * 4
            actual = next((row for row in measured['records'] if row['model'] == f'Qwen3.5-{size}' and row['batch'] == batch), None) if capacity == 512 else None
            if actual:
                assert actual['unchanged_owner_categories']['state_bytes'] == offset * batch * 4
                assert actual['unchanged_owner_categories']['shared_output_pool_bytes'] == before
            cases.append({'model': size, 'batch': batch, 'capacity': capacity,
                          'canonical_state_bytes_unchanged': offset * batch * 4,
                          'canonical_layout': canonical, 'programs': programs,
                          'current_logical_shared_output_pool_bytes': before,
                          'prospective_compact_logical_shared_output_pool_bytes': after,
                          'logical_output_pool_difference_bytes_only': before - after,
                          'current_layout_matches_measured_profile': bool(actual)})
result = {'sources_sha256': sources, 'cases': cases, 'implemented': False,
          'measured_allocation_reduction': False,
          'scope': 'Pure layout arithmetic using checkpoint configurations and the existing measured P64/D4 '
                   'owner layout. Proposed full-attention outputs contain per-token K/V deltas instead of '
                   'full-capacity cache snapshots. Canonical FP32 state, weights, attention math and linear '
                   'state/payload are unchanged in this proposal. No native lowering, commit, rollback, '
                   'ownership, poisoning, memory or performance validation has occurred. Logical output '
                   'pool differences cannot be added to process-counter, TRT-workspace or peak savings.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps([{key: row[key] for key in ('model', 'batch', 'capacity', 'logical_output_pool_difference_bytes_only')}
                  for row in cases if row['current_layout_matches_measured_profile']], indent=2))
