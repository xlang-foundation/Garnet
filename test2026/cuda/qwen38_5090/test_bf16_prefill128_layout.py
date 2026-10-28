"""CPU-only target-prefill metadata/index contracts; no model or GPU admission."""
from pathlib import Path
import copy
import hashlib
import unittest

from .bf16_prefill128_layout import build_bf16_prefill128_layout, target_kv_index, TARGET_WORDS
from .bf16_low_batch_layout import build_bf16_low_batch_layout, bf16_low_batch_byte_offset
from xModel.qwen35.vl_0_8b.bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_program

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def test_canonical_layout_is_unchanged_and_target_intervals_cover_exactly(self):
        base = build_bf16_low_batch_layout(chunk=32, retain_linear_inputs=False)
        candidate = build_bf16_prefill128_layout()
        self.assertEqual(candidate['canonical_state_width'], 106823680)
        self.assertEqual(candidate['linear_replay_width'], 39714816)
        self.assertEqual(candidate['output_width'], TARGET_WORDS)
        self.assertEqual(candidate['output_width'] * 4, 178257920)
        self.assertEqual(candidate['payload_width'], 0)
        self.assertFalse(candidate['runtime_qualified'])
        canonical, target = 0, 128 * 5120
        for original, layer in zip(base['layers'], candidate['layers']):
            for key in ('index', 'kind', 'offset', 'state1_shape', 'state2_shape', 'state1_width', 'state2_width', 'state_precision', 'replay_offset'):
                self.assertEqual(layer[key], original[key])
            self.assertEqual(layer['offset'], canonical)
            self.assertEqual(layer['target_offset'], target)
            canonical += layer['state1_width'] + layer['state2_width']
            target += layer['output_state1_width'] + layer['output_state2_width']
        self.assertEqual(target, TARGET_WORDS)
        self.assertEqual(canonical, 106823680)

    def test_first_last_and_head_boundaries_map_to_distinct_kv_intervals(self):
        layout = build_bf16_prefill128_layout()
        base = build_bf16_low_batch_layout(chunk=32, retain_linear_inputs=False)
        for index in range(3, 64, 4):
            layer = layout['layers'][index]
            self.assertEqual(target_kv_index(layout, index, 'key', 0, 0, 0), layer['target_offset'])
            self.assertEqual(target_kv_index(layout, index, 'key', 127, 3, 255) + 1,
                             target_kv_index(layout, index, 'value', 0, 0, 0))
            self.assertEqual(target_kv_index(layout, index, 'value', 127, 3, 255) + 1,
                             layer['target_offset'] + 2 * 128 * 4 * 256)
            for head in range(4):
                for slot in (0, 1, 511, 512, 513, 3968, 4095):
                    # Canonical packed byte addresses remain those of the admitted
                    # old layout; widening the target is not widening the cache.
                    first = 4 * base['layers'][index]['offset'] + 2 * ((head * 4096 + slot) * 256)
                    self.assertEqual(bf16_low_batch_byte_offset(base, index, 'key', 0, head, slot, 0), first)

    def test_dimensions_retain_and_malformed_indices_rejected(self):
        for key, values in (('batch', (0, 2, 4, 8, True, 1.0)), ('capacity', (4095, 4097, True, 4096.0)),
                            ('chunk', (4, 32, 64, 256, True, 128.0)), ('retain_linear_inputs', (True, 0, 1, None))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    build_bf16_prefill128_layout(**{key: value})
        layout = build_bf16_prefill128_layout()
        for arguments in ((0, 'key', 0, 0, 0), (3, 'other', 0, 0, 0), (3, 'key', -1, 0, 0),
                          (3, 'key', 128, 0, 0), (3, 'key', 0, 4, 0), (3, 'key', 0, 0, 256),
                          (3, 'key', True, 0, 0), (3.0, 'key', 0, 0, 0)):
            with self.assertRaises(ValueError):
                target_kv_index(layout, *arguments)
        damaged = copy.deepcopy(layout)
        damaged['layers'][3]['target_offset'] += 1
        with self.assertRaises(ValueError):
            target_kv_index(damaged, 3, 'key', 0, 0, 0)

    def test_qualified_python_profiles_still_reject_prefill128(self):
        import json
        raw = (M / 'evidence/nvfp4-config.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8')
        config = json.loads(raw)['text_config']
        with self.assertRaises(ValueError):
            validate_bf16_low_batch_kv_program(True, 1, 4096, 128, False, config,
                external_compressed_weights=True, joined_mlp_projections=True)
        self.assertIsNone(validate_bf16_low_batch_kv_program(True, 1, 4096, 32, False, config,
                external_compressed_weights=True, joined_mlp_projections=True))
