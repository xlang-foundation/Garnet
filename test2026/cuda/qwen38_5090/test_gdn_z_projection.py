"""Synthetic CPU faults only; no fabricated scientific or model vectors."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from .gdn_chunk_stage_layout import pieces
from .gdn_z_projection import N, K, M, bf16_output, vectors, checkpoint_tensor, replay
from .prefill128_fp8_projection import input_oracle


class GDNZProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.broadcast_to(np.arange(M, dtype='<f4')[:, None] / 128, (M, K)).copy().tobytes()
        cls.z32 = np.full((M, N), 0x3f80, dtype='<u2').tobytes()
        changed = bytearray(cls.z32)
        changed[-2:] = struct.pack('<H', 0x3f81)
        cls.z128 = bytes(changed)
        cls.fixture = {'input-m128-k5120.f32': cls.x,
                       'expected-four-m32-n6144.bf16': cls.z32,
                       'expected-m128-n6144.bf16': cls.z128,
                       'input-scale.f32': struct.pack('<f', 1)}
        cast, packed = input_oracle(cls.x, 1)
        cls.actual = {'cast-m128.bf16': cast, 'quantized-m128.e4m3': packed}
        for profile, raw in (('four-m32', cls.z32), ('m128', cls.z128)):
            for repeat in (0, 1):
                cls.actual[profile + '-r' + str(repeat) + '.bf16'] = raw

    def test_complete_control_and_shared(self):
        result = replay(self.fixture.__getitem__, self.actual.__getitem__, 'control')
        self.assertEqual(result['complete_bf16_output_words_replayed'], 4 * M * N)
        shared = {**self.actual, 'm128-r0.bf16': self.z32, 'm128-r1.bf16': self.z32}
        self.assertIn('_shared_', replay(self.fixture.__getitem__, shared.__getitem__, 'shared')['status'])

    def test_last_output_word_and_cast_fault(self):
        with self.assertRaises(AssertionError):
            replay(self.fixture.__getitem__, self.actual.__getitem__, 'shared')
        faulty = {**self.actual, 'cast-m128.bf16': bytes(len(self.actual['cast-m128.bf16']))}
        with self.assertRaises(AssertionError):
            replay(self.fixture.__getitem__, faulty.__getitem__, 'control')

    def test_repeated_rows_rejected(self):
        faulty = {**self.fixture, 'input-m128-k5120.f32': self.x[:16 * K * 4] * 8}
        with self.assertRaises(AssertionError):
            replay(faulty.__getitem__, self.actual.__getitem__, 'control')

    def test_bf16_restoration_and_nonfinite_rejected(self):
        raw = (np.frombuffer(self.z32, dtype='<u2').astype('<u4') << 16).tobytes()
        self.assertEqual(bf16_output(raw), self.z32)
        for word in (0x3f800001, 0x7f800000):
            faulty = bytearray(raw)
            faulty[-4:] = struct.pack('<I', word)
            with self.assertRaises(AssertionError):
                bf16_output(bytes(faulty))

    def test_actual_piece_selection_and_common_input_gate(self):
        files = {}

        def entry(name, raw):
            rows = []
            for offset, count in pieces(len(raw) // 4):
                filename = name + '-' + str(offset) + '.f32'
                value = raw[offset * 4:(offset + count) * 4]
                files[filename] = value
                rows.append({'file': filename, 'offset': offset, 'words': count,
                             'sha256': hashlib.sha256(value).hexdigest()})
            return {'words': len(raw) // 4, 'sha256': hashlib.sha256(raw).hexdigest(), 'parts': rows}

        f32 = lambda raw: (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).tobytes()
        cases = []
        for tokens in (32, 128):
            chunks = []
            z = f32(self.z32 if tokens == 32 else self.z128)
            for start in range(0, M, tokens):
                chunks.append({'start': start, 'arrays': {
                    'input_norm': entry('t' + str(tokens) + '-x' + str(start), self.x[start * K * 4:(start + tokens) * K * 4]),
                    'z': entry('t' + str(tokens) + '-z' + str(start), z[start * N * 4:(start + tokens) * N * 4])}})
            cases.append({'layer': 0, 'tokens': tokens, 'diagnostic': True,
                          'primary_matches_ordinary': True, 'input_lineage': 'synthetic_test_only',
                          'repeats': [chunks]})
        report = {'cases': cases}
        values, result = vectors(report, files.__getitem__, 0)
        self.assertEqual(values['input-m128-k5120.f32'], self.x)
        self.assertEqual(result['different_bf16_words'], 1)
        self.assertEqual(result['first_different_word'], M * N - 1)
        faulty = copy.deepcopy(report)
        faulty['cases'][0]['primary_matches_ordinary'] = False
        with self.assertRaises(AssertionError):
            vectors(faulty, files.__getitem__, 0)
        faulty = copy.deepcopy(report)
        faulty['cases'][0]['repeats'][0][0]['arrays']['input_norm'] = entry('changed-input', bytes(32 * K * 4))
        with self.assertRaises(AssertionError):
            vectors(faulty, files.__getitem__, 0)

    def test_checkpoint_tensor_shape_scale_and_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = 'test.input_scale'

            def write(info, value):
                header = json.dumps({key: info}).encode()
                (root / 'part.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + value)

            index = {'weight_map': {key: 'part.safetensors'}}
            good = {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]}
            write(good, struct.pack('<f', 0.25))
            raw, metadata = checkpoint_tensor(root, index, key, 'F32', [])
            self.assertEqual(struct.unpack('<f', raw)[0], 0.25)
            self.assertEqual(metadata['shard'], 'part.safetensors')
            for info, value in (({**good, 'shape': [1]}, struct.pack('<f', 1)),
                                ({**good, 'data_offsets': [0, 8]}, struct.pack('<f', 1)),
                                (good, struct.pack('<f', 0)),
                                (good, struct.pack('<f', float('nan')))):
                write(info, value)
                with self.assertRaises(AssertionError):
                    checkpoint_tensor(root, index, key, 'F32', [])


if __name__ == '__main__':
    unittest.main()
