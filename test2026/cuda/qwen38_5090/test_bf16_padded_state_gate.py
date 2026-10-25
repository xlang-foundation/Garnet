"""Independent row permutation, exclusion, and malformed-witness controls."""
import copy
import unittest
from test2026.cuda.qwen38_5090.test_bf16_same_precision_gate import states
from test2026.cuda.qwen38_5090.bf16_padded_state_gate import same_logical_row


class PaddedStateGateTests(unittest.TestCase):
    def test_row_permutation_is_explicit_and_readonly(self):
        packed, reference = states()
        for key in ('slots', 'logical_row_sha256'):
            packed[key][0], packed[key][3] = packed[key][3], packed[key][0]
        for component in packed['components']:
            component['row_sha256'][0], component['row_sha256'][3] = component['row_sha256'][3], component['row_sha256'][0]
        before = copy.deepcopy((packed, reference))
        self.assertTrue(same_logical_row(packed, 0, reference, 3))
        self.assertTrue(same_logical_row(packed, 3, reference, 0))
        self.assertFalse(same_logical_row(packed, 0, reference, 0))
        self.assertEqual((packed, reference), before)

    def test_only_selected_valid_row_is_admitted(self):
        packed, reference = states()
        packed['logical_row_sha256'][3] = 'f' * 64
        self.assertTrue(same_logical_row(packed, 0, reference, 0))
        self.assertFalse(same_logical_row(packed, 3, reference, 3))
        packed['components'][127]['row_sha256'][0] = 'f' * 64
        self.assertFalse(same_logical_row(packed, 0, reference, 0))

    def test_invalid_indices_and_unselected_malformed_metadata_rejected(self):
        packed, reference = states()
        for row in (-1, 4, True, 0.0):
            with self.subTest(row=row), self.assertRaises(ValueError):
                same_logical_row(packed, row, reference, 0)
        packed['components'][-1]['row_sha256'][3] = 'invalid'
        with self.assertRaises(ValueError):
            same_logical_row(packed, 0, reference, 0)


if __name__ == '__main__':
    unittest.main()
