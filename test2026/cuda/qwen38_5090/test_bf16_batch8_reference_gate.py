"""Synthetic projection faults only; no actual B8-vs-B4 numerical admission."""
import copy
import hashlib
import struct
import unittest
from .test_bf16_same_precision_gate import states
from .bf16_batch8_reference_gate import four_complete_rows, same_reference_state, same_reference_vectors


def synthetic():
    packed, reference = states()
    for name in ('physical_row_sha256', 'logical_row_sha256'):
        packed[name].extend(hashlib.sha256(('other-'+value).encode()).hexdigest() for value in list(packed[name]))
    packed['slots'].extend([value+1 for value in list(packed['slots'])])
    for component in packed['components']:
        component['row_sha256'].extend(hashlib.sha256(('other-'+value).encode()).hexdigest() for value in list(component['row_sha256']))
    rows = [4, 0, 7, 2]
    reference['slots'] = [packed['slots'][row] for row in rows]
    reference['logical_row_sha256'] = [packed['logical_row_sha256'][row] for row in rows]
    reference['physical_row_sha256'] = list(reference['logical_row_sha256'])
    for actual, expected in zip(packed['components'], reference['components']):
        expected['row_sha256'] = [actual['row_sha256'][row] for row in rows]
    return packed, reference, rows


class Controls(unittest.TestCase):
    def test_selected_rows_keep_real_width_and_do_not_mutate_inputs(self):
        packed, reference, rows = synthetic()
        before = copy.deepcopy((packed, reference))
        self.assertTrue(same_reference_state(packed, reference, rows))
        selected = four_complete_rows(packed, rows)
        self.assertEqual(selected['physical_words_per_row'], 106823680)
        self.assertEqual(reference['physical_words_per_row'], 173932544)
        self.assertEqual((packed, reference), before)

    def test_different_state_component_global_hash_slot_and_row_order_rejected(self):
        for fault in ('component', 'global', 'slot', 'order'):
            packed, reference, rows = synthetic()
            if fault == 'component': reference['components'][-1]['row_sha256'][3] = 'f'*64
            elif fault == 'global': reference['logical_row_sha256'][3] = 'f'*64;reference['physical_row_sha256'][3] = 'f'*64
            elif fault == 'slot': reference['slots'][3] += 1
            else: rows = list(reversed(rows))
            with self.subTest(fault=fault): self.assertFalse(same_reference_state(packed, reference, rows))

    def test_duplicate_boolean_out_of_range_or_missing_row_indices_rejected(self):
        packed, reference, _ = synthetic()
        for rows in ([0, 0, 1, 2], [True, 1, 2, 3], [0, 1, 2, 8], [0, 1, 2], '0123'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                same_reference_state(packed, reference, rows)

    def test_unselected_rows_are_still_validated(self):
        packed, _, _ = synthetic()
        packed['components'][-1]['row_sha256'][7] = 'invalid'
        with self.assertRaises(ValueError): four_complete_rows(packed, [0, 1, 2, 3])

    def test_reference_must_keep_complete_independent_bf16_value_fp32_layout(self):
        for fault in ('physical', 'roundtrip', 'missing'):
            packed, reference, rows = synthetic()
            if fault == 'physical': reference['physical_words_per_row'] = 106823680
            elif fault == 'roundtrip': reference['reference_bf16_roundtrip_checked'] = False
            else: reference['components'].pop()
            with self.subTest(fault=fault), self.assertRaises(ValueError): same_reference_state(packed, reference, rows)

    def test_complete_raw_vector_rows_not_hash_or_prefix_equality(self):
        rows = [4, 0, 7, 2]
        values = [struct.pack('<f', float(i))*5120 for i in range(8)]
        actual = b''.join(values);reference = b''.join(values[i] for i in rows)
        self.assertTrue(same_reference_vectors(actual, reference, rows))
        self.assertFalse(same_reference_vectors(actual, reference[:-4]+struct.pack('<f', 99.), rows))
        with self.assertRaises(ValueError): same_reference_vectors(actual, reference[:-4], rows)
        unselected_bad = bytearray(actual);unselected_bad[5120*4:5120*4+4] = struct.pack('<f', float('nan'))
        with self.assertRaises(ValueError): same_reference_vectors(unselected_bad, reference, rows)


if __name__ == '__main__':
    unittest.main()
