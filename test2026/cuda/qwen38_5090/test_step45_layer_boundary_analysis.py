"""CPU adversarial controls for the separate step44/45 observer protocol."""
import copy
import hashlib
import struct
import unittest
from unittest.mock import patch
from . import step45_layer_boundary_analysis as a


def fixture(batch):
    keys = ['selected'] if batch == 1 else ['a', 'b', 'selected', 'd']
    row = {'keys': keys, 'generated_ids': [[100 + i for i in range(53)] for _ in keys], 'layer_observations': []}
    report = {'physical_batch': batch, 'rows': [row], 'boundary_runs': 52,
              'observer': {'layers': list(range(64)), 'tokens': 4, 'hidden': 5120, 'hidden_width': 20480,
                           'primary_width': 17, 'wide_width': 17 + 64 * 20480, 'selected_steps': [44, 45]}}
    raw = bytes(batch * a.ROW_BYTES)
    files = {}
    for step in a.STEPS:
        e = {'step': step, 'program_runs': step, 'consumed_tokens': [143 if step == 44 else 144] * batch, 'boundaries': {}}
        for layer in a.LAYERS:
            name = f'step{step}-layer{layer}.f32'
            files[name] = raw
            e['boundaries'][str(layer)] = {'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        row['layer_observations'].append(e)
    return report, files


class Contract(unittest.TestCase):
    def test_all_layers_steps_and_selected_physical_row(self):
        for batch in (1, 4):
            report, files = fixture(batch)
            observed = a.layers(report, files.__getitem__, 'selected')
            self.assertEqual(set(observed), {44, 45})
            self.assertEqual(len(observed[45]['boundaries']), 64)
            self.assertEqual(len(observed[45]['boundaries'][63]), a.ROW_BYTES)
        report['rows'][0]['keys'] = ['selected', 'b', 'c', 'd']
        with self.assertRaises(AssertionError): a.layers(report, files.__getitem__, 'selected')

    def test_wrong_geometry_step_order_width_runs_and_consumed_ID_rejected(self):
        changes = [lambda r: r['observer'].update(tokens=1), lambda r: r['observer'].update(wide_width=18),
                   lambda r: r.update(boundary_runs=51), lambda r: r['rows'][0]['layer_observations'].reverse(),
                   lambda r: r['rows'][0]['layer_observations'][0].update(consumed_tokens=[1] * 4)]
        for change in changes:
            r, files = fixture(4); change(r)
            with self.assertRaises(AssertionError): a.layers(r, files.__getitem__, 'selected')

    def test_inactive_and_active_words_distinguished(self):
        r, rf = fixture(1); c, cf = fixture(4)
        name = 'step45-layer34.f32'
        raw = bytearray(cf[name]); raw[2 * a.ROW_BYTES + 5120 * 4:2 * a.ROW_BYTES + 5120 * 4 + 4] = struct.pack('<f', 1)
        cf[name] = bytes(raw); c['rows'][0]['layer_observations'][1]['boundaries']['34']['sha256'] = hashlib.sha256(raw).hexdigest()
        x = a.layers(r, rf.__getitem__, 'selected')[45]['boundaries'][34]
        y = a.layers(c, cf.__getitem__, 'selected')[45]['boundaries'][34]
        d = a.difference(x, y, active_words=5120)
        self.assertEqual((d['active_different_words'], d['inactive_different_words']), (0, 1))

    def test_missing_layer_duplicate_file_corrupt_hash_and_nonfinite_rejected(self):
        r, files = fixture(4)
        for change in [lambda v: v.pop('63'), lambda v: v['63'].update(file=v['62']['file']), lambda v: v['63'].update(sha256='0' * 64)]:
            c = copy.deepcopy(r); change(c['rows'][0]['layer_observations'][0]['boundaries'])
            with self.assertRaises(AssertionError): a.layers(c, files.__getitem__, 'selected')
        name = 'step44-layer63.f32'; raw = struct.pack('<f', float('nan')) + files[name][4:]; files[name] = raw
        r['rows'][0]['layer_observations'][0]['boundaries']['63']['sha256'] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError): a.layers(r, files.__getitem__, 'selected')

    def test_primary_parity_must_run_before_any_layer_interpretation(self):
        with patch.object(a, 'primary_analysis', side_effect=AssertionError('primary drift')) as primary, patch.object(a, 'layers') as outputs:
            with self.assertRaisesRegex(AssertionError, 'primary drift'): a.analyze(*([None] * 10))
            primary.assert_called_once(); outputs.assert_not_called()

    def test_timeline_parity_required_and_first_observed_boundary_reported(self):
        r, rf = fixture(1); c, cf = fixture(4)
        for report in (r, c): report['rows'][0]['key35_coordinate_trajectory'] = {'pinned': True}
        oldr, oldc = copy.deepcopy(r), copy.deepcopy(c)
        name = 'step45-layer34.f32'; raw = bytearray(cf[name]); raw[2*a.ROW_BYTES:2*a.ROW_BYTES+4] = struct.pack('<f', 1)
        cf[name] = bytes(raw); c['rows'][0]['layer_observations'][1]['boundaries']['34']['sha256'] = hashlib.sha256(raw).hexdigest()
        with patch.object(a, 'primary_analysis', return_value={'tested': True}):
            d = a.analyze({}, 'selected', r, c, oldr, oldc, rf.__getitem__, cf.__getitem__, None, None)
            self.assertIsNone(d['rows'][0]['first_differing_active_layer'])
            self.assertEqual(d['rows'][1]['first_differing_active_layer'], 34)
            self.assertFalse(d['two_times_target_achieved'])
            c['rows'][0]['key35_coordinate_trajectory'] = {'drift': True}
            with self.assertRaisesRegex(AssertionError, 'timeline'): a.analyze({}, 'selected', r, c, oldr, oldc, rf.__getitem__, cf.__getitem__, None, None)


if __name__ == '__main__': unittest.main()
