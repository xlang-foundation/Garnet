#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Negative controls against a completed real-weight operator collection.

Clones only small raw reports/outputs; pinned large model assets remain shared.
No inference is executed. Numerical agreement is deliberately not a pass gate.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from audit_mistral_mlp_probe import audit


class AuditControls(unittest.TestCase):
    campaign = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.status = copy.deepcopy(json.loads((self.campaign/'status.json').read_text()))
        for run in self.status['runs']:
            previous = Path(run['output'])
            out = self.root/previous.name
            shutil.copytree(previous, out)
            run['output'] = run['argv'][4] = str(out)

    def save(self):
        (self.root/'status.json').write_text(json.dumps(self.status))

    def rejects(self, reason):
        self.save()
        with self.assertRaisesRegex(ValueError, reason):
            audit(self.root)

    def test_unchanged_collection(self):
        self.save()
        report = audit(self.root)
        self.assertTrue(report['collection_integrity_passed'])
        self.assertFalse(report['coreml_numerical_quality_established'])
        self.assertFalse(report['observed_ane_execution_established'])
        self.assertFalse(report['full_model_or_serving_speed_established'])

    def test_incomplete(self):
        self.status['completed'] = False
        self.rejects('incomplete collection')

    def test_overlapping_processes(self):
        self.status['runs'][1]['started_ns'] = self.status['runs'][0]['started_ns']
        self.rejects('overlap')

    def test_failed_child(self):
        self.status['runs'][0]['returncode'] = 1
        self.rejects('failed run')

    def test_changed_secondary_manifest(self):
        self.status['exports'][-1]['manifest_sha256'] = '0'*64
        self.rejects('changed export manifest')

    def test_missing_cohort(self):
        self.status['runs'].pop()
        self.rejects('missing serial cohort')

    def test_nonfinite_output(self):
        out = Path(self.status['runs'][0]['output'])/'output.f32'
        words = np.fromfile(out, dtype='<f4')
        words[0] = np.nan
        words.tofile(out)
        self.rejects('nonfinite')

    def test_output_changes_between_cohorts(self):
        out = Path(self.status['runs'][-1]['output'])/'output.f32'
        words = np.fromfile(out, dtype='<f4')
        words[0] += 1
        words.tofile(out)
        self.rejects('output changed between fresh cohorts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    AuditControls.campaign = args.campaign.resolve()
    unittest.main(argv=[__file__])
