"""Synthetic CPU request driver; no Nsight executable, model or GPU imported."""
from pathlib import Path
from unittest.mock import patch
import copy
import json

from . import probe_native_cuda_phases as probe
from . import test_native_host_stage_probe as fixture
from .collect_native_cuda_phases import merge_reports


class ProfiledDriverControls(fixture.ProbeControls):
    def worker_factory(self, command, path):
        child = super().worker_factory(command, path)
        close = child.close
        def close_fixture():
            close()
            # Explicitly synthetic witness only; exercises driver validation.
            Path(command[-1]).with_suffix('.model-closed.json').write_text(json.dumps({
                'status': 'native_model_close_calls_returned', 'mode': command[4],
                'marker_count': 144, 'library_sha256': self.library_sha,
                'scope': 'Synthetic CPU cleanup fixture; no actual model'}))
        child.close = close_fixture
        return child

    def run_probe(self, name):
        out = self.root / 'results' / name
        out.mkdir()
        reports = []
        for mode in ('ordinary', 'mtp'):
            try:
                with patch.object(probe, 'Worker', self.worker_factory), patch.object(probe.sys, 'argv',
                    ['probe', '--root', str(self.root), '--output', str(out), '--cache', str(self.root / 'unused-fixture-cache'), '--mode', mode]):
                    code = probe.main()
                self.assertEqual(code, 0)
                reports.append(json.loads((out / ('report-' + mode + '.json')).read_bytes()))
            except BaseException:
                # Retain the synthetic failure under the old fixture's expected
                # path; production collection keeps the original per-mode file.
                (out / 'report.json').write_bytes((out / ('report-' + mode + '.json')).read_bytes())
                raise
        merged = merge_reports(reports)
        (out / 'report.json').write_text(json.dumps(merged))
        return 0, merged

    def test_merging_requires_both_actual_modes_worker_exits_and_all12_requests(self):
        self.run_probe('synthetic-merge')
        base = self.root / 'results/synthetic-merge'
        reports = [json.loads((base / ('report-' + mode + '.json')).read_bytes()) for mode in ('ordinary', 'mtp')]
        for kind in ('same_mode', 'bool_exit', 'missing_request', 'wrong_ready_mode'):
            bad = copy.deepcopy(reports)
            if kind == 'same_mode':
                bad[1] = copy.deepcopy(bad[0])
            elif kind == 'bool_exit':
                bad[1]['modes']['mtp']['worker_exit_code'] = False
            elif kind == 'missing_request':
                bad[1]['modes']['mtp']['requests'].pop()
            else:
                bad[1]['modes']['mtp']['ready']['generation_mode'] = 'ordinary'
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                merge_reports(bad)
