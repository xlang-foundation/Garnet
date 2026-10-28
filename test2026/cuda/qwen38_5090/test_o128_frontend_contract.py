import ast,copy,json,unittest
from pathlib import Path
from .o128_frontend_contract import CASES,Q,O,KV,ZERO,expected_counts,replay_case,replay
def row(spec):
    return dict(spec=copy.deepcopy(spec),actual_flags=dict(spec['flags']),library_sha256='8414795d2953feae49532340b206328972c8a23767276319e6e4fb3e1912c18a',actual_Garnet_frontend_requested=True,status='expected_native_rejection' if spec['expected_error'] else 'actual_frontend_compiled',compiled=spec['expected_error'] is None,graph_stats=None if spec['expected_error'] else {**expected_counts(spec),**{n:0 for n in ZERO},'checkpoint_weight_precision':'modelopt_mixed','native_fp8_projection_weights':4},native_error=spec['expected_error'])
class FrontendControls(unittest.TestCase):
    def test_matrix_and_mandatory_actual_errors(self):
        self.assertEqual(len(CASES),17);self.assertEqual(len({s['key'] for s in CASES}),17)
        report=dict(status='actual_O128_frontend_contracts_collected',fresh_process_per_case=True,cases=[row(s) for s in CASES],model_or_performance_qualified=False);result=replay(report);self.assertEqual((result['compiled_cases'],result['native_rejections']),(6,11))
        for i,s in enumerate(CASES):
            bad=copy.deepcopy(report)
            if s['expected_error']:bad['cases'][i]['native_error']='unrelated error'
            else:bad['cases'][i]['graph_stats'][Q]=True
            with self.assertRaises(AssertionError):replay(bad)
    def test_native_selection_and_case_identity_are_strict(self):
        good=row(CASES[0]);self.assertEqual(expected_counts(CASES[0]),{Q:1,O:1,KV:2})
        for field in (Q,O,KV,*ZERO):
            for value in (True,None,999):
                bad=copy.deepcopy(good);bad['graph_stats'][field]=value
                with self.assertRaises(AssertionError):replay_case(bad,CASES[0])
        bad=copy.deepcopy(good);bad['actual_flags']['GARNET_FP8_FULL_ATTN_O128_CANONICAL']='0'
        with self.assertRaises(AssertionError):replay_case(bad,CASES[0])
    def test_source_uses_fresh_native_process_and_real_decoder(self):
        m=Path(__file__).resolve().parent
        for n in ('o128_frontend_contract.py','probe_o128_frontend.py','launch_o128_frontend_case.py','run_o128_frontend_cases.py'):ast.parse((m/n).read_bytes())
        launch=(m/'launch_o128_frontend_case.py').read_text();probe=(m/'probe_o128_frontend.py').read_text()
        self.assertIn('xlang3.exe',launch);self.assertIn('env=env',launch);self.assertIn('Qwen35CUDAChunk(h,k,v,a,pos,slot',probe);self.assertIn('graph.build(forward(*symbols)',probe)
        for r in json.loads((m/'evidence/o128-frontend-owner-derivation.json').read_bytes())['derivations']:
            import hashlib
            raw=(m/r['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),r['source_sha256'])
            for c in r['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(m/r['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),r['target_sha256'])
