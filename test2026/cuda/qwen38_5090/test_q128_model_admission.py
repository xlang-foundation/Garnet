from pathlib import Path
import ast,copy,hashlib,json,unittest
from unittest.mock import patch
from . import q128_model_layout as layout
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class ModelAdmissionTests(unittest.TestCase):
    def test_probe_batch_metadata_exact_derivation_preserves_original_comparisons(self):
        p=json.loads((M/'evidence/q128-model-probe-derivation.json').read_bytes());self.assertTrue(p['original_full_state_vector_cache_token_repeat_predicates_unchanged']);self.assertTrue(p['no_gate_exceptions_caught_or_waived'])
        owner=json.loads((M/'evidence/q128-model-owner-derivation.json').read_bytes());self.assertTrue(owner['no_original_numerical_state_vector_cache_token_or_repeat_gate_waived'])
        for row in p['derivations']+owner['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
        raw=(M/'probe_q128_model_admission.py').read_bytes();self.assertIn(b'range(128 if tokens==32 or q_enabled else 0)',raw);self.assertIn(b'same_state(report[\'ordinary_prefill\'],expected[\'ordinary_prefill\']',raw);self.assertIn(b'same_state(report[\'ordinary_final\'],expected[\'ordinary_final\']',raw);self.assertIn(b'ordinary==expected[\'ordinary_ids\']',raw);self.assertIn(b'except BaseException as error:report.update(status=\'failed\',error=repr(error));save();raise',raw)
    def test_enabled_Q128_always_uses_correct_T32_oracle_not_failed_native_profile(self):
        self.assertEqual(layout.reference_profile(128,True),'baseline');self.assertEqual(layout.reference_profile(128,False),'native');self.assertEqual(layout.reference_profile(32,False),'baseline')
        for t,q in ((32,True),(4,False),(128,1),(True,False)):
            with self.assertRaises(AssertionError):layout.reference_profile(t,q)
    def test_physical_Q_counter_missing_bool_wrong_route_and_decode_route_rejected(self):
        for t,q in ((32,False),(128,False),(128,True)):
            graphs={'4':{layout.Q:0},str(t):{layout.Q:16 if q else 0}}
            with patch.object(layout,'original_check_graphs') as old:
                layout.check_graphs(graphs,t,True,q);old.assert_called_once_with(graphs,t,True)
                for chunk in graphs:
                    for v in (True,'16',-1,0,1,15,16,17):
                        if type(v) is int and v==graphs[chunk][layout.Q]:continue
                        bad=copy.deepcopy(graphs);bad[chunk][layout.Q]=v
                        with self.assertRaises(AssertionError):layout.check_graphs(bad,t,True,q)
                bad=copy.deepcopy(graphs);del bad[str(t)][layout.Q]
                with self.assertRaises(KeyError):layout.check_graphs(bad,t,True,q)
