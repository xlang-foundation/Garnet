from pathlib import Path
import ast,copy,hashlib,json,unittest
from unittest.mock import patch
from . import o128_model_layout as layout
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class O128ModelAdmissionTests(unittest.TestCase):
    def test_probe_owner_derivations_leave_all_original_numeric_predicates_intact(self):
        probe=json.loads((M/'evidence/o128-model-probe-derivation.json').read_bytes());owner=json.loads((M/'evidence/o128-model-owner-derivation.json').read_bytes());self.assertTrue(probe['original_full_state_vector_cache_token_repeat_predicates_unchanged']);self.assertTrue(probe['known_failed_Q_only_never_candidate_admission_oracle']);self.assertTrue(probe['no_gate_exceptions_caught_or_waived']);self.assertTrue(owner['no_original_numerical_state_vector_cache_token_or_repeat_gate_waived'])
        for row in probe['derivations']+owner['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
        raw=(M/'probe_o128_model_admission.py').read_bytes()
        for value in (b'range(128 if admit else 0)',b"same_state(report['ordinary_prefill'],expected['ordinary_prefill']",b"same_state(report['ordinary_final'],expected['ordinary_final']",b"ordinary==expected['ordinary_ids']",b"report['original_reference_primary_equivalence_passed']=admit",b"instrumentation_equivalence_only=(not admit),model_admission_passed=admit",b"except BaseException as error:report.update(status='failed',error=repr(error));save();raise"):self.assertIn(value,raw)
    def test_failed_Q_only_reference_cannot_admit_QplusO_or_decode(self):
        self.assertEqual(layout.reference(32,False,False),('kv128-prefix-cache-attempt1','baseline',True));self.assertEqual(layout.reference(128,True,True),('kv128-prefix-cache-attempt1','baseline',True));self.assertEqual(layout.reference(128,True,False),('q128-model-admission-attempt1','enabled128',False))
        for args in ((32,True,False),(32,False,True),(128,False,False),(128,False,True),(128,1,True),(128,True,1),(True,False,False),(4,False,False)):
            with self.assertRaises(AssertionError):layout.reference(*args)
    def test_O_counter_missing_bool_wrong_branch_rejected_without_Q_guard_changes(self):
        for t,q,o in ((32,False,False),(128,True,False),(128,True,True)):
            graphs={'4':{layout.O:0},str(t):{layout.O:16 if o else 0}}
            with patch.object(layout,'q_check_graphs') as old:
                layout.check_graphs(graphs,t,True,q,o);old.assert_called_once_with(graphs,t,True,q)
                for chunk in graphs:
                    for value in (True,'16',-1,0,1,15,16,17):
                        if type(value) is int and value==graphs[chunk][layout.O]:continue
                        bad=copy.deepcopy(graphs);bad[chunk][layout.O]=value
                        with self.assertRaises(AssertionError):layout.check_graphs(bad,t,True,q,o)
                bad=copy.deepcopy(graphs);del bad[str(t)][layout.O]
                with self.assertRaises(KeyError):layout.check_graphs(bad,t,True,q,o)
