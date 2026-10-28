from pathlib import Path
import ast,copy,hashlib,json,unittest
from .o128_full4k_model import Q,O,KV,OLD,RECON,STAGE,validate_targets,validate_selection,replay_full4k_model
from . import test_kv128_prefix_cache as fixture
from .test_prefill128_model import data
def selected():
    graphs=fixture.Prefix().graphs(128)
    for chunk,row in graphs.items():row.update({Q:16 if chunk=='128' else 0,O:16 if chunk=='128' else 0,STAGE:0})
    return dict(fp8_m128_policy='1',fp8_z128_policy='1',fp8_q128_policy='1',fp8_o128_policy='1',fp8_target_graphs=graphs,fp8_auxiliary_graphs={name:{field:0 for field in (Q,O,KV,OLD,RECON,STAGE,'native_fp8_m128_canonical_projections')} for name in ('draft4','draft16','prefill_norm','vision')})
class Full4KControls(unittest.TestCase):
    def test_strict_physical_families_and_no_mutation_of_raw_graphs(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/o128-full4k-probe-derivation.json').read_bytes());self.assertTrue(p['original_six_case_complete_state_vector_ID_callback_reset_predicates_unchanged']);self.assertTrue(p['new_Q_O_KV_reconstructed_Z_physical_checks_additive_only']);self.assertFalse(p['GPU_or_model_executed'])
        owner=json.loads((m/'evidence/o128-full4k-owner-derivation.json').read_bytes());self.assertTrue(owner['all_original_numeric_state_vector_ID_callback_reset_predicates_unchanged']);self.assertFalse(owner['GPU_or_model_executed'])
        for row in p['derivations']+owner['derivations']:
            raw=(m/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
        source=(m/'run_o128_full4k_model.py').read_text();self.assertIn("needed['exit_code']==0",source);self.assertIn("actual_O128_one_text_case_original_admission_and_stages_replayed",source);self.assertLess(source.index("needed['exit_code']==0"),source.index("str(M/'test_o128_full4k_model.bat')"))
        r=selected();before=copy.deepcopy(r);validate_targets(r['fp8_target_graphs'],'1');validate_selection(r);self.assertEqual(r,before)
        for field in (Q,O,KV,OLD,RECON,STAGE,'native_fp8_m128_canonical_projections'):
            for chunk in r['fp8_target_graphs']:
                for value in (True,-1,r['fp8_target_graphs'][chunk][field]+1):
                    bad=selected();bad['fp8_target_graphs'][chunk][field]=value
                    with self.assertRaises(AssertionError):validate_selection(bad)
        with self.assertRaises(AssertionError):validate_targets(r['fp8_target_graphs'],'0')
    def test_auxiliary_routes_and_coupled_flags_cannot_hide_selection(self):
        for name in ('draft4','draft16','prefill_norm','vision'):
            for field in (Q,O,KV,OLD,RECON,STAGE):
                for value in (1,False,None):
                    bad=selected()
                    if value is None:del bad['fp8_auxiliary_graphs'][name][field]
                    else:bad['fp8_auxiliary_graphs'][name][field]=value
                    with self.assertRaises(AssertionError):validate_selection(bad)
        for field in ('fp8_q128_policy','fp8_o128_policy','fp8_z128_policy','fp8_m128_policy'):
            for value in ('0',True,1):
                bad=selected();bad[field]=value
                with self.assertRaises(AssertionError):validate_selection(bad)
        good=selected();good['fp8_auxiliary_graphs']['prefill_norm']=None;validate_selection(good)
    def test_original_six_case_oracle_still_rejects_consistent_candidate_state_corruption(self):
        actual,reference,files,cases=data();actual.update(selected());before=copy.deepcopy(actual);result=replay_full4k_model(actual,files.__getitem__,reference,files.__getitem__,cases);self.assertEqual(result['captured_states'],32);self.assertTrue(result['all_original_six_case_state_vector_ID_callback_reset_predicates_unchanged']);self.assertEqual(actual,before)
        for row in actual['cases']:
            if row['case_index']==0:
                for name in ('ordinary_final','candidate_final'):row[name]['logical']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):replay_full4k_model(actual,files.__getitem__,reference,files.__getitem__,cases)
