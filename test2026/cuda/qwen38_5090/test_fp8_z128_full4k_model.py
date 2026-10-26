"""Both native selections and unchanged complete state/vector/ID oracle, CPU only."""
from pathlib import Path
import copy,hashlib,json,unittest
from .prefill128_fp8_model import FIELD as OLD
from .fp8_z128_full4k_model import FIELD,validate_targets,validate_selection,replay_full4k_model
from .test_prefill128_model import data

def selected():
    return {'fp8_m128_policy':'1','fp8_z128_policy':'1','fp8_target_graphs':{'4':{OLD:0,FIELD:0},'128':{OLD:48,FIELD:48}},'fp8_auxiliary_graphs':{name:{OLD:0,FIELD:0} for name in ('draft4','draft16','prefill_norm','vision')}}

class Controls(unittest.TestCase):
    def test_both_target_counters_and_off_control(self):
        a=selected();self.assertEqual(validate_targets(a['fp8_target_graphs'],'1'),{'m128':{'4':0,'128':48},'z128':{'4':0,'128':48}})
        off=copy.deepcopy(a['fp8_target_graphs']);off['128']={OLD:0,FIELD:0};self.assertEqual(validate_targets(off,'0'),{'m128':{'4':0,'128':0},'z128':{'4':0,'128':0}})
        for field in (OLD,FIELD):
            for tokens,value in (('128',47),('4',1),('128',False)):
                bad=selected();bad['fp8_target_graphs'][tokens][field]=value
                with self.assertRaises(AssertionError):validate_targets(bad['fp8_target_graphs'],'1')
        bad=selected();del bad['fp8_target_graphs']['128'][FIELD]
        with self.assertRaises(AssertionError):validate_targets(bad['fp8_target_graphs'],'1')

    def test_auxiliary_policies_and_missing_fields(self):
        good=selected();validate_selection(good);good['fp8_auxiliary_graphs']['prefill_norm']=None;validate_selection(good)
        for name in ('draft4','draft16','prefill_norm','vision'):
            for value in (1,False,None):
                bad=selected()
                if value is None:del bad['fp8_auxiliary_graphs'][name][FIELD]
                else:bad['fp8_auxiliary_graphs'][name][FIELD]=value
                with self.assertRaises(AssertionError):validate_selection(bad)
        bad=selected();bad['fp8_z128_policy']='0'
        with self.assertRaises(AssertionError):validate_selection(bad)

    def test_new_selection_cannot_bypass_original_full_oracle(self):
        a,r,f,c=data();a.update(selected());result=replay_full4k_model(a,f.__getitem__,r,f.__getitem__,c);self.assertEqual(result['captured_states'],32)
        for row in a['cases']:
            if row['case_index']==0:
                for name in ('ordinary_final','candidate_final'):row[name]['logical']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):replay_full4k_model(a,f.__getitem__,r,f.__getitem__,c)

    def test_exact_model_derivations_and_native_composition(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();record=json.loads((m/'evidence/fp8-z128-full4k-model-derivation.json').read_bytes())
        for row in record['derivations']:
            raw=(m/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),change.get('count',1));raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        self.assertFalse(record['numerical_oracle_relaxed']);self.assertEqual(record['candidate_source_count'],35)
        for module in ('test_fp8_z128_native_source','test_fp8_z128_native_integration'):self.assertIn(module,record['composed_native_test_modules'])

if __name__=='__main__':unittest.main()
