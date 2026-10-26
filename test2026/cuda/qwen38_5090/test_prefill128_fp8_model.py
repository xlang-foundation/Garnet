"""CPU-only selection faults and original strict full4K oracle preservation."""
from pathlib import Path
import copy,hashlib,json,unittest
from .prefill128_fp8_model import FIELD,validate_targets,validate_selection,replay_prefill128_fp8_model
from .test_prefill128_model import data

def selected():
    return {'fp8_m128_policy':'1','fp8_target_graphs':{'4':{FIELD:0},'128':{FIELD:48}},'fp8_auxiliary_graphs':{name:{FIELD:0} for name in ('draft4','draft16','prefill_norm','vision')}}

class Controls(unittest.TestCase):
    def test_exact_target_selection_and_off_control(self):
        a=selected();self.assertEqual(validate_targets(a['fp8_target_graphs'],'1'),{'4':0,'128':48})
        off=copy.deepcopy(a['fp8_target_graphs']);off['128'][FIELD]=0;self.assertEqual(validate_targets(off,'0'),{'4':0,'128':0})
        for fault in ('missing','bool','wrong128','wrong4','off_selected','bad_policy','extra'):
            graphs=copy.deepcopy(a['fp8_target_graphs']);policy='1'
            if fault=='missing':del graphs['128'][FIELD]
            elif fault=='bool':graphs['4'][FIELD]=False
            elif fault=='wrong128':graphs['128'][FIELD]=47
            elif fault=='wrong4':graphs['4'][FIELD]=48
            elif fault=='off_selected':policy='0'
            elif fault=='bad_policy':policy=True
            else:graphs['16']={FIELD:0}
            with self.subTest(fault=fault),self.assertRaises(AssertionError):validate_targets(graphs,policy)

    def test_all_auxiliary_graphs_must_not_select_m128(self):
        a=selected();validate_selection(a);a['fp8_auxiliary_graphs']['prefill_norm']=None;validate_selection(a)
        for name in ('draft4','draft16','prefill_norm','vision'):
            for bad in ({FIELD:1},{FIELD:False},{}):
                a=selected();a['fp8_auxiliary_graphs'][name]=bad
                with self.subTest(name=name,bad=bad),self.assertRaises(AssertionError):validate_selection(a)

    def test_valid_selection_still_requires_original_complete_retained_oracle(self):
        a,r,f,c=data();a.update(selected());result=replay_prefill128_fp8_model(a,f.__getitem__,r,f.__getitem__,c);self.assertEqual(result['captured_states'],32)
        for row in a['cases']:
            if row['case_index']==0:
                for name in ('ordinary_final','candidate_final'):row[name]['logical']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):replay_prefill128_fp8_model(a,f.__getitem__,r,f.__getitem__,c)

    def test_distinct_model_method_derivations_are_exact(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();record=json.loads((m/'evidence/prefill128-fp8-model-source-derivation.json').read_bytes())
        for row in record['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
