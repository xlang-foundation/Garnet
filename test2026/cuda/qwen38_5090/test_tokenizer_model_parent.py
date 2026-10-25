"""Synthetic CPU mutation controls for the extra retained-model comparison gate."""
import copy,unittest
from .compare_tokenizer_model_parent import compare_reports
def sample():
 state={'logical':{'physical_row_sha256':['a'],'components':[{'sha256':['b']}]},'vectors':{'hidden':{'file':'hidden.f32','bytes':4},'normalized':{'file':'norm.f32','bytes':4}}};case={'key':'text','repeat':0,'ordinary_ids':[[1,2]],'generated_ids':[[1,2]],'callbacks':[[0,1,1],[0,2,2]],**{n:copy.deepcopy(state) for n in ('ordinary_state','mtp_state','prefill_state','mtp_prefill_state')}};r={'physical_batch':1,'capacity':4096,'cases':[case],'empty_state':copy.deepcopy(state),'final_reset':copy.deepcopy(state)};return r,copy.deepcopy(r)
read=lambda n:b'1234'
class RetainedModelGate(unittest.TestCase):
 def test_complete_states_and_raw_rows(self):
  a,b=sample();r=compare_reports(a,b,read,read);self.assertEqual((r['cases'],r['captured_states'],r['raw_vectors']),(1,6,12))
 def test_case_identity(self):
  a,b=sample();b['cases'][0]['key']='image'
  with self.assertRaises(ValueError):compare_reports(a,b,read,read)
 def test_canonical_component_hash(self):
  a,b=sample();b['cases'][0]['prefill_state']['logical']['components'][0]['sha256'][0]='c'
  with self.assertRaises(ValueError):compare_reports(a,b,read,read)
 def test_complete_raw_vector(self):
  a,b=sample()
  with self.assertRaises(ValueError):compare_reports(a,b,read,lambda n:b'1235')
 def test_missing_or_extra_state(self):
  for action in ('missing','extra'):
   a,b=sample()
   if action=='missing':del b['final_reset']
   else:b['extra_state']=copy.deepcopy(b['empty_state'])
   with self.assertRaises(ValueError):compare_reports(a,b,read,read)
 def test_callbacks(self):
  a,b=sample();b['cases'][0]['callbacks'][0][1]=99
  with self.assertRaises(ValueError):compare_reports(a,b,read,read)
if __name__=='__main__':unittest.main()
