"""Synthetic CPU checker controls; these records are never GPU evidence."""
import copy,hashlib,unittest
from .bf16_low_batch_commit_oracle import case_plan,expected
from .replay_bf16_low_batch_commit_observations import replay

class B1CommitObservationControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=b''.join(expected(c) for c in case_plan());cls.records=[{'event':'gpu_identity','name':'Synthetic CPU fixture','sm':120,'uuid':'GPU-SYNTHETIC-CPU-CONTROL'}];cursor=0
  for c in case_plan():
   size=len(expected(c));cls.records.append({'event':'case_collected',**c,'witness_offset':cursor,'witness_bytes':size,'gpu_full_state_exact':True,'host_full_state_exact':True,'canaries_exact':True});cursor+=size
   if c['id'] in (12,19):cls.records.append({'event':'profile_passed','batch':1,'chunk':c['chunk'],'full_state_byte_cases':13 if c['chunk']==4 else 7,'graph_profiles':2 if c['chunk']==4 else 1,'graph_replays_each':3,'rejected_calls':33,'readonly_inputs_exact':True,'canaries_exact':True})
  cls.records.append({'status':'bf16_low_batch_commit_collected_needs_byte_replay','cases':20,'witness_bytes':104704,'performance_admitted':False,'model_qualified':False})
 def check(self,records=None,data=None):return replay(self.records if records is None else records,self.raw if data is None else data,self.raw,gpu_uuid='GPU-SYNTHETIC-CPU-CONTROL')
 def test_frozen_integer_fixture_and_complete_stream(self):
  self.assertEqual(hashlib.sha256(self.raw).hexdigest(),'bf4dde202004993cf53d3c20ff7a4edfa3ffc90a6ff75444aa587a5782880c69');self.assertEqual(len(self.check()),20)
 def test_raw_corruption_and_truncation(self):
  changed=bytearray(self.raw);changed[-1]^=1
  with self.assertRaises(ValueError):self.check(data=bytes(changed))
  with self.assertRaises(AssertionError):self.check(data=self.raw[:-1])
 def test_missing_or_reordered_case(self):
  r=copy.deepcopy(self.records);r[1],r[2]=r[2],r[1]
  with self.assertRaises(AssertionError):self.check(records=r)
  with self.assertRaises(AssertionError):self.check(records=self.records[:-1])
 def test_wrong_identity_and_profile_counts(self):
  r=copy.deepcopy(self.records);r[0]['uuid']='GPU-WRONG'
  with self.assertRaises(AssertionError):self.check(records=r)
  r=copy.deepcopy(self.records);r[14]['rejected_calls']=32
  with self.assertRaises(AssertionError):self.check(records=r)
 def test_bool_dimensions_and_false_canary(self):
  r=copy.deepcopy(self.records);r[1]['batch']=True
  with self.assertRaises(AssertionError):self.check(records=r)
  r=copy.deepcopy(self.records);r[1]['canaries_exact']=False
  with self.assertRaises(AssertionError):self.check(records=r)
 def test_offset_and_model_admission_tampering(self):
  r=copy.deepcopy(self.records);r[1]['witness_offset']=1
  with self.assertRaises(AssertionError):self.check(records=r)
  r=copy.deepcopy(self.records);r[-1]['model_qualified']=True
  with self.assertRaises(AssertionError):self.check(records=r)

if __name__=='__main__':unittest.main()
