"""Synthetic CPU controls for reserve accounting and unchanged full4K gates."""
from pathlib import Path
import copy,importlib,sys,tempfile,unittest
from unittest.mock import patch

class ReservedContextControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(Path(__file__).resolve().parent))
  cls.identity=importlib.import_module('reference_fixed4k_sglang_reserved_identity')
  cls.boundary=importlib.import_module('reference_fixed4k_boundary')
  from .test_reference_fixed4k_identity import ordinary
  cls.fixture=staticmethod(ordinary)
 def proposed(self):
  info=self.fixture();info.update(context_length=4098,max_total_tokens=32784);return info
 def test_two_slot_worker_scheduler_bound_matches_observed126_and_requested128(self):
  for context,wanted in ((4096,126),(4098,128)):
   max_req_len=min(context-1,32784-1)
   self.assertEqual(min(128,max_req_len-3968-1),wanted)
 def test_explicit_engine_and_workload_capacities_are_both_retained(self):
  result=self.identity.validate(self.proposed(),'ordinary')
  self.assertEqual(result['context_capacity'],4096)
  self.assertEqual(result['configured_engine_context_capacity'],4098)
  self.assertEqual(result['verified_settings']['context_length'],4098)
  self.assertEqual(result['configured_engine_token_pool'],32784)
 def test_original_capacity_and_wrong_pool_precision_cache_types_fail(self):
  for key,value in (('context_length',4096),('context_length',True),('max_total_tokens',32768),('kv_cache_dtype','fp8'),('enable_mm_global_cache',True)):
   info=self.proposed();info[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):self.identity.validate(info,'ordinary')
 def test_mtp_retains_existing_exact_resolved_algorithm_gate(self):
  info=self.proposed()
  with self.assertRaises(ValueError):self.identity.validate(info,'mtp')
  info.update(speculative_algorithm='EAGLE',speculative_draft_model_path=self.identity.CHECKPOINT,speculative_num_steps=3,speculative_eagle_topk=1,speculative_num_draft_tokens=4)
  self.assertEqual(self.identity.validate(info,'mtp')['generation_mode'],'mtp')
 def test_frozen_boundary_requests_are_text_and_image_with_full_ids(self):
  cases=self.boundary.frozen_cases(Path(__file__).with_name('evidence')/'fixed-length-4096-corpus.zip')
  self.assertEqual([row[1] for row in cases],['text','image'])
  self.assertTrue(all(len(row[3])==3968 for row in cases))
 def test_126_output_rejection_is_preserved_before_second_request(self):
  import io,json
  from .test_fixed_length_reference import sample
  # Synthetic fixture only; not a server or quality measurement.
  cases=self.boundary.frozen_cases(Path(__file__).with_name('evidence')/'fixed-length-4096-corpus.zip')
  value,_=sample('sglang');value['choices'][0]['prompt_token_ids']=cases[0][3]
  value['choices'][0]['response_token_ids']=[1]*126;value['usage'].update(completion_tokens=126,total_tokens=4094)
  with tempfile.TemporaryDirectory() as tmp:
   output=Path(tmp)/'attempt'
   with patch.object(self.boundary,'urlopen',return_value=io.BytesIO(json.dumps(value).encode())) as call:
    with self.assertRaises(ValueError):self.boundary.collect(Path(__file__).with_name('evidence')/'fixed-length-4096-corpus.zip',output,'http://127.0.0.1:18081')
   self.assertEqual(call.call_count,1);report=json.loads((output/'report.json').read_bytes())
   self.assertEqual(report['status'],'failed');self.assertEqual(report['cases'],[])
   self.assertTrue((output/(cases[0][0]+'-response.json')).is_file())

if __name__=='__main__':unittest.main()
