"""Historical schema and synthetic negative controls; never server/performance evidence."""
from pathlib import Path
import copy,json,unittest
from .reference_fixed4k_sglang_identity import validate,CHECKPOINT

def ordinary():
 info=json.loads((Path(__file__).with_name('evidence')/'reference4k-sglang-identity-fixture.json').read_bytes())['actual_historical_fields']
 info['mm_preprocess_cache_size_mb']=0 # Proposed launch option, not historical observation.
 return info

class IdentityControls(unittest.TestCase):
 def test_historical_cache_default_is_not_accepted_as_cache_disabled(self):
  info=ordinary();info['mm_preprocess_cache_size_mb']=None
  with self.assertRaises(ValueError):validate(info,'ordinary')
 def test_proposed_ordinary_cache_zero_with_actual_other_fields(self):
  self.assertEqual(validate(ordinary(),'ordinary')['context_capacity'],4096)
 def test_precision_capacity_cache_graph_and_boolean_mismatches_are_rejected(self):
  for key,value in (('mamba_ssm_dtype','bfloat16'),('context_length',True),('max_running_requests',4),('max_total_tokens',8192),('enable_mm_global_cache',True),('enable_prefix_mm_cache',True),('mm_preprocess_cache_size_mb',False),('cuda_graph_bs_decode',[1,2,4]),('model_path','other-checkpoint')):
   info=ordinary();info[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):validate(info,'ordinary')
 def test_mtp_requires_resolved_eagle_and_exact_proposed_depth(self):
  info=ordinary()
  with self.assertRaises(ValueError):validate(info,'mtp')
  info.update(speculative_algorithm='EAGLE',speculative_draft_model_path=CHECKPOINT,speculative_num_steps=3,speculative_eagle_topk=1,speculative_num_draft_tokens=4)
  self.assertEqual(validate(info,'mtp')['generation_mode'],'mtp')
  for key,value in (('speculative_algorithm','NEXTN'),('speculative_algorithm','EAGLE3'),('speculative_num_steps',7),('speculative_eagle_topk',True),('speculative_draft_model_path','other-model')):
   changed=copy.deepcopy(info);changed[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate(changed,'mtp')
if __name__=='__main__':unittest.main()
