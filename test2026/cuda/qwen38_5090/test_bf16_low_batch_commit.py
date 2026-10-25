"""CPU-only geometry/byte-reference controls; never imports Garnet or CUDA."""
import unittest
from .bf16_low_batch_layout import build_bf16_low_batch_layout,bf16_low_batch_byte_offset
from .bf16_low_batch_commit_oracle import case_plan,samples,expected,compare,pattern,rounded_bf16
from xModel.qwen35.vl_0_8b.kv_bf16_study_layout import build_bf16_kv_study_layout

class LowBatchCommitControls(unittest.TestCase):
 def test_separate_contract_and_legacy_rejection(self):
  for chunk in (4,32):
   layout=build_bf16_low_batch_layout(chunk=chunk,retain_linear_inputs=chunk==4)
   self.assertEqual(layout['batch'],1);self.assertFalse(layout['runtime_qualified'])
   self.assertEqual(layout['canonical_state_width'],106823680);self.assertEqual(len(layout['layers']),64)
   with self.assertRaises(ValueError):build_bf16_kv_study_layout(batch=1,chunk=chunk,retain_linear_inputs=chunk==4)
   prior=build_bf16_kv_study_layout(batch=4,chunk=chunk,retain_linear_inputs=chunk==4)
   self.assertNotEqual(prior['contract'],layout['contract'])
   for a,b in zip(layout['layers'],prior['layers']):
    self.assertEqual({k:v for k,v in a.items() if not k.endswith('shape')},{k:v for k,v in b.items() if not k.endswith('shape')})
 def test_strict_profile_and_offsets(self):
  for batch in (True,False,0,2,4,8,1.0,'1'):
   with self.assertRaises(ValueError):build_bf16_low_batch_layout(batch=batch)
  for capacity,chunk,retain in ((4095,4,True),(4096,8,False),(4096,32,True),(4096,4,False)):
   with self.assertRaises(ValueError):build_bf16_low_batch_layout(capacity=capacity,chunk=chunk,retain_linear_inputs=retain)
  l=build_bf16_low_batch_layout()
  for layer in range(3,64,4):
   self.assertEqual(bf16_low_batch_byte_offset(l,layer,'value',0,3,4095,255)+2,4*(l['layers'][layer]['offset']+4194304))
  with self.assertRaises(ValueError):bf16_low_batch_byte_offset(l,3,'key',1,0,0,0)
 def test_complete_plan_and_boundaries(self):
  plan=case_plan();self.assertEqual(len(plan),20);self.assertEqual([c['id'] for c in plan],list(range(20)))
  self.assertEqual([sum(c['chunk']==t for c in plan) for t in (4,32)],[13,7]);self.assertEqual(sum(c['phase']=='graph' for c in plan),3)
  self.assertTrue(any(c['count']==4 and c['start']==4092 for c in plan));self.assertTrue(any(c['count']==32 and c['start']==4064 for c in plan))
  for c in plan:
   s=samples(c);self.assertEqual({x['layer'] for x in s},set(range(64)))
   self.assertTrue(all(0<=x['canonical_byte'] and x['canonical_byte']+x['bytes']<=427294720 for x in s))
 def test_exact_reference_and_mutation_rejection(self):
  for c in case_plan():
   raw=expected(c);self.assertTrue(compare(c,raw)['exact']);bad=bytearray(raw);bad[-1]^=1
   with self.assertRaises(ValueError):compare(c,bytes(bad))
  c=case_plan()[0];c['batch']=True
  with self.assertRaises(ValueError):expected(c)
  c=case_plan()[0];c['chunk']=32
  with self.assertRaises(ValueError):expected(c)
 def test_zero_prefix_and_integer_bf16_ties(self):
  for c in case_plan():
   if c['count']==0:self.assertTrue(all(s['bits']==(0xd3d3d3d3 if s['bytes']==4 else 0xd3d3) for s in samples(c)))
  self.assertEqual([rounded_bf16(v) for v in (0x3f808000,0x3f818000,0xbf808000,0xbf818000)],[0x3f80,0x3f82,0xbf80,0xbf82])
  self.assertEqual(pattern(0,19),0x3f000013);self.assertEqual(pattern(1,19),0xbf000074)

if __name__=='__main__':unittest.main()
