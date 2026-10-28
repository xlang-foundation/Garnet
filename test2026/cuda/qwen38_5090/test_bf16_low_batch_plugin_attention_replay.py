"""Synthetic mutation controls over a real closed fixture; never new GPU evidence."""
import copy,json,sys,unittest,zipfile
from pathlib import Path
from replay_bf16_low_batch_plugin_attention import records,compare

class ForwardingReplayControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with zipfile.ZipFile(Path(sys.argv[1])) as z:
   base='results/bf16-low-batch-attention-native-attempt1/'
   cls.parent=records(z.read(base+'gpu-ledger/commands.log'));cls.binary=z.read(base+'witnesses.f32')
 def sample(self):
  sample=copy.deepcopy(self.parent);sample[-1].update(status='bf16_low_batch_plugin_collected_needs_fp64_replay',host_plugin_enqueue_calls=87,negative_calls_use_direct_kernel=True);return sample
 def test_synthetic_valid_fixture(self):self.assertEqual(compare(self.parent,self.binary,self.sample(),self.binary)['selected_fp32_scalar_outputs_exact'],119040)
 def test_raw_corruption_and_truncation(self):
  for raw in (self.binary[:-1],bytes([self.binary[0]^1])+self.binary[1:]):
   with self.assertRaises(ValueError):compare(self.parent,self.binary,self.sample(),raw)
 def test_stream_order_and_missing(self):
  samples=[self.sample()[:-1],self.sample()];samples[1][1],samples[1][3]=samples[1][3],samples[1][1]
  for sample in samples:
   with self.assertRaises(ValueError):compare(self.parent,self.binary,sample,self.binary)
 def test_control_and_identity_changes(self):
  for record,key,value in [(0,'uuid','GPU-synthetic-wrong'),(2,'canaries_exact',False),(2,'rejected_calls',51),(4,'graph_replay_exact',False),(2,'readonly_exact',1)]:
   sample=self.sample();sample[record][key]=value
   with self.assertRaises(ValueError):compare(self.parent,self.binary,sample,self.binary)
 def test_wrapper_and_admission_counters(self):
  for key,value in [('host_plugin_enqueue_calls',86),('negative_calls_use_direct_kernel',False),('performance_admitted',True),('model_qualified',True),('cases',True)]:
   sample=self.sample();sample[-1][key]=value
   with self.assertRaises(ValueError):compare(self.parent,self.binary,sample,self.binary)
 def test_parent_corpus_required(self):
  for parent,raw in [(self.parent[:-1],self.binary),(self.parent,self.binary[:-4])]:
   with self.assertRaises(ValueError):compare(parent,raw,self.sample(),self.binary)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
