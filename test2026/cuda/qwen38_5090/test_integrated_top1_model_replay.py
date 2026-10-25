"""Synthetic checker mutation controls using the actual closed parent fixture."""
from pathlib import Path
import copy,json,sys,unittest,zipfile
from integrated_top1_model_replay import FIELDS,compare

class ReplayControls(unittest.TestCase):
 def fresh(self,mode='1'):
  r=copy.deepcopy(self.parent);r['garnet_dll_sha256']='b'*64;r['sampler_mode']=mode
  snapshot=dict(zip(FIELDS,(mode=='1',128,8192,0,0,'host dispatch attempts including capture construction; excludes GPU graph replays')))
  r['external_compressed_pool'].update(snapshot);r['batch_top1_dispatch_before']=snapshot
  r['batch_top1_dispatch_after']=dict(snapshot);r['batch_top1_dispatch_after'][FIELDS[4 if mode=='1' else 3]]=1
  return r
 def check(self,r,vectors=None):return compare(self.parent,r,self.vectors,self.vectors if vectors is None else vectors,'b'*64,r['sampler_mode'])
 def test_both_synthetic_modes_preserve_full_actual_parent(self):
  for mode in ('0','1'):
   with self.subTest(mode=mode):self.assertEqual(self.check(self.fresh(mode))['raw_vectors_exact'],102)
 def test_token_mutation_rejected(self):
  r=self.fresh();r['cases'][0]['generated_ids'][0][0]^=1
  with self.assertRaises(AssertionError):self.check(r)
 def test_state_mutation_rejected(self):
  r=self.fresh();r['cases'][0]['ordinary_state']['slots'][0]+=1
  with self.assertRaises(AssertionError):self.check(r)
 def test_raw_vector_mutation_rejected(self):
  vectors=dict(self.vectors);key=next(iter(vectors));v=bytearray(vectors[key]);v[0]^=1;vectors[key]=bytes(v)
  with self.assertRaises(AssertionError):self.check(self.fresh(),vectors)
 def test_wrong_route_rejected(self):
  r=self.fresh();r['batch_top1_dispatch_after'][FIELDS[3]]=1
  with self.assertRaises(AssertionError):self.check(r)
 def test_owned_allocation_mutation_rejected(self):
  r=self.fresh();r['external_compressed_pool']['host_embedding_mapped_tensor_bytes']+=2
  with self.assertRaises(AssertionError):self.check(r)
 def test_missing_case_or_extra_vector_rejected(self):
  r=self.fresh();r['cases'].pop()
  with self.assertRaises(AssertionError):self.check(r)
  vectors=dict(self.vectors);vectors['extra.f32']=bytes(81920)
  with self.assertRaises(AssertionError):self.check(self.fresh(),vectors)

if __name__=='__main__':
 assert len(sys.argv)==2;archive=Path(sys.argv[1]);prefix='results/bf16-padded-lifecycle-scope-fixed-attempt2/native/'
 with zipfile.ZipFile(archive) as z:
  ReplayControls.parent=json.loads(z.read(prefix+'report.json'))
  ReplayControls.vectors={n.removeprefix(prefix):z.read(n) for n in z.namelist() if n.startswith(prefix) and n.endswith('.f32')}
 result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplayControls))
 raise SystemExit(0 if result.wasSuccessful() else 1)
