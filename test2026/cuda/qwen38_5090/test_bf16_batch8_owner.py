"""CPU ownership/archive failure controls; no native runtime or GPU import."""
from pathlib import Path
import hashlib,json,tempfile,time,unittest,zipfile
from .run_bf16_batch8_admission import closed_reference,retire

class OwnershipControls(unittest.TestCase):
 def make_closed(self,root,**changes):
  name='sglang-fixed4k-ordinary-attempt2';out=root/'results'/name;out.mkdir(parents=True)
  w={'exit_code':1,'cleanup_finished':True,'server_exit_code':-15,'source_sha256':{'x':'hash'},'source_after_sha256':{'x':'hash'},'runtime_dependencies':{'garnet':'hash'},'runtime_dependencies_after':{'garnet':'hash'}};w.update(changes);raw=json.dumps(w).encode();(out/'workflow.json').write_bytes(raw)
  archive=root/'results'/(name+'-evidence.zip');key='results/'+name+'/workflow.json'
  with zipfile.ZipFile(archive,'x') as z:z.writestr(key,raw);z.writestr('evidence-manifest.json',json.dumps({'files':{key:hashlib.sha256(raw).hexdigest()}}))
  return name,hashlib.sha256(archive.read_bytes()).hexdigest()
 def test_closed_failed_reference_allows_only_subsequent_untimed_admission(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);name,digest=self.make_closed(root);self.assertEqual(closed_reference(root,name,digest)['exit_code'],1)
 def test_live_unclosed_unknown_or_drifted_reference_is_rejected(self):
  for changes in ({'cleanup_finished':False},{'server_exit_code':None},{'exit_code':None},{'source_after_sha256':{}},{'runtime_dependencies_after':{}}):
   with self.subTest(changes=changes),tempfile.TemporaryDirectory() as t:
    root=Path(t);name,digest=self.make_closed(root,**changes)
    with self.assertRaises(AssertionError):closed_reference(root,name,digest)
  with tempfile.TemporaryDirectory() as t:
   for name in ('../escape','sglang-fixed4k-ordinary-attempt0','other'):
    with self.assertRaises(ValueError):closed_reference(Path(t),name,'a'*64)
 def test_owned_cache_only_retires_new_exact_engine_files(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);cache=root/'cache';cache.mkdir();(cache/'study-owner.json').write_text(json.dumps({'study':str(root)}));old=cache/'garnet-cuda-graph-old';old.mkdir();(old/'model.engine').write_bytes(b'old');before={p.name for p in cache.iterdir()}
   new=cache/'garnet-cuda-graph-new';new.mkdir();(new/'model.engine').write_bytes(b'new');other=cache/'other';other.mkdir()
   records=retire(cache,root,before,time.time()-1);self.assertEqual(len(records),1);self.assertEqual(records[0]['sha256'],hashlib.sha256(b'new').hexdigest());self.assertTrue(old.exists());self.assertTrue(other.exists());self.assertFalse(new.exists())
 def test_unknown_or_unowned_cache_is_preserved(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);cache=root/'cache';cache.mkdir();(cache/'study-owner.json').write_text(json.dumps({'study':'other'}));new=cache/'garnet-cuda-graph-new';new.mkdir();(new/'model.engine').write_bytes(b'keep')
   with self.assertRaises(AssertionError):retire(cache,root,set(),0)
   self.assertTrue((new/'model.engine').exists())
   (cache/'study-owner.json').write_text(json.dumps({'study':str(root)}));(new/'unknown').write_bytes(b'keep')
   with self.assertRaises(AssertionError):retire(cache,root,set(),0)
   self.assertTrue((new/'model.engine').exists());self.assertTrue((new/'unknown').exists())
if __name__=='__main__':unittest.main()
