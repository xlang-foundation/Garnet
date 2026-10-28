"""CPU closed-native archive guard controls; no runtime/model/GPU import."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .run_mtp_prefill16_model import closed_native_admission

def fixture(root,**changes):
 name='bf16-batch8-model-admission-attempt1';folder=root/'results'/name;folder.mkdir(parents=True)
 w={'exit_code':1,'cleanup_finished':True,'library_unchanged':True,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}};w.update(changes);raw=json.dumps(w).encode();(folder/'workflow.json').write_bytes(raw);path='results/'+name+'/workflow.json';archive=root/'results'/(name+'-evidence.zip')
 with zipfile.ZipFile(archive,'x') as z:z.writestr(path,raw);z.writestr('evidence-manifest.json',json.dumps({'files':{path:hashlib.sha256(raw).hexdigest()}}))
 return hashlib.sha256(archive.read_bytes()).hexdigest()

class ModelOwnerControls(unittest.TestCase):
 def test_closed_prior_failure_is_preserved_and_only_releases_untimed_next_job(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);digest=fixture(root);self.assertEqual(closed_native_admission(root,digest)['exit_code'],1)
 def test_live_drifted_or_changed_runtime_is_refused(self):
  for changes in ({'cleanup_finished':False},{'library_unchanged':False},{'exit_code':None},{'exit_code':True},{'source_after_sha256':{}},{'runtime_dependencies_after':{}}):
   with self.subTest(changes=changes),tempfile.TemporaryDirectory() as t:
    root=Path(t);digest=fixture(root,**changes)
    with self.assertRaises(AssertionError):closed_native_admission(root,digest)
  with self.assertRaises(ValueError):closed_native_admission(Path('.'),'a'*65)
if __name__=='__main__':unittest.main()
