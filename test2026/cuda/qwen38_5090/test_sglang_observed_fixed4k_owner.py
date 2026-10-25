"""Synthetic native-closure/namespace controls; no model/server/GPU import."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .run_sglang_observed_fixed4k import closed_native_full4k,job_path


def fixture(root,**changes):
    name='mtp-prefill16-full4k-admission-attempt1';folder=root/'results'/name;folder.mkdir(parents=True);w={'exit_code':0,'cleanup_finished':True,'library_unchanged':True,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}};w.update(changes);raw=json.dumps(w).encode();(folder/'workflow.json').write_bytes(raw);path='results/'+name+'/workflow.json';archive=root/'results'/(name+'-evidence.zip')
    with zipfile.ZipFile(archive,'x') as z:z.writestr(path,raw);z.writestr('evidence-manifest.json',json.dumps({'files':{path:hashlib.sha256(raw).hexdigest()}}))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();q={'status':'mtp_prefill16_full4k_target_state_replayed' if w['exit_code']==0 else 'mtp_prefill16_full4k_failure_preserved','archive_sha256':digest};(root/'results'/(name+'-verified.json')).write_text(json.dumps(q));return digest


class ObservedOwnerControls(unittest.TestCase):
    def test_named_closed_success_or_preserved_failure_only_release_exclusive_next_job(self):
        for code in (0,1):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);digest=fixture(root,exit_code=code);self.assertEqual(closed_native_full4k(root,digest)['exit_code'],code)
    def test_live_drifted_or_changed_library_cannot_release_job(self):
        for changes in ({'cleanup_finished':False},{'library_unchanged':False},{'exit_code':None},{'exit_code':True},{'source_after_sha256':{}},{'runtime_dependencies_after':{}}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as t:
                root=Path(t);digest=fixture(root,**changes)
                with self.assertRaises(AssertionError):closed_native_full4k(root,digest)
        with self.assertRaises(ValueError):closed_native_full4k(Path('.'),'a'*65)
    def test_fresh_ordinary_job_namespace_and_mismatched_proof(self):
        root=Path('synthetic');self.assertEqual(job_path(root,'sglang-observed-fixed4k-ordinary-attempt1'),root/'results/sglang-observed-fixed4k-ordinary-attempt1')
        for name in ('../other','sglang-fixed4k-ordinary-attempt1','sglang-observed-fixed4k-mtp-attempt1','sglang-observed-fixed4k-ordinary-attempt0'):
            with self.assertRaises(ValueError):job_path(root,name)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);digest=fixture(root);p=root/'results/mtp-prefill16-full4k-admission-attempt1-verified.json';q=json.loads(p.read_bytes());q['archive_sha256']='f'*64;p.write_text(json.dumps(q))
            with self.assertRaises(AssertionError):closed_native_full4k(root,digest)


if __name__=='__main__':unittest.main()
