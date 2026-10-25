"""Synthetic closed-reference lifecycle controls; no model/runtime/GPU import."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .run_mtp_prefill16_full4k import closed_characterization


def fixture(root,**changes):
    name='sglang-output-characterization-attempt1';folder=root/'results'/name;folder.mkdir(parents=True);w={'exit_code':1,'cleanup_finished':True,'server_exit_code':143,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}};w.update(changes);raw=json.dumps(w).encode();(folder/'workflow.json').write_bytes(raw);path='results/'+name+'/workflow.json';archive=root/'results'/(name+'-evidence.zip')
    with zipfile.ZipFile(archive,'x') as z:z.writestr(path,raw);z.writestr('evidence-manifest.json',json.dumps({'files':{path:hashlib.sha256(raw).hexdigest()}}))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();proof={'status':'sglang_output_characterization_replayed' if w['exit_code']==0 else 'sglang_output_characterization_failure_preserved','archive_sha256':digest};(root/'results'/(name+'-verified.json')).write_text(json.dumps(proof));return name,digest


class Full4KOwnerControls(unittest.TestCase):
    def test_closed_success_and_preserved_failure_only_release_next_correctness_job(self):
        for exit_code in (0,1):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);name,digest=fixture(root,exit_code=exit_code);self.assertEqual(closed_characterization(root,name,digest)['exit_code'],exit_code)
    def test_live_or_changed_source_runtime_server_ownership_is_rejected(self):
        for changes in ({'cleanup_finished':False},{'exit_code':None},{'exit_code':True},{'server_exit_code':None},{'server_exit_code':True},{'source_after_sha256':{}},{'runtime_dependencies_after':{}}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as t:
                root=Path(t);name,digest=fixture(root,**changes)
                with self.assertRaises(AssertionError):closed_characterization(root,name,digest)
    def test_wrong_job_hash_or_proof_cannot_release_job(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);name,digest=fixture(root)
            for bad_name,bad_digest in (('../other',digest),(name,digest+'0')):
                with self.assertRaises(ValueError):closed_characterization(root,bad_name,bad_digest)
            p=root/'results'/(name+'-verified.json');d=json.loads(p.read_bytes());d['status']='sglang_output_characterization_replayed';p.write_text(json.dumps(d))
            with self.assertRaises(AssertionError):closed_characterization(root,name,digest)


if __name__=='__main__':unittest.main()
