"""CPU-only last-reference cleanup and complete component-case requirements."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .run_bf16_prefill128_commit_component import closed_reference,REFERENCE
from .verify_bf16_prefill128_commit_component import inventory

class Controls(unittest.TestCase):
    def test_reference_must_be_closed_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'results'/REFERENCE;folder.mkdir(parents=True)
            archive=root/'results'/(REFERENCE+'-evidence.zip')
            good={'exit_code':0,'cleanup_finished':True,'server_exit_code':0,'source_sha256':{'s':'h'},
                  'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}}
            for patch,accepted in (({},True),({'exit_code':1},True),({'exit_code':False},False),({'cleanup_finished':False},False),
                                   ({'server_exit_code':True},False),({'source_after_sha256':{}},False),({'runtime_dependencies_after':{}},False)):
                w={**good,**patch};raw=json.dumps(w).encode();(folder/'workflow.json').write_bytes(raw)
                with zipfile.ZipFile(archive,'w') as z:z.writestr('results/'+REFERENCE+'/workflow.json',raw)
                digest=hashlib.sha256(archive.read_bytes()).hexdigest()
                if accepted:self.assertTrue(closed_reference(root,digest)['cleanup_finished'])
                else:
                    with self.subTest(patch=patch),self.assertRaises(AssertionError):closed_reference(root,digest)
            with self.assertRaises(AssertionError):closed_reference(root,'0'*64)

    def test_span_graph_and_malformed_inventory(self):
        cases=inventory();self.assertEqual(len(cases),51)
        self.assertEqual(sum(c[-1]==0 for c in cases),16);self.assertEqual(sum(c[2] for c in cases),3)
        for span in ((0,0),(4096,0),(3968,128),(4095,1),(511,128),(512,128),(513,128)):
            self.assertTrue(any(c[:2]==span and c[-1]==0 for c in cases))
        self.assertEqual({c[-1] for c in cases if c[-1]>=10},set(range(10,17))|set(range(20,27)))

if __name__=='__main__':unittest.main()
