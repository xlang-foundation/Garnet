"""CPU owner and actual-record coverage controls for T128 attention qualification."""
from pathlib import Path
import copy,hashlib,json,tempfile,unittest,zipfile
from .run_bf16_prefill128_attention_component import closed_reference,REFERENCE
from .verify_bf16_prefill128_attention_component import inventory,WORDS,GPU
from .bf16_prefill128_attention_oracle import case_plan

class Controls(unittest.TestCase):
    def test_closed_successful_reference_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'results'/REFERENCE;output.mkdir(parents=True)
            archive=root/'results'/(REFERENCE+'-evidence.zip')
            good={'exit_code':0,'cleanup_finished':True,'server_exit_code':0,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}}
            for patch,accepted in (({},True),({'exit_code':1},False),({'exit_code':False},False),({'cleanup_finished':False},False),({'server_exit_code':True},False),({'source_after_sha256':{}},False),({'runtime_dependencies_after':{}},False)):
                raw=json.dumps(dict(good,**patch)).encode();(output/'workflow.json').write_bytes(raw)
                with zipfile.ZipFile(archive,'w') as z:z.writestr('results/'+REFERENCE+'/workflow.json',raw)
                digest=hashlib.sha256(archive.read_bytes()).hexdigest()
                if accepted:self.assertTrue(closed_reference(root,digest)['cleanup_finished'])
                else:
                    with self.subTest(patch=patch),self.assertRaises(AssertionError):closed_reference(root,digest)
            with self.assertRaises(AssertionError):closed_reference(root,'0'*64)

    def test_full_case_record_inventory_and_faults(self):
        records=[{'event':'gpu_identity','sm':120,'uuid':GPU}]
        for i,c in enumerate(case_plan()):
            records.extend([dict(event='case_started',**c),dict(event='case_collected',**c,witness_offset=i*WORDS*4,witness_bytes=WORDS*4,full_output_words=WORDS,samples=[],rejected_groups=65 if c['id'] in (0,14) else 0,rejected_invocations=128 if c['id'] in (0,14) else 0,readonly_exact=True,canaries_exact=True,prefix_cache_exact=bool(c['end']),graph_replay_exact=c['id'] in (1,15,127))])
        records.append({'status':'prefill128_attention_collected_needs_full_fp64_replay','cases':36,'witness_bytes':36*WORDS*4,'performance_admitted':False,'model_qualified':False})
        self.assertEqual(len(inventory(records)),36)
        for key,value in (('witness_bytes',4),('full_output_words',1),('rejected_invocations',65),('readonly_exact',False),('prefix_cache_exact',1),('graph_replay_exact',True),('chunk',32),('samples',[[0,0,0]])):
            broken=copy.deepcopy(records);broken[2][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):inventory(broken)
        with self.assertRaises(AssertionError):inventory(records[:-1])

if __name__=='__main__':unittest.main()
