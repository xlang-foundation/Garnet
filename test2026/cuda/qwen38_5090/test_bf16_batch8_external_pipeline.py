"""CPU faults for reference-parent admission and exact lifecycle/raw replay derivation."""
from pathlib import Path
import ast,copy,hashlib,json,unittest
from .bf16_batch8_external_reference import BASELINE,validate_parent

M=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def parent():
    proof={'status':'paged_batch8_model_preservation_replayed','library_sha256':BASELINE,'archive_sha256':'a'*64,
           'preserved_model':{'complete_states':29,'raw_vectors':58,'all_retained_states_vectors_tokens_exact':True},
           'external_b1_b4_math_qualified':False,'mtp_qualified':False,'http_qualified':False,
           'performance_admitted':False,'two_times_target_achieved':False}
    w={'library_sha256':BASELINE,'exit_code':0,'cleanup_finished':True,'library_unchanged':True,
       'source_sha256':{'source':'b'*64},'source_after_sha256':{'source':'b'*64},
       'runtime_dependencies':{'garnet.dll':BASELINE},'runtime_dependencies_after':{'garnet.dll':BASELINE},
       'environment_before':{'sdk':'c'*64},'environment_after':{'sdk':'c'*64}}
    return proof,w

class Controls(unittest.TestCase):
    def test_parent_requires_terminal_cleanup_full_state_and_source_runtime_sdk_identity(self):
        proof,w=parent();validate_parent(proof,w,BASELINE,'a'*64)
        for fault in ('exit','bool_exit','cleanup','library','source','runtime','sdk','states','vectors','tokens','archive'):
            p,a=copy.deepcopy(proof),copy.deepcopy(w)
            if fault=='exit':a['exit_code']=1
            elif fault=='bool_exit':a['exit_code']=False
            elif fault=='cleanup':a['cleanup_finished']=False
            elif fault=='library':a['library_sha256']='d'*64
            elif fault=='source':a['source_after_sha256']['source']='d'*64
            elif fault=='runtime':a['runtime_dependencies_after']['garnet.dll']='d'*64
            elif fault=='sdk':a['environment_after']['sdk']='d'*64
            elif fault=='states':p['preserved_model']['complete_states']=28
            elif fault=='vectors':p['preserved_model']['raw_vectors']=57
            elif fault=='tokens':p['preserved_model']['all_retained_states_vectors_tokens_exact']=False
            else:p['archive_sha256']='d'*64
            with self.subTest(fault=fault),self.assertRaises(AssertionError):validate_parent(p,a,BASELINE,'a'*64)

    def test_previous_parent_cannot_be_relabelled_as_mtp_http_or_performance(self):
        for key in ('external_b1_b4_math_qualified','mtp_qualified','http_qualified','performance_admitted','two_times_target_achieved'):
            p,w=parent();p[key]=True
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_parent(p,w,BASELINE,'a'*64)

    def test_exact_derivation_keeps_lifecycle_cleanup_and_raw_resource_replay(self):
        record=json.loads((M/'evidence/bf16-batch8-external-reference-pipeline-derivation.json').read_bytes())
        for name,row in record['files'].items():
            raw=row['source_text'].encode();self.assertEqual(sha(raw),row['source_sha256'])
            text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['before']),step['count']);text=text.replace(step['before'],step['after'])
            self.assertEqual(text.encode(),(M/name).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        verifier=(M/'verify_bf16_batch8_external_reference.py').read_text()
        for operation in ('checked_archive','replay_reference','statistics.median','max(gaps)','source_after_sha256','environment_after'):
            self.assertIn(operation,verifier)
        self.assertNotIn('synthetic_reference',verifier)

    def test_preparation_covers_new_runtime_launcher_and_test_fixtures(self):
        tree=ast.parse((M/'prepare_bf16_batch8_external_reference.py').read_bytes())
        methods=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=='METHODS' for t in n.targets))
        self.assertEqual(len(methods),len(set(methods)))
        for name in ('probe_bf16_batch8_external_reference.py','test_bf16_batch8_external_reference.bat',
                     'run_bf16_batch8_external_reference.py','verify_bf16_batch8_external_reference.py',
                     'bf16_batch8_reference_gate.py','bf16_batch8_external_reference.py',
                     'test_bf16_batch8_external_pipeline.py','evidence/bf16-batch8-external-reference-pipeline-derivation.json'):
            self.assertIn(name,methods);self.assertTrue((M/name).is_file())
        launcher=(M/'test_bf16_batch8_external_reference.bat').read_bytes()
        self.assertNotIn(b'\r',launcher)
        self.assertIn(b'probe_bf16_batch8_external_reference.py',launcher)
        self.assertIn(b'set CUDA_VISIBLE_DEVICES=0',launcher)

if __name__=='__main__':unittest.main()
