"""Pure CPU object provenance faults, no runtime/GPU or production build mutation."""
from pathlib import Path
import hashlib,json,unittest
from .run_bf16_prefill128_fp8_native_build import validate_cuda_objects

class Controls(unittest.TestCase):
    def test_only_three_owned_cuda_objects_can_change_and_no_set_drift(self):
        names=['qwen38_bf16_kv_low_batch_overlay.cu.obj','qwen38_bf16_prefill128_attention.cu.obj','qwen38_bf16_prefill128_commit.cu.obj']+['other'+str(i)+'.cu.obj' for i in range(31)]
        before={'build/'+name:'old' for name in names};self.assertEqual(validate_cuda_objects(before,dict(before)),[])
        for name in names[:3]:
            after=dict(before);after['build/'+name]='new';self.assertEqual(validate_cuda_objects(before,after),['build/'+name])
        for action in ('foreign','added','removed','renamed'):
            after=dict(before)
            if action=='foreign':after['build/other0.cu.obj']='new'
            elif action=='added':after['build/new.cu.obj']='new'
            elif action=='removed':del after['build/other0.cu.obj']
            else:after['outside/other0.cu.obj']=after.pop('build/other0.cu.obj')
            with self.assertRaises(AssertionError):validate_cuda_objects(before,after)

    def test_build_method_source_derivations_are_exact(self):
        root=Path(__file__).resolve().parent;record=json.loads((root/'evidence/prefill128-fp8-native-build-derivation.json').read_bytes());sha=lambda raw:hashlib.sha256(raw).hexdigest()
        for row in record['files']:
            raw=(root/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(root/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
