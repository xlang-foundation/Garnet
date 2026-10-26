from pathlib import Path
import hashlib,unittest
from .run_bf16_batch8_external_reference_bound import bound_identity

class Controls(unittest.TestCase):
    def preparation(self):
        path=Path(__file__).with_name('run_bf16_batch8_external_reference_bound.py');key='Garnet/test2026/cuda/qwen38_5090/'+path.name
        pins={key:hashlib.sha256(path.read_bytes()).hexdigest(),'Garnet/owned-method':'b'*64}
        return {'files':pins,'new_files':dict(pins)},key
    def test_bound_inventory_is_owned_copy_without_enumeration(self):
        p,key=self.preparation();r=bound_identity(p);self.assertEqual(r['owned_source_sha256'],p['files'])
        r['owned_source_sha256']['new']='value';self.assertNotIn('new',p['files'])
        self.assertIn('unrelated untracked files excluded',r['scope'])
    def test_missing_or_drifted_launcher_cannot_bind(self):
        p,key=self.preparation();p['files'][key]='0'*64
        with self.assertRaises(AssertionError):bound_identity(p)
        p,key=self.preparation();p['new_files'].pop(key)
        with self.assertRaises(KeyError):bound_identity(p)

if __name__=='__main__':unittest.main()
