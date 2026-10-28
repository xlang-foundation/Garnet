import hashlib,json,unittest
from .b4_decode_m16_source import sources,M
class Composition(unittest.TestCase):
    def test_exact_parent_and_isolated_new_route(self):
        old,new,p=sources();self.assertEqual(len(old),61);self.assertEqual(len(new),67);self.assertEqual([n for n in old if old[n]!=new[n]],['src/backends/tensorrt/modelopt_fp8_lowering.cpp'])
        self.assertEqual(p['BlasWorkspace'],4<<20);self.assertFalse(p['model_or_quality_or_performance_qualified'])
    def test_modified_candidate_and_new_profile_corruption_rejected(self):
        old,new,p=sources()
        for target in p['modified_files']+p['new_files']:
            with self.assertRaises(AssertionError):sources(lambda name:new[name]+(b' ' if name==target else b''))
    def test_actual_SDK_control_lineage(self):
        _,_,p=sources();raw=(M/'evidence/decode-fp8-sdk-control-attempt1-verified.json').read_bytes();q=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),p['SDK_control_proof_sha256']);self.assertEqual(q['archive_sha256'],p['SDK_control_archive_sha256']);self.assertTrue(q['strict_replay_passed']);self.assertEqual(q['algorithm_replay']['reference_algorithms'],dict(qkv=[35,13,17,3,2,0,1],z=[35,13,17,3,2,0,1]))
if __name__=='__main__':unittest.main()
