import hashlib,json,unittest
from .b4_v_m16_source import sources,M
class Composition(unittest.TestCase):
    def test_exact_parent_one_dispatch_new_V_route_and_original_math(self):
        old,new,p=sources();self.assertEqual(len(old),67);self.assertEqual(len(new),73);self.assertEqual([name for name in old if old[name]!=new[name]],['src/backends/tensorrt/modelopt_fp8_lowering.cpp']);self.assertEqual(p['BlasWorkspace'],4<<20);self.assertFalse(p['model_or_quality_or_performance_qualified']);self.assertEqual(p['new_CUDA_kernels'],0)
    def test_candidate_body_corruption_rejected(self):
        old,new,p=sources()
        for target in p['modified_files']+p['new_files']:
            with self.assertRaises(AssertionError):sources(lambda name:new[name]+(b' ' if name==target else b''))
    def test_faithful_component_pass_and_failed_SDK_and_model_lineage_preserved(self):
        _,_,p=sources();reports={}
        for job,row in p['parents'].items():
            raw=(M/'evidence'/(job+'-verified.json')).read_bytes();q=json.loads(raw);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['proof_sha256']);self.assertEqual(q['archive_sha256'],row['archive_sha256']);self.assertEqual(q['exit_code'],row['exit_code']);reports[job]=q
        self.assertTrue(reports['connected-fa-value-attempt4']['component_baselines_exact']);sdk=reports['fa-value-sdk-control-attempt1'];self.assertFalse(sdk['strict_replay_passed']);self.assertTrue(sdk['observation']['same_M4_algorithm_at_M16_full_words_exact']);self.assertEqual(sdk['algorithm_replay']['reference_algorithms'],dict(v3=[35,13,17,3,2,0,1],v11=[35,13,17,3,2,0,1]));self.assertFalse(reports['b4-b1-short-m16-admission-attempt1']['strict_short_admission_passed'])
    def test_separate_defaultoff_V_scope_and_wire_preserved(self):
        _,new,_=sources();p='src/backends/tensorrt/';profile=new[p+'modelopt_fp8_v_m16_profile.h'];self.assertIn(b'GARNET_FP8_V_BATCH4_DECODE_CANONICAL',profile);self.assertIn(b'if(layer%4!=3)continue;',profile);self.assertIn(b'v_proj.weight',profile);self.assertNotIn(b'in_proj_qkv.weight',profile);self.assertIn(b'0x56313634u',new[p+'modelopt_fp8_v_m16_plugin.h']);self.assertIn(b'0x47313634u',new[p+'modelopt_fp8_gdn_m16_plugin.h']);self.assertNotEqual(new[p+'modelopt_fp8_v_m16_plugin.h'],new[p+'modelopt_fp8_gdn_m16_plugin.h'])
if __name__=='__main__':unittest.main()
