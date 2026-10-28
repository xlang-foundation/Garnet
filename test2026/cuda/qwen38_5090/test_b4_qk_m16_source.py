import hashlib,json,unittest
from .b4_qk_m16_source import sources,M
class Composition(unittest.TestCase):
    def test_exact_parent_only_dispatch_and_six_new_files(self):
        old,new,p=sources();self.assertEqual(len(old),73);self.assertEqual(len(new),79);self.assertEqual([n for n in old if old[n]!=new[n]],['src/backends/tensorrt/modelopt_fp8_lowering.cpp']);self.assertEqual(p['new_CUDA_kernels'],0);self.assertFalse(p['model_or_quality_or_performance_qualified'])
    def test_each_candidate_body_corruption_rejected(self):
        _,new,p=sources()
        for target in p['modified_files']+p['new_files']:
            with self.assertRaises(AssertionError):sources(lambda name:new[name]+(b' ' if name==target else b''))
    def test_live_query_and_failed_fullmodel_lineage_preserved(self):
        _,_,p=sources();reports={}
        for job,row in p['parents'].items():
            raw=(M/'evidence'/(job+'-verified.json')).read_bytes();q=json.loads(raw);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['proof_sha256']);self.assertEqual(q['archive_sha256'],row['archive_sha256']);self.assertEqual(q['exit_code'],row['exit_code']);reports[job]=q
        self.assertFalse(reports['b4-v-m16-short-attempt1']['strict_short_admission_passed']);self.assertFalse(reports['fp8-decode-shape-configs-attempt2']['model_matmul_or_performance_qualified']);self.assertEqual(reports['fp8-decode-shape-configs-attempt2']['analysis']['rows'][3]['M4_configuration'],[35,18,16,7,2,1,2])
    def test_defaultoff_QK_scope_and_new_wire(self):
        _,new,p=sources();root='src/backends/tensorrt/';profile=new[root+'modelopt_fp8_qk_m16_profile.h'];self.assertIn(b'GARNET_FP8_QK_BATCH4_DECODE_CANONICAL',profile);self.assertIn(b'if(layer%4!=3)continue;',profile);self.assertIn(b'q_proj.weight',profile);self.assertIn(b'k_proj.weight',profile);self.assertNotIn(b'v_proj.weight',profile);self.assertNotIn(b'o_proj.weight',profile);self.assertIn(b'0x514b3136u',new[root+'modelopt_fp8_qk_m16_plugin.h']);self.assertIn(b'0x56313634u',new[root+'modelopt_fp8_v_m16_plugin.h']);self.assertIn(b'0x47313634u',new[root+'modelopt_fp8_gdn_m16_plugin.h'])
    def test_geometry_specific_configs_and_provider_audit(self):
        _,_,p=sources();self.assertEqual(p['observed_M4_configurations'],{'1024':[35,13,17,3,2,0,1],'12288':[35,18,16,7,2,1,2]});raw=(M/'evidence/fp8-decode-shape-configs-v2-provider-audit.json').read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),p['provider_audit_sha256']);self.assertFalse(json.loads(raw)['numerical_or_model_or_performance_qualified'])
if __name__=='__main__':unittest.main()
