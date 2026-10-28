import hashlib,json,unittest
from .b4_k_m4x4_source import sources,M

class Composition(unittest.TestCase):
    def test_only_dispatch_and_six_new_files_all_legacy_math_unchanged(self):
        old,new,p=sources();self.assertEqual(len(old),79);self.assertEqual(len(new),85)
        self.assertEqual([n for n in old if old[n]!=new[n]],['src/backends/tensorrt/modelopt_fp8_lowering.cpp']);self.assertEqual(p['new_CUDA_kernels'],0)
        for n in old:
            if n.endswith(('.cu','.cuh')) or 'qk_m16_' in n or 'v_m16_' in n or 'gdn_m16_' in n:self.assertEqual(new[n],old[n])
    def test_each_candidate_body_corruption_rejected(self):
        _,new,p=sources()
        for target in p['modified_files']+p['new_files']:
            with self.assertRaises(AssertionError):sources(lambda name:new[name]+(b' ' if name==target else b''))
    def test_defaultoff_key_only_scope_and_unique_wire(self):
        _,new,p=sources();root='src/backends/tensorrt/';profile=new[root+'modelopt_fp8_k_m4x4_profile.h'];self.assertIn(b'GARNET_FP8_K_BATCH4_DECODE_M4X4',profile);self.assertIn(b'k_proj.weight',profile)
        for name in (b'q_proj.weight',b'v_proj.weight',b'o_proj.weight'):self.assertNotIn(name,profile)
        self.assertIn(b'batch!=4||tokens!=4||inputRows!=16',profile);self.assertIn(b'0x4b343434u',new[root+'modelopt_fp8_k_m4x4_plugin.h'])
        self.assertIn(b'0x514b3136u',new[root+'modelopt_fp8_qk_m16_plugin.h']);self.assertIn(b'0x56313634u',new[root+'modelopt_fp8_v_m16_plugin.h']);self.assertIn(b'0x47313634u',new[root+'modelopt_fp8_gdn_m16_plugin.h'])
    def test_actual_M4_descriptors_four_ordered_calls_and_unchanged_quantization_source(self):
        _,new,p=sources();root='src/backends/tensorrt/';cpp=new[root+'modelopt_fp8_k_m4x4_canonical.cpp'];plugin=new[root+'modelopt_fp8_k_m4x4_plugin.cpp']
        for phrase in (b'context->input,CUDA_R_8F_E4M3,5120,4,5120',b'context->output,CUDA_R_16BF,rows,4,rows',b'for(int group=0;group<4;++group)',b'size_t(group)*4*5120',b'size_t(group)*4*1024*sizeof(uint16_t)',b'if(status!=CUBLAS_STATUS_SUCCESS)return status;'):self.assertIn(phrase,cpp)
        self.assertNotIn(b'context->input,CUDA_R_8F_E4M3,5120,16,5120',cpp)
        self.assertIn(b'QuantizeModelOptFP8Activation(input[0],packed,state_.inputScale,16,5120,stream)',plugin)
        self.assertEqual(p['observed_M4_configuration'],[35,13,17,3,2,0,1]);self.assertFalse(p['model_or_quality_or_performance_qualified'])
    def test_measured_parent_drift_is_preserved_without_unique_cause_claim(self):
        _,_,p=sources();reports={}
        for job,row in p['parents'].items():
            raw=(M/'evidence'/(job+'-verified.json')).read_bytes();q=json.loads(raw);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['proof_sha256']);self.assertEqual(q['archive_sha256'],row['archive_sha256']);self.assertEqual(q['exit_code'],row['exit_code']);reports[job]=q
        q=reports['b4-full4k-selection52-attempt1'];self.assertFalse(q['strict_full4k_admission_passed']);self.assertTrue(q['observation']['all52_consumed_IDs_exact']);self.assertEqual(q['observation']['checkpoints'][1]['comparison']['different_component_indices'][0],70)
        self.assertFalse(reports['b4-full4k-exact-attempt1']['strict_full4k_admission_passed']);self.assertIsNone(reports['b4-full4k-decode-prefix-attempt1']['observation']['first_nonexact_step'])
if __name__=='__main__':unittest.main()
