import unittest
from .kv128_canonical_source import sources
class KV128(unittest.TestCase):
    def test_checked_source_composition(self):
        old,new,p=sources();self.assertEqual((len(old),len(new),len(p['modified_files']),len(p['new_files'])),(44,50,5,6))
    def test_no_external_opaque_algorithm_import(self):
        _,new,_=sources();raw=new['src/backends/tensorrt/modelopt_fp8_kv128_canonical.cpp']
        self.assertNotIn(b'std::memcmp',raw);self.assertNotIn(b'expectedOpaque',raw)
        self.assertIn(b'context->info.opaque=choices[i].algo',raw)
        self.assertEqual(raw.count(b'KVConfigEqual<'),6);self.assertEqual(raw.count(b'trace.api("cublasLtMatmulAlgoCheck"'),2)
    def test_isolated_wire_and_explicit_route(self):
        _,new,_=sources();prefix='src/backends/tensorrt/'
        self.assertIn(b'0x4b313238',new[prefix+'modelopt_fp8_kv128_plugin.h'])
        self.assertIn(b'GarnetModelOptFP8KV128Canonical',new[prefix+'modelopt_fp8_kv128_plugin.h'])
        self.assertIn(b'GARNET_FP8_FULL_ATTN_KV128_CANONICAL',new[prefix+'modelopt_fp8_lowering.cpp'])
        self.assertIn(b'fp8_kv128',new[prefix+'modelopt_fp8_kv128_plugin.cpp'])
    def test_counter_and_no_new_cuda(self):
        _,new,p=sources();self.assertFalse(any(n.endswith('.cu') for n in p['new_files']))
        self.assertIn(b'native_fp8_kv128_canonical_projections',new['src/backends/tensorrt/cuda_compiled_graph.cpp'])
if __name__=='__main__':unittest.main()
