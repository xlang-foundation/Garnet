"""CPU-only composition and negative controls; native wire/init tests are separate."""
import copy,hashlib,json,unittest
from .z128_reconstructed_plugin_source import sources,validate,parent_controls,T,M
class Controls(unittest.TestCase):
    def test_exact_qualified_source_composition(self):
        old,new,p=sources();self.assertTrue(validate(old,new,p));self.assertEqual(len(p['new_files']),4);self.assertEqual(len(p['modified_files']),7)
    def test_historical_86_controls_keep_exact_parent_source_assertions(self):
        self.assertEqual(parent_controls(),86)
    def test_wrong_factory_wire_counter_or_math_source_is_rejected(self):
        old,new,p=sources()
        for name,a,b in [('modelopt_fp8_z128_reconstructed_plugin.cpp',b'CreateModelOptFP8Z128ReconstructedContext()',b'CreateModelOptFP8Z128Context()'),('modelopt_fp8_z128_reconstructed_plugin.h',b'0x52313238',b'0x5a313238'),('modelopt_fp8_lowering.cpp',b'else if(canonicalZ128)++',b'if(canonicalZ128)++'),('modelopt_fp8_z128_reconstructed_plugin.cpp',b'128,5120,stream',b'32,5120,stream')]:
            bad=dict(new);key=T+name;self.assertIn(a,bad[key]);bad[key]=bad[key].replace(a,b)
            with self.assertRaises(AssertionError):validate(old,bad,p)
    def test_new_counter_has_reset_copy_and_stats_and_separate_default(self):
        _,new,_=sources();self.assertIn(b'else if(canonicalZ128)++nativeFP8Z128ProjectionCount_',new[T+'modelopt_fp8_lowering.cpp'])
        self.assertIn(b'nativeFP8Z128ReconstructedProjectionCount_=0;',new[T+'trt_builder.cpp'])
        raw=new[T+'cuda_compiled_graph.cpp'];self.assertIn(b'nativeFP8Z128ReconstructedProjectionCount_=0;',raw);self.assertIn(b'builder.NativeFP8Z128ReconstructedProjectionCount()',raw);self.assertIn(b'info["native_fp8_z128_reconstructed_projections"]',raw)
    def test_exact_additive_CPU_controls_and_route_contract(self):
        p=json.loads((M/'evidence/z128-reconstructed-plugin-controls-derivation.json').read_bytes())
        for row in p['files']:
            raw=(M/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                before=c['before'].encode();self.assertEqual(raw.count(before),c['count']);raw=raw.replace(before,c['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
    def test_new_initialize_trace_rejects_legacy_marker_wrong_geometry_and_unpaired(self):
        from .z128_reconstructed_plugin_init_contract import replay
        begin=dict(id=1,phase='begin',route='fp8_z128_reconstructed',instance='cpu',rows=6144,columns=5120,policy=0x52313238,context_present=False,factory_called=False,result=-1,exception=False)
        end=dict(begin,phase='end',factory_called=True,context_present=True,result=0);self.assertEqual(replay([begin,end])['factory_calls'],1)
        for field,value in [('policy',0x5a313238),('rows',5120),('columns',6144),('route','fp8_z128')]:
            bad=[dict(begin),dict(end)];bad[0][field]=bad[1][field]=value
            with self.assertRaises(AssertionError):replay(bad)
        with self.assertRaises(AssertionError):replay([begin])
if __name__=='__main__':unittest.main()
