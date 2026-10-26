"""CPU nativebuild provenance and report faults, no native or GPU execution."""
from pathlib import Path
import hashlib,json,unittest
from .fp8_z128_native_build_contract import z_wire_contract


class Controls(unittest.TestCase):
    def test_actual_Z_wire_report_types_counts_and_scope(self):
        good={'status':'fp8_z128_plugin_cpu_contract_passed','wire_bytes':20,'wire_policy':0x5a313238,'bad_wires':34,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':5,'workspace_bytes':4849664,'unexpected_gpu_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'runtime_version':101601,'header_version':101601,'loaded_tensorrt_dll':'D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll'}
        z_wire_contract(good)
        for key,value in good.items():
            bad=dict(good);del bad[key]
            with self.assertRaises((AssertionError,KeyError)):z_wire_contract(bad)
            bad=dict(good);bad[key]=not value if type(value) is bool else value+1 if type(value) is int else 'wrong'
            with self.assertRaises(AssertionError):z_wire_contract(bad)
        with self.assertRaises(AssertionError):z_wire_contract(dict(good,unexpected_gpu_calls=False))

    def test_nativebuild_methods_are_exact_derivatives(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();r=json.loads((m/'evidence/fp8-z128-native-build-derivation.json').read_bytes())
        for row in r['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
