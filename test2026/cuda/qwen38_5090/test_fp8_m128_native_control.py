"""CPU fault controls; no native runtime/model/GPU execution."""
import copy,unittest
from .fp8_m128_native_control import wire_contract,native_log

class Controls(unittest.TestCase):
    def test_actual_wire_contract_rejects_missing_wrong_and_bool_integer_fields(self):
        report={'status':'fp8_m128_plugin_cpu_contract_passed','wire_bytes':20,'wire_policy':128,'roundtrips':1,'clones':1,'bad_wires':33,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':3,'workspace_bytes':9175040,'unexpected_gpu_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'runtime_version':101700,'header_version':101700,'loaded_tensorrt_dll':'D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll'}
        wire_contract(report)
        for key in report:
            bad=copy.deepcopy(report);del bad[key]
            with self.assertRaises((AssertionError,KeyError)):wire_contract(bad)
        for key,value in (('wire_policy',1),('bad_descriptors',22),('unexpected_gpu_calls',1),('model_qualified',True),('runtime_version',1),('roundtrips',True),('loaded_tensorrt_dll','C:/other/nvinfer_10.dll')):
            bad=dict(report);bad[key]=value
            with self.assertRaises(AssertionError):wire_contract(bad)

    def test_native_algorithm_workspace_opaque_and_observation_faults(self):
        prefix='CUBLASLT_VERSION 130400\nCUBLASLT_DLL D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll\n'
        log=prefix+'NATIVE_M128_CONTEXT algorithm=35 device=0 required_workspace=5246879 opaque=abcd\n'+''.join('ALGORITHM status=0 id_query=0 id=35 workspace='+str((8 if m==128 else 4)<<20)+' opaque=abcd\n' for m in (16,32,128) for r in (0,1))+'NATIVE_M128_TYPED_HELPER_PASSED\n'
        native_log(log,'native')
        for old,new in (('130400','130300'),('workspace=8388608','workspace=4194304'),('required_workspace=5246879','required_workspace=1'),('opaque=abcd','opaque=ffff'),('id=35','id=67'),('status=0','status=1'),('NATIVE_M128_TYPED_HELPER_PASSED','NATIVE_M128_TYPED_HELPER_FAILED')):
            with self.assertRaises(AssertionError):native_log(log.replace(old,new,1),'native')
        with self.assertRaises(AssertionError):native_log(log+'SHARED_ALGORITHM_CHECK\n','native')

if __name__=='__main__':unittest.main()
