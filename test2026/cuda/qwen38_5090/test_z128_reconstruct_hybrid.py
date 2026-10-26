from pathlib import Path
import copy,hashlib,json,unittest
from .z128_reconstruct_hybrid import identity
M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
class Controls(unittest.TestCase):
    def test_exact_metadata_only_probe_derivatives(self):
        p=json.loads((M/'evidence/z128-reconstruct-hybrid-derivation.json').read_bytes());self.assertFalse(p['factory_or_CPU_driver_or_algorithm_or_matmul_changed'])
        for row in p['files']:
            raw=(M/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for step in row['changes']:
                before=step['before'].encode();self.assertEqual(raw.count(before),step['count']);raw=raw.replace(before,step['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        self.assertIn(b'GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS',(M/'probe_z128_reconstruct_hybrid_sdk.cpp').read_bytes());self.assertIn(b'if(!blas||!driver||!GetModuleHandleExA',(M/'probe_z128_reconstruct_hybrid_sdk.cpp').read_bytes())
    def test_null_named_DLL_is_not_an_identity_and_static_linkage_is_checked(self):
        exe='D:/CantorAI/5090-study-20261023/results/runtime-test/constructor.exe';paths={'cublasLt64_13.dll':'D:/CantorAI/5090-study-20261023/build/bin/cublasLt64_13.dll','cudaGetDevice':exe,'nvcuda.dll':'C:/Windows/System32/nvcuda.dll'}
        report={'cuda_runtime_linkage':'hybrid_loader','cuda_runtime_status':0,'cuda_driver_status':0,'cuda_runtime_version':13040,'cuda_runtime_header_version':13020,'cuda_driver_version':13040,'named_cudart_DLL_present':False,'module_paths':[{'name':name,'path_hex':path.encode().hex()} for name,path in paths.items()]};deps='    cublasLt64_13.dll\n    KERNEL32.dll\n';mapping=' 0001:0000 cudaGetDevice 000140001000 f cudart:generated_cuda_runtime_api_loader.obj\n';self.assertTrue(identity(report,exe,deps,mapping)['runtime_origin_by_address_and_hybrid_loader_map'])
        windows=copy.deepcopy(report)
        for row in windows['module_paths']:row['path_hex']=bytes.fromhex(row['path_hex']).decode().replace('/',chr(92)).encode().hex()
        self.assertTrue(identity(windows,exe.replace('/',chr(92)),deps,mapping)['runtime_origin_by_address_and_hybrid_loader_map'])
        for key,value in [('cuda_runtime_linkage','dynamic'),('cuda_runtime_version',13010),('cuda_runtime_status',7)]:
            bad=copy.deepcopy(report);bad[key]=value
            with self.assertRaises(AssertionError):identity(bad,exe,deps,mapping)
        with self.assertRaises(AssertionError):identity(report,exe,deps+' cudart64_13.dll\n',mapping)
        with self.assertRaises(AssertionError):identity(report,exe,deps,'cudaGetDevice unknown.lib:unknown.obj')
        bad=copy.deepcopy(report);bad['module_paths'][1]['path_hex']=paths['cublasLt64_13.dll'].encode().hex()
        with self.assertRaises(AssertionError):identity(bad,exe,deps,mapping)
if __name__=='__main__':unittest.main()
