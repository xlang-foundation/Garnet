"""Native package isolation, receipt, scope and exact source derivation contracts."""
from pathlib import Path
import copy,hashlib,json,unittest
from .fp8_z128_native_audit_contract import OBSERVER,validate
M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()

def receipt():
    return {'abi':24,'observer_library':OBSERVER,'library_sha256':'11'*32,'native_package_library_sha256':'22'*32,'cublaslt_sha256':'33'*32,'expected_m32_opaque':'43'+'00'*63,'observed_helper_files':{'Garnet/test2026/cuda/qwen38_5090/native_fp8_z128_candidate/modelopt_fp8_z128_canonical.cpp':'44'*32,'Garnet/test2026/cuda/qwen38_5090/native_fp8_z128_candidate/modelopt_fp8_z128_canonical.h':'55'*32}}

class NativePackageAudit(unittest.TestCase):
    def test_receipt_exact_pins_and_hash_faults(self):
        r=receipt();self.assertEqual(validate(r,'11'*32,'22'*32,'33'*32)['abi'],24)
        for key in ('library_sha256','native_package_library_sha256','cublaslt_sha256'):
            for value in ('','ff'*32,'gg'*32,True):
                bad=copy.deepcopy(r);bad[key]=value
                with self.assertRaises((AssertionError,ValueError)):validate(bad,'11'*32,'22'*32,'33'*32)

    def test_receipt_ABI_library_scope_and_opaque_faults(self):
        for key,value in (('abi',25),('abi',True),('observer_library','../other.dll'),('expected_m32_opaque','00'),('observed_helper_files',{})):
            bad=receipt();bad[key]=value
            with self.assertRaises(AssertionError):validate(bad,'11'*32,'22'*32,'33'*32)

    def test_exact_all_derivations_and_native_source_identity(self):
        d=json.loads((M/'evidence/fp8-z128-native-package-audit-derivation.json').read_bytes());self.assertEqual(len(d['derivations']),9);self.assertFalse(d['native_helper_or_Garnet_DLL_modified'])
        self.assertEqual(sha((M/'fp8_z128_context_native.cpp').read_bytes()),d['native_adapter_source_sha256'])
        for row in d['derivations']:
            raw=(M/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),1);raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        for name in ('fp8-z128-native-package-audit-verifier-derivation.json','fp8-z128-native-package-audit-analyzer-derivation.json'):
            row=json.loads((M/'evidence'/name).read_bytes());raw=(M/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),1);raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])

    def test_native_call_identity_before_observer_and_no_ffi(self):
        bridge=(M/'fp8_z128_native_audit_bridge.py').read_text();self.assertNotIn('import ctypes',bridge)
        self.assertLess(bridge.index('validate(receipt'),bridge.index('importlib.import_module'))
        self.assertLess(bridge.index('native.library_path()'),bridge.index('native.observe('))
        self.assertIn('loaded.resolve()==package.resolve()',bridge)
        native=(M/'fp8_z128_context_native_observed.cpp').read_text();loader=native[native.index('X3Status Load(void* rawHost'):]
        self.assertNotIn('LoadLibrary',loader);self.assertNotIn('AuditZ128Context',loader)
        self.assertIn('GetModuleHandleExA',native);self.assertNotIn('DllMain',native)

    def test_EHsc_and_CPU_ABI_gate_before_cold_and_main(self):
        build=(M/'build_fp8_z128_context_native_eh.bat').read_text();self.assertIn('/EHsc',build);self.assertNotIn('nvcc',build)
        owner=(M/'run_fp8_z128_native_package_audit.py').read_text()
        self.assertLess(owner.index("out/'native-abi-ledger'"),owner.index("out/'cold-ledger'"));self.assertLess(owner.index("out/'cold-ledger'"),owner.index("out/'native-ledger'"))
        self.assertIn("'performance_admitted':False",owner);self.assertIn("audit=ROOT/p['observer_library']",owner)

if __name__=='__main__':unittest.main()
