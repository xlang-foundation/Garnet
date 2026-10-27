from pathlib import Path
import ast,hashlib,json,unittest
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class O128CPUControlsTests(unittest.TestCase):
    def test_eight_exact_CPU_derivatives_preserve_original_control_sources(self):
        p=json.loads((M/'evidence/o128-canonical-CPU-derivations.json').read_bytes());self.assertEqual(len(p['derivations']),8);self.assertFalse(p['GPU_or_model_executed']);self.assertTrue(p['original_shared_M128_context_factory_stubs_unchanged'])
        build=json.loads((M/'evidence/o128-canonical-native-build-derivation.json').read_bytes());self.assertEqual(len(build['derivations']),2);self.assertTrue(build['old_native_and_numeric_model_gates_not_waived'])
        for row in p['derivations']+build['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
        owner=(M/'run_o128_canonical_native_build.py').read_text();self.assertLess(owner.index('for n,b in parent.items()'),owner.index("run([sys.executable,'-m','unittest'"));self.assertLess(owner.index("run([sys.executable,'-m','unittest'"),owner.index('for n,b in candidate.items()'))
        for name in ('build_q128_profile_cpu.bat','build_o128_profile_cpu.bat','build_q128_factory_cpu.bat','build_q128_plugin_wire_cpu.bat','build_o128_plugin_wire_cpu.bat','build_q128_plugin_init_cpu.bat','build_o128_plugin_init_cpu.bat'):self.assertIn(name,owner)
    def test_shared_context_separate_wire_cross_Q_and_all_eight_init_routes(self):
        from .o128_plugin_init_contract import ROUTES,cpu_contract
        self.assertEqual(len(ROUTES),8);self.assertIn('fp8_o128',ROUTES)
        r=dict(status='actual_o128_plugin_initialize_cpu_controls_passed',actual_initialize_calls=72,factory_calls=56,successes=24,failures=48,cached_clone_initializations=8,unexpected_GPU_operator_calls=0,trace_enabled=False,model_or_performance_qualified=False)
        self.assertEqual(cpu_contract(r,b'',False),dict(trace_enabled=False,initialize_calls=72,factory_calls=56,GPU_operator_calls=0))
        raw=(M/'probe_o128_plugin_wire.cpp').read_bytes()
        for value in (b'ModelOptFP8O128Plugin plugin(5120,6144',b'ModelOptFP8Q128Plugin previousQ(12288,5120',b'0x4f313238u',b'previousQWire[4]==0x51313238u',b'badWire==36',b'crossWire==11',b'workspace_bytes\\\":9175040'):self.assertIn(value,raw)
        raw=(M/'probe_o128_plugin_init.cpp').read_bytes();self.assertIn(b'route<8',raw);self.assertIn(b'route==7)return OwnedPlugin(new ModelOptFP8O128Plugin(5120,6144',raw);self.assertIn(b'route==6)return OwnedPlugin(new ModelOptFP8Q128Plugin(12288,5120',raw)
        raw=(M/'probe_o128_profile_cpu.cpp').read_bytes();self.assertIn(b'selected==16',raw);self.assertIn(b'rejected==262',raw);self.assertIn(b'128,5120,6144,name',raw)
