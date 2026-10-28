from pathlib import Path
import ast,hashlib,json,unittest
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class Q128CPUControlsTests(unittest.TestCase):
    def test_all_CPU_templates_exact_derivatives_and_original_files_preserved(self):
        p=json.loads((M/'evidence/q128-canonical-CPU-derivations.json').read_bytes());self.assertEqual(len(p['derivations']),15);self.assertFalse(p['GPU_or_model_executed'])
        build=json.loads((M/'evidence/q128-canonical-native-build-derivation.json').read_bytes());self.assertEqual(len(build['derivations']),2)
        for row in p['derivations']+build['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
            if row['target'].endswith('.py'):ast.parse(raw)
        owner=(M/'run_q128_canonical_native_build.py').read_text();self.assertIn('len(parent)==50 and qualified51==observation51',owner)
        self.assertLess(owner.index('for n,b in parent.items()'),owner.index("run([sys.executable,'-m','unittest'"));self.assertLess(owner.index("run([sys.executable,'-m','unittest'"),owner.index('for n,b in candidate.items()'))
    def test_new_Q_identity_and_fault_envelopes_cover_all_three_geometries(self):
        from .run_q128_factory_cpu import MODES
        self.assertEqual(len(MODES),86)
        for m in (16,32,128):
            for what in ('check_state','workspace'):self.assertIn(what+'_'+str(m),MODES)
        for i in range(7):
            for what in ('value','bytes'):self.assertIn('get_'+what+'_'+str(i),MODES)
        for i in range(1,28):
            for what in ('fail','throw'):self.assertIn(what+'_'+str(i),MODES)
        raw=(M/'probe_q128_plugin_wire.cpp').read_bytes();self.assertIn(b'ModelOptFP8Q128Plugin plugin(12288,5120',raw);self.assertIn(b'ModelOptFP8KV128Plugin previousKV(1024,5120',raw);self.assertIn(b'badWire==35',raw);self.assertIn(b'crossWire==9',raw);self.assertIn(b'markerOnly[4]=0x4b313238u',raw)
        raw=(M/'probe_q128_profile_cpu.cpp').read_bytes();self.assertIn(b'selected==16',raw);self.assertIn(b'rejected==262',raw)
        raw=(M/'probe_q128_plugin_init.cpp').read_bytes();self.assertIn(b'route<7',raw);self.assertIn(b'route==6)return OwnedPlugin(new ModelOptFP8Q128Plugin(12288',raw);self.assertIn(b'route==5)return OwnedPlugin(new ModelOptFP8KV128Plugin(1024',raw)
