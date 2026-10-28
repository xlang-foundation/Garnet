from pathlib import Path
import hashlib,json,unittest
class Controls(unittest.TestCase):
    def test_only_fixture_source_directory_and_attempt_artifacts_change(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/z128-reconstructed-plugin-attempt2-derivation.json').read_bytes());self.assertEqual(len(p['derivations']),4)
        for row in p['derivations']:
            raw=(m/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                a=c['before'].encode();self.assertEqual(raw.count(a),c['count']);raw=raw.replace(a,c['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
        raw=(m/'build_z128_reconstructed_plugin_init_cpu_attempt2.bat').read_bytes();self.assertIn(b'set "CANDIDATE=%SRC%\\backends\\tensorrt"',raw);self.assertNotIn(b'native_modelopt_init_trace_candidate',raw)
