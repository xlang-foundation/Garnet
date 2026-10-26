"""Source provenance for a separate native policy; GPU correctness has its own gate."""
from pathlib import Path
import hashlib,json,unittest
ROOT=Path(__file__).resolve().parents[3];E=Path(__file__).with_name('evidence');sha=lambda raw:hashlib.sha256(raw).hexdigest()

def source(name):
    path=ROOT/name
    if not path.exists():
        # Isolated remote helper study uses exact pinned copies while production
        # native files stay absent/restored. Never mask a present source drift.
        assert name.startswith('src/backends/tensorrt/')
        path=E.parent/'native_fp8_z128_candidate'/Path(name).name
    return path.read_bytes()

class SourceProvenance(unittest.TestCase):
    def test_actual_control_is_exact_derivative_of_closed_real128_collector(self):
        p=json.loads((E/'fp8-z128-native-control-derivation.json').read_bytes());raw=source(p['source']);self.assertEqual(sha(raw),p['source_sha256']);text=raw.decode()
        for step in p['steps']:
            self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
        self.assertEqual(text.encode(),source(p['target']));self.assertEqual(sha(text.encode()),p['target_sha256'])

    def test_exact_new_policy_derivation_and_untouched_legacy(self):
        p=json.loads((E/'fp8-z128-native-source-derivation.json').read_bytes());self.assertEqual(len(p['files']),5)
        for row in p['files']:
            raw=source(row['source']);self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['steps']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),source(row['target']));self.assertEqual(sha(text.encode()),row['target_sha256'])
        for name,h in p['legacy_untouched_sha256'].items():self.assertEqual(sha(source(name)),h)

    def test_opaque_configuration_is_the_closed_real_M32_control(self):
        p=json.loads((E/'fp8-z128-native-source-derivation.json').read_bytes());raw=(E/'gdn-z-native-control-attempt1-verified.json').read_bytes();q=json.loads(raw)
        self.assertEqual(sha(raw),p['qualified_control_proof_sha256']);self.assertEqual(q['archive_sha256'],p['qualified_control_archive_sha256'])
        for profile in q['profiles']:
            for row in profile['algorithm_replay']['calls']:
                if row['m']==32 or profile['algorithm_replay']['mode']=='shared':self.assertEqual(row['actual_opaque'],p['opaque_configuration'])
        self.assertFalse(q['model_or_quality_or_performance_qualified'])

if __name__=='__main__':unittest.main()
