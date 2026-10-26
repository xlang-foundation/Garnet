"""Exact additive builder routing and composed prior stage/native provenance."""
from pathlib import Path
import hashlib,json,unittest
M=Path(__file__).resolve().parent;ROOT=M.parents[2];E=M/'evidence';sha=lambda raw:hashlib.sha256(raw).hexdigest()

class IntegrationProvenance(unittest.TestCase):
    def test_exact_additive_routing_counter_source_changes(self):
        r=json.loads((E/'fp8-z128-native-integration-derivation.json').read_bytes());self.assertEqual(len(r['modified_files']),5)
        for row in r['modified_files']:
            raw=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(raw),row['before_sha256']);text=raw.decode()
            for step in row['steps']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(ROOT/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        self.assertEqual(sha((E/'fp8-z128-native-source-derivation.json').read_bytes()),r['source_policy_derivation_sha256'])

    def test_prior_gdn_stage_and_m128_policy_are_exact_compositions_before_Z(self):
        current=json.loads((E/'fp8-z128-native-integration-derivation.json').read_bytes());original={row['target']:row['original_snapshot'] for row in current['modified_files']}
        prior=lambda name:(ROOT/original.get(name,name)).read_bytes()
        stage=json.loads((E/'gdn-chunk-stage-native-derivation.json').read_bytes());stage_before={row['target']:row['original_snapshot'] for row in stage['modified']}
        for row in stage['modified']:
            raw=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(raw),row['before_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),prior(row['target']));self.assertEqual(sha(text.encode()),row['target_sha256'])
        for name,h in stage['new_files'].items():self.assertEqual(sha((ROOT/name).read_bytes()),h)
        fp8=json.loads((E/'fp8-m128-native-source-derivation.json').read_bytes())
        for row in fp8['modified_files']:
            raw=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(raw),row['before_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            actual=(ROOT/stage_before[row['target']]).read_bytes() if row['target'] in stage_before else prior(row['target'])
            self.assertEqual(text.encode(),actual);self.assertEqual(sha(actual),row['target_sha256'])
        for name,h in {**fp8['new_files'],**fp8['legacy_untouched_sha256']}.items():self.assertEqual(sha((ROOT/name).read_bytes()),h)

if __name__=='__main__':unittest.main()
