"""Pure CPU exact source composition; no native runtime/GPU execution."""
from pathlib import Path
import hashlib,json,unittest
M=Path(__file__).resolve().parent;ROOT=M.parents[2];E=M/'evidence';sha=lambda raw:hashlib.sha256(raw).hexdigest()

class Controls(unittest.TestCase):
    def test_new_source_derivation_and_profile_header(self):
        r=json.loads((E/'gdn-chunk-stage-native-derivation.json').read_bytes())
        for row in r['modified']:
            raw=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(raw),row['before_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(ROOT/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        for name,h in r['new_files'].items():self.assertEqual(sha((ROOT/name).read_bytes()),h)

    def test_previous_fp8_native_source_is_exact_composed_before_stage(self):
        stage=json.loads((E/'gdn-chunk-stage-native-derivation.json').read_bytes());before={r['target']:r['original_snapshot'] for r in stage['modified']}
        fp8=json.loads((E/'fp8-m128-native-source-derivation.json').read_bytes())
        for row in fp8['modified_files']:
            raw=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(raw),row['before_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            actual=(ROOT/before.get(row['target'],row['target'])).read_bytes();self.assertEqual(text.encode(),actual);self.assertEqual(sha(actual),row['target_sha256'])
        for name,h in {**fp8['new_files'],**fp8['legacy_untouched_sha256']}.items():self.assertEqual(sha((ROOT/name).read_bytes()),h)

    def test_original_diagnostic_guard_and_all_math_bodies_unchanged(self):
        row=json.loads((E/'gdn-chunk-stage-native-derivation.json').read_bytes())['modified'][0];before=(ROOT/row['original_snapshot']).read_text();after=(ROOT/row['target']).read_text()
        start='                    auto* stageAttribute=FindKeyword(kwParams,"diagnostic_linear_stages");';end='                    std::vector<ITensor*> diagnosticStages;'
        original=before.split(start,1)[1].split(end,1)[0];self.assertIn(start+original,after)
        self.assertIn('const int stageWordLimit=chunkStageStudy?dimensions[1]*10240:1048576;',after)
        for step in row['replacements']:self.assertNotIn('src/cuda/',step['old'])
        # Only trt_builder.cpp plus one new pure header belong to this change.
        record=json.loads((E/'gdn-chunk-stage-native-derivation.json').read_bytes());self.assertEqual([r['target'] for r in record['modified']],['src/backends/tensorrt/trt_builder.cpp'])
        self.assertEqual(set(record['new_files']),{'src/backends/tensorrt/qwen38_gdn_chunk_stage_profile.h'})

if __name__=='__main__':unittest.main()
