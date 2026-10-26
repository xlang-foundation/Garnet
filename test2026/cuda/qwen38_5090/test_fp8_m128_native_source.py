"""CPU-only composed source provenance; runtime/plugin/model gates remain separate."""
from pathlib import Path
import hashlib,json,unittest
M=Path(__file__).resolve().parent;ROOT=M.parents[2];E=M/'evidence';sha=lambda raw:hashlib.sha256(raw).hexdigest()

class Controls(unittest.TestCase):
    def test_recorded_source_edits_and_new_native_files(self):
        record=json.loads((E/'fp8-m128-native-source-derivation.json').read_bytes())
        for row in record['modified_files']:
            source=(ROOT/row['original_snapshot']).read_bytes();self.assertEqual(sha(source),row['before_sha256']);text=source.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(ROOT/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        for name,digest in record['new_files'].items():self.assertEqual(sha((ROOT/name).read_bytes()),digest)

    def test_original_t128_and_python_provenance_remains_composed_exactly(self):
        latest=json.loads((E/'fp8-m128-native-source-derivation.json').read_bytes());before={row['target']:row for row in latest['modified_files']}
        for filename in ('prefill128-native-runtime-derivation.json','prefill128-python-runtime-derivation.json'):
            for row in json.loads((E/filename).read_bytes())['files']:
                source=(ROOT/row['source']).read_bytes();self.assertEqual(sha(source),row['source_sha256'])
                actual=(ROOT/(before[row['target']]['original_snapshot'] if row['target'] in before else row['target'])).read_bytes()
                self.assertEqual(sha(actual),row['target_sha256'])
                if 'replacements' in row:
                    text=source.decode()
                    for step in row['replacements']:
                        self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
                    self.assertEqual(text.encode(),actual)

    def test_legacy_plugin_helper_and_fresh_cache_constructor_are_unchanged(self):
        record=json.loads((E/'fp8-m128-native-source-derivation.json').read_bytes())
        for name,digest in record['legacy_untouched_sha256'].items():self.assertEqual(sha((ROOT/name).read_bytes()),digest)
        row=next(row for row in record['modified_files'] if row['target'].endswith('/cuda_compiled_graph.cpp'))
        before=(ROOT/row['original_snapshot']).read_text();after=(ROOT/row['target']).read_text()
        extract=lambda text:text.split('CUDACompiledGraph::CUDACompiledGraph() {',1)[1].split('CUDACompiledGraph::~CUDACompiledGraph()',1)[0]
        self.assertEqual(extract(before),extract(after));self.assertIn('garnet-cuda-graph-',extract(after));self.assertIn('++next',extract(after))
        lowering=(ROOT/'src/backends/tensorrt/modelopt_fp8_lowering.cpp').read_text()
        self.assertIn('shape.nbDims==3?int(shape.d[0]):0,shape.nbDims==3?int(shape.d[1]):0',lowering)
        self.assertIn('if(canonicalM128)++nativeFP8M128ProjectionCount_',lowering)

if __name__=='__main__':unittest.main()
