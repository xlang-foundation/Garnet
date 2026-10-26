"""CPU controls for the exact workspace-only change and full native-source declaration."""
from pathlib import Path
import hashlib,json,tempfile,unittest
from .run_prefill128_fp8_workspace8 import primitive_source_hashes,PRIMITIVES

class Controls(unittest.TestCase):
    def test_exact_recorded_workspace_only_and_method_source_transforms(self):
        m=Path(__file__).resolve().parent;record=json.loads((m/'evidence/prefill128-fp8-workspace8-derivation.json').read_bytes());sha=lambda raw:hashlib.sha256(raw).hexdigest()
        for row in record['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        self.assertEqual(len(record['files'][0]['replacements']),2)
        source=(m/'actual_prefill128_fp8_workspace8.cu').read_text();self.assertIn('sameAlgorithm && m == 128 ? (8ULL << 20) : (4ULL << 20)',source)

    def test_native_cuda_sources_cannot_be_omitted_from_preparation(self):
        expected={'src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh'}
        self.assertEqual(set(PRIMITIVES),expected)
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)
            for name in expected:
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
            result=primitive_source_hashes(repo)
            self.assertEqual(result,{'Garnet/'+name:hashlib.sha256(name.encode()).hexdigest() for name in expected})
            (repo/'src/cuda/modelopt_fp8.cu').unlink()
            with self.assertRaises(FileNotFoundError):primitive_source_hashes(repo)
        m=Path(__file__).resolve().parent;self.assertIn("**primitive_source_hashes(ROOT/'Garnet')",(m/'run_prefill128_fp8_workspace8.py').read_text())

if __name__=='__main__':unittest.main()
