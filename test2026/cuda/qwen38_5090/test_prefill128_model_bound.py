"""Reproduce the actual missing-provenance preflight failure without GPU use."""
from pathlib import Path
import hashlib,json,unittest
from .prepare_bf16_prefill128_model_bound import required_provenance,assert_provenance_closure

class DependencyControls(unittest.TestCase):
    def test_actual_failed_preparation_omits_five_component_provenance_sources(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/bf16-prefill128-model-preparation.json').read_bytes())
        names={key.removeprefix('Garnet/test2026/cuda/qwen38_5090/') for key in p['new_files']}
        missing=required_provenance(m)-names
        self.assertEqual(missing,{'qwen38_bf16_prefill128_attention.cu','qwen38_bf16_prefill128_attention.cuh','qwen38_bf16_prefill128_commit.cu','qwen38_bf16_prefill128_commit.cuh','qwen38_bf16_prefill128_layout_contract.h'})
        with self.assertRaises(AssertionError):assert_provenance_closure(m,names)

    def test_each_declared_provenance_dependency_is_required(self):
        m=Path(__file__).resolve().parent;names=required_provenance(m);assert_provenance_closure(m,names)
        for name in names:
            with self.subTest(name=name),self.assertRaises(AssertionError):assert_provenance_closure(m,names-{name})

    def test_retry_is_exact_recorded_method_transform_without_model_source_change(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();record=json.loads((m/'evidence/prefill128-model-bound-source-derivation.json').read_bytes())
        for row in record['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])
        original=(m/'probe_bf16_prefill128_model.py').read_bytes();bound=(m/'probe_bf16_prefill128_model_bound.py').read_bytes()
        self.assertEqual(original.replace(b'bf16-prefill128-model-preparation.json',b'bf16-prefill128-model-bound-preparation.json'),bound)

if __name__=='__main__':unittest.main()
