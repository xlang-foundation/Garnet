from pathlib import Path
import ast,hashlib,json,unittest
from .full_chunk_stage_layout import STAGES
from .bf16_model_state_witness import logical_state_witness
from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class CollectorRetryTests(unittest.TestCase):
    def test_stage_width_cannot_shadow_canonical_state_closure_and_original_guard_stays_strict(self):
        record=json.loads((M/'evidence/full-chunk-stage-collector-retry-derivation.json').read_bytes())
        for row in record['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        tree=ast.parse((M/'probe_full_chunk_stage_attempt2.py').read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');state=next(n for n in main.body if isinstance(n,ast.Try) for n in n.body if isinstance(n,ast.FunctionDef) and n.name=='state')
        call=next(n for n in ast.walk(state) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='logical_state_witness');self.assertIsInstance(call.args[4],ast.Name);self.assertEqual(call.args[4].id,'canonical_width')
        assignments=[n for n in ast.walk(main) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='canonical_width'];self.assertEqual(len(assignments),1)
        canonical_width=106823680
        for name,width in STAGES:pass
        self.assertEqual(width,5120);layout=build_bf16_low_batch_layout()['layers']
        class Validated(Exception):pass
        class IO:
            def zeros(self,shape):raise Validated('Complete canonical metadata accepted before any GPU read')
        with self.assertRaisesRegex(ValueError,'metadata does not cover canonical row'):logical_state_witness(IO(),None,layout,1,width,[3968],packed_bf16=True)
        with self.assertRaises(Validated):logical_state_witness(IO(),None,layout,1,canonical_width,[3968],packed_bf16=True)
        self.assertTrue(record['all51native_source_DLL_math_and_original_numeric_gates_unchanged'])
