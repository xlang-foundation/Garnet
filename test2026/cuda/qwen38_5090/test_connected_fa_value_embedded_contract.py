from pathlib import Path
import ast,json,unittest
from .connected_fa_value_layout import CASES
from . import fa_value_sdk_control as oracle
class EmbeddedContract(unittest.TestCase):
    def test_constants_dependency_free_and_original_probe_math_exact(self):
        m=Path(__file__).resolve().parent;self.assertEqual(CASES,oracle.CASES);tree=ast.parse((m/'connected_fa_value_layout.py').read_bytes());self.assertFalse(any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(tree)))
        record=json.loads((m/'evidence/connected-fa-value-v2-derivation.json').read_bytes());oracle.verify_derivations(record,lambda name:(m/name).read_bytes());rows=[r for r in record['derivations'] if r['target']=='probe_connected_fa_value_v2.py'];self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['changes']),2)
        imports=[node.module for node in ast.walk(ast.parse((m/'probe_connected_fa_value_v2.py').read_bytes())) if isinstance(node,ast.ImportFrom)];self.assertIn('test2026.cuda.qwen38_5090.connected_fa_value_layout',imports);self.assertNotIn('test2026.cuda.qwen38_5090.fa_value_sdk_control',imports)
if __name__=='__main__':unittest.main()
