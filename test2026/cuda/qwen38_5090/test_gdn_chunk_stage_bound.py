"""CPU actual readback retry/source provenance controls; no native runtime/GPU."""
from pathlib import Path
import ast,hashlib,json,tempfile,unittest

class Controls(unittest.TestCase):
    def test_actual_failed_probe_readback_replaced_by_explicit_loop(self):
        m=Path(__file__).resolve().parent;tree=ast.parse((m/'probe_gdn_chunk_stage_witness_bound.py').read_bytes());main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main');raw=next(node for node in main.body if isinstance(node,ast.FunctionDef) and node.name=='raw')
        self.assertFalse(any(isinstance(node,ast.GeneratorExp) for node in ast.walk(raw)))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'a.f32').write_bytes(b'abcd');(out/'b.f32').write_bytes(b'efgh');ns={'out':out};exec(compile(ast.Module(body=[raw],type_ignores=[]),'bounded-readback','exec'),ns)
            self.assertEqual(ns['raw']({'parts':[{'file':'a.f32'},{'file':'b.f32'}]}),b'abcdefgh')

    def test_all_retry_methods_are_exact_recorded_derivatives(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();record=json.loads((m/'evidence/gdn-chunk-stage-witness-bound-derivation.json').read_bytes())
        for row in record['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
