import ast,hashlib,json,unittest
from pathlib import Path
class Controls(unittest.TestCase):
    def test_metadata_is_exact_original_body_without_numpy_and_retry_keeps_collector_math(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/fa3-projection-witness-attempt2-derivation.json').read_bytes());self.assertEqual(len(p['files']),4)
        for row in p['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                before=c['before'].encode();self.assertEqual(raw.count(before),c['count']);raw=raw.replace(before,c['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
        original=(m/p['layout_source']).read_bytes();self.assertEqual(hashlib.sha256(original).hexdigest(),p['layout_source_sha256']);body=original[original.index(p['layout_body_start'].encode()):original.index(p['layout_body_end'].encode())]
        layout=(m/p['layout_target']).read_bytes();self.assertEqual(hashlib.sha256(layout).hexdigest(),p['layout_target_sha256']);self.assertEqual(layout[layout.index(b'STAGES='):],body)
        tree=ast.parse(layout);self.assertTrue(all(alias.name in ('hashlib','math') for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names));self.assertNotIn(b'numpy',layout)
