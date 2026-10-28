from pathlib import Path
import copy,hashlib,json,unittest
from .fa3_projection_vendor import compare
from .test_fa3_projection_witness import fixture
class Controls(unittest.TestCase):
    def test_exact_vendor_only_derivation_and_complete_byte_parity_rejects_value_drift(self):
        m=Path(__file__).resolve().parent;p=json.loads((m/'evidence/fa3-projection-vendor-derivation.json').read_bytes());self.assertEqual(len(p['files']),4)
        for row in p['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                before=c['before'].encode();self.assertEqual(raw.count(before),c['count']);raw=raw.replace(before,c['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
        r,f,c,store=fixture();self.assertEqual(compare(r,r,f.__getitem__,f.__getitem__,c)['complete_finite_array_entries_compared'],51)
        bad=copy.deepcopy(r);row=bad['cases'][1];e=row['repeats'][0][0]['arrays']['value'];raw=bytearray(f[e['parts'][0]['file']]);raw[0]^=1;new=store(bytes(raw))
        for seq in row['repeats']:seq[0]['arrays']['value']=new
        with self.assertRaises(AssertionError):compare(bad,r,f.__getitem__,f.__getitem__,c)
