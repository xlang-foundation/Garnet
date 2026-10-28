from pathlib import Path
import copy,hashlib,json,unittest
from .kv128_prefix_cache_witness import check_graphs,OLD,RECON,KV
M=Path(__file__).resolve().parent
class Prefix(unittest.TestCase):
    def graphs(self,tokens):return {str(t):{'native_fp8_m128_canonical_projections':48 if t==128 else 0,OLD:0,RECON:48 if t==128 else 0,KV:32 if t==128 else 0} for t in (4,tokens)}
    def test_physical_routes_and_unchanged_logical_Z_check(self):
        for t in (32,128):
            g=self.graphs(t);saved=copy.deepcopy(g);self.assertTrue(check_graphs(g,t));self.assertEqual(g,saved)
            for chunk in g:
                for field in ('native_fp8_m128_canonical_projections',OLD,RECON,KV):
                    for value in (True,-1,g[chunk][field]+1):
                        bad=copy.deepcopy(g);bad[chunk][field]=value
                        with self.assertRaises(AssertionError):check_graphs(bad,t)
        bad=self.graphs(128);bad['128'][OLD]=1;bad['128'][RECON]=47
        with self.assertRaises(AssertionError):check_graphs(bad,128)
    def test_exact_capture_owner_and_batch_derivations(self):
        p=json.loads((M/'evidence/kv128-prefix-cache-method-derivation.json').read_bytes())
        for row in p['files']:
            raw=(M/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                self.assertEqual(raw.count(c['before'].encode()),c['count']);raw=raw.replace(c['before'].encode(),c['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
    def test_both_profiles_use_original_T32_oracle(self):
        raw=(M/'probe_kv128_prefix_cache.py').read_bytes()
        self.assertIn(b"parent='mtp-prefill16-full4k-admission-attempt1'",raw)
        self.assertNotIn(b"else 'fp8-z128-full4k-model-admission-attempt1'",raw)
        self.assertIn(b"expected=old['cases'][0]",raw)
        for guard in (b"same_state(report['ordinary_prefill']",b"ordinary==expected['ordinary_ids']",b"same_state(report['ordinary_final']"):
            self.assertIn(guard,raw)
    def test_complete_state_diagnostic_keeps_first_component(self):
        from .test_prefill128_model import data
        from .kv128_prefix_cache_replay import diagnose_state
        actual,reference,files,_=data();a=copy.deepcopy(actual['ordinary_observations'][0]['ordinary_prefill']);b=reference['cases'][0]['ordinary_prefill']
        a['logical']['components'][7]['row_sha256']=['f'*64];a['logical']['physical_row_sha256']=['d'*64];a['logical']['logical_row_sha256']=['e'*64]
        result=diagnose_state(a,b,files.__getitem__,files.__getitem__)
        self.assertFalse(result['physical_state_exact']);self.assertFalse(result['logical_state_exact']);self.assertEqual(result['first_different_component']['index'],7)
if __name__=='__main__':unittest.main()
