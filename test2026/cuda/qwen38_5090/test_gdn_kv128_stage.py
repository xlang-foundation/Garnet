from pathlib import Path
import hashlib,json,unittest
from .gdn_kv128_stage_layout import check_graph,layout
from .gdn_kv128_stage_replay import compare_entries
M=Path(__file__).resolve().parent
class GDN50(unittest.TestCase):
    def stats(self,t,d):return dict(checkpoint_weight_precision='modelopt_mixed',native_nvfp4_projection_weights=2,native_fp8_projection_weights=3,external_compressed_weight_inputs=5,native_int4_g128_projection_weights=0,native_fp8_m128_canonical_projections=int(t==128),native_fp8_z128_canonical_projections=0,native_fp8_z128_reconstructed_projections=int(t==128),native_fp8_kv128_canonical_projections=0,device_output_bytes=layout(t,d)['total_words']*4,graph_capture_failures=0,graph_captures=1,graph_replays=3)
    def test_all_profiles_and_counter_faults(self):
        for t in (32,128):
            for d in (False,True):
                s=self.stats(t,d);self.assertTrue(check_graph(s,t,d))
                for key in ('native_fp8_z128_canonical_projections','native_fp8_z128_reconstructed_projections','native_fp8_kv128_canonical_projections','native_fp8_m128_canonical_projections'):
                    for bad in (True,-1,s[key]+1):
                        q=dict(s);q[key]=bad
                        with self.assertRaises(AssertionError):check_graph(q,t,d)
    def test_original_capture_and_owner_derivation(self):
        p=json.loads((M/'evidence/gdn-kv128-stage-method-derivation.json').read_bytes())
        for row in p['files']:
            raw=(M/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                self.assertEqual(raw.count(c['before'].encode()),c['count']);raw=raw.replace(c['before'].encode(),c['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
    def test_control_piece_corruption_rejected(self):
        raw=b'\0\0\x80?';h=hashlib.sha256(raw).hexdigest();entry=dict(words=1,sha256=h,parts=[dict(file='a.f32',offset=0,words=1,sha256=h)])
        self.assertEqual(compare_entries(entry,entry,lambda n:raw,lambda n:raw),1)
        with self.assertRaises(AssertionError):compare_entries(entry,entry,lambda n:raw,lambda n:b'\0\0\0@')
if __name__=='__main__':unittest.main()
