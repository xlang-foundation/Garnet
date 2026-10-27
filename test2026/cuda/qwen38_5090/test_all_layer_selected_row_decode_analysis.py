import copy,hashlib,struct,unittest
from . import test_all_layer_full4k_decode_analysis as fixtures
from .all_layer_selected_row_decode_analysis import analyze_selected_row
from .all_layer_full4k_decode_analysis import ROW_BYTES

class Selected(unittest.TestCase):
    def fixture(self):
        s,r,c,pr,pc,f=fixtures.Analysis().fixture();key=s['groups'][2]['keys'][3]
        for report in (pc,c):report['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][3]='f'*64
        r['rows']=[row for row in r['rows'] if row['keys']==[key]];r['boundary_runs']=7;r['rows'][0]['layer_observations'][0]['program_runs']=1
        return s,key,r,c,pr,pc,f
    def run_gate(self,s,key,r,c,pr,pc,f):return analyze_selected_row(s,key,r,c,pr,pc,lambda k,n:f[n],lambda n:f[n],lambda k,n:f[n],lambda n:f[n])
    def test_selected_affected_row_with_all_four_candidate_primary_rows(self):
        args=self.fixture();s,key,r,c,pr,pc,f=args;q=self.run_gate(*args)
        self.assertEqual(q['selected_key'],key);self.assertEqual(q['physical_row'],3);self.assertEqual(q['primary_parity']['reference']['snapshots'],7);self.assertEqual(q['primary_parity']['candidate']['snapshots'],7)
        self.assertIsNone(q['first_differing_active_layer']);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
        entry=c['rows'][0]['layer_observations'][0]['boundaries']['63'];raw=bytearray(f[entry['file']]);raw[3*ROW_BYTES+5119*4:3*ROW_BYTES+5120*4]=struct.pack('<f',123.0);f[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.run_gate(*args)['first_differing_active_layer'],63)
    def test_unaffected_row_extra_oracle_or_other_B4_primary_drift_rejected(self):
        for mutation in ('unaffected','extra','other_primary'):
            args=self.fixture();s,key,r,c,pr,pc,f=args
            if mutation=='unaffected':
                for report in (pc,c):report['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][3]=pr['rows'][3]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][0]
            elif mutation=='extra':r['rows'].append(copy.deepcopy(r['rows'][0]))
            else:c['rows'][0]['post_step_states'][6]['capture']['logical']['components'][0]['row_sha256'][0]='f'*64
            with self.assertRaises(AssertionError):self.run_gate(*args)
if __name__=='__main__':unittest.main()
