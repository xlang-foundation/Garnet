import copy,hashlib,struct,unittest
from . import test_b4_full4k_decode_prefix as fixtures
from .all_layer_full4k_decode_analysis import analyze,ROW_BYTES
sha=lambda raw:hashlib.sha256(raw).hexdigest()

class Analysis(unittest.TestCase):
    def fixture(self):
        s,r,c,f,_,_=fixtures.Prefix().fixture()
        for mode,report in (('reference',r),('candidate',c)):report['mode']=mode;report['library_sha256']='a'*64
        pr,pc=copy.deepcopy(r),copy.deepcopy(c)
        for mode,report in (('reference',r),('candidate',c)):
            report['observer']=dict(layers=list(range(64)),tokens=4,hidden=5120,hidden_width=20480,primary_width=17,wide_width=17+64*20480,selected_steps=[1])
            report['boundary_runs']=len(report['rows'])*7
            for group,row in enumerate(report['rows']):
                raw=b''.join(struct.pack('<f',float(ord(key)-ord('a')))*(ROW_BYTES//4) for key in row['keys']);boundaries={}
                for layer in range(64):
                    name=mode+'-group'+str(group)+'-layer'+str(layer)+'.f32';f[name]=raw;boundaries[str(layer)]=dict(file=name,bytes=len(raw),sha256=sha(raw))
                row['layer_observations']=[dict(step=1,program_runs=group*7+1,consumed_tokens=[ids[0] for ids in row['generated_ids']],boundaries=boundaries)]
        return s,r,c,pr,pc,f
    def run_gate(self,s,r,c,pr,pc,f):return analyze(s,r,c,pr,pc,lambda k,n:f[n],lambda n:f[n],lambda k,n:f[n],lambda n:f[n])
    def test_all64_and_allseven_primary_snapshots_enforced(self):
        s,r,c,pr,pc,f=self.fixture();q=self.run_gate(s,r,c,pr,pc,f);self.assertEqual(len(q['rows']),4)
        self.assertTrue(all(row['first_differing_active_layer'] is None for row in q['rows']));self.assertEqual(q['primary_parity']['reference']['snapshots'],28);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
    def test_last_layer_last_active_word_and_inactive_word_are_distinguished(self):
        for word in (5119,20479):
            s,r,c,pr,pc,f=self.fixture();entry=c['rows'][0]['layer_observations'][0]['boundaries']['63'];raw=bytearray(f[entry['file']]);offset=3*ROW_BYTES+word*4;raw[offset:offset+4]=struct.pack('<f',123.0);f[entry['file']]=bytes(raw);entry['sha256']=sha(raw)
            row=self.run_gate(s,r,c,pr,pc,f)['rows'][3];self.assertEqual(row['first_differing_full_T4_layer'],63);self.assertEqual(row['first_differing_active_layer'],63 if word==5119 else None)
    def test_observer_primary_state_or_vector_change_rejected_before_layer_interpretation(self):
        for mutation in ('component','vector'):
            s,r,c,pr,pc,f=self.fixture();capture=c['rows'][0]['post_step_states'][6]['capture']
            if mutation=='component':capture['logical']['components'][127]['row_sha256'][3]='f'*64
            else:entry=capture['vectors']['hidden'];raw=bytearray(f[entry['file']]);raw[-4:]=struct.pack('<f',123.0);entry['file']='observer-only-mutated.f32';f[entry['file']]=bytes(raw);entry['sha256']=sha(raw)
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,pr,pc,f)
    def test_missing_layer_nan_or_bad_step_rejected(self):
        for mutation in ('layer','NaN','step'):
            s,r,c,pr,pc,f=self.fixture();entry=c['rows'][0]['layer_observations'][0]
            if mutation=='layer':entry['boundaries'].pop('63')
            elif mutation=='step':c['observer']['selected_steps']=[8]
            else:e=entry['boundaries']['63'];raw=bytearray(f[e['file']]);raw[-4:]=struct.pack('<f',float('nan'));f[e['file']]=bytes(raw);e['sha256']=sha(raw)
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,pr,pc,f)
if __name__=='__main__':unittest.main()
