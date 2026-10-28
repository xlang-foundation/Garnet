import copy,hashlib,struct,unittest
from . import test_b4_full4k_decode_prefix as fixtures
from .b4_full4k_selection52 import observe,same_profile_full_parent,same_profile_step7,first_mixed_difference,STATUS,STEPS

class Selection52(unittest.TestCase):
    def fixture(self):
        s,r,c,f,fullr,fullc=fixtures.Prefix().fixture();earlyr,earlyc=copy.deepcopy(r),copy.deepcopy(c);key=s['groups'][2]['keys'][2]
        fullc['rows'][2]['generated_ids'][2][52]+=1
        r['rows']=[row for row in r['rows'] if row['keys']==[key]]
        for report,parent in ((r,fullr),(c,fullc)):
            report['status']=STATUS;old={tuple(row['keys']):row for row in parent['rows']}
            for row in report['rows']:
                row['generated_ids']=[v[:53] for v in old[tuple(row['keys'])]['generated_ids']]
                template=copy.deepcopy(row['post_step_states'][6]['capture']);row['post_step_states']=[]
                for step in STEPS:
                    capture=copy.deepcopy(template);capture['logical']['slots']=[3968+step]*report['physical_batch'];row['post_step_states'].append(dict(step=step,capture=capture))
                row['final_state']=copy.deepcopy(row['post_step_states'][-1]['capture'])
        return s,key,r,c,f,fullr,fullc,earlyr,earlyc
    def run_gate(self,s,key,r,c,f,fr,fc,er,ec):
        for a,b,d in ((r,fr,er),(c,fc,ec)):
            same_profile_full_parent(a,b,lambda n:f[n],lambda n:f[n]);same_profile_step7(a,d,lambda n:f[n],lambda n:f[n])
        return observe(s,key,r,c,lambda k,n:f[n],lambda n:f[n])
    def test_exact_state_with_different_unconsumed_ID_is_selection_path_only(self):
        args=self.fixture();s,key,r,c,f,fr,fc,er,ec=args;choice=first_mixed_difference(s,fr,fc);self.assertEqual(choice['selected_key'],key);self.assertEqual(choice['physical_row'],2)
        q=self.run_gate(*args);self.assertTrue(q['all52_consumed_IDs_exact']);self.assertTrue(q['token_selection_path_differs_with_exact_decoder_state']);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
    def test_component127_or_full_raw_vector_drift_is_not_relabelled_head_only(self):
        for mutation in ('component','vector'):
            args=self.fixture();s,key,r,c,f,*_=args;capture=c['rows'][0]['post_step_states'][1]['capture']
            if mutation=='component':capture['logical']['components'][127]['row_sha256'][2]='f'*64
            else:entry=capture['vectors']['hidden'];raw=bytearray(f[entry['file']]);raw[2*20480:2*20480+4]=struct.pack('<f',123.0);entry['file']='only-step52.f32';f[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest()
            c['rows'][0]['final_state']=copy.deepcopy(capture);q=self.run_gate(*args);self.assertFalse(q['step52_state_hidden_normalized_exact']);self.assertFalse(q['token_selection_path_differs_with_exact_decoder_state'])
    def test_changed_consumed_ID_wrong_key_or_step_rejected(self):
        for mutation in ('ID','key','step'):
            args=self.fixture();s,key,r,c,f,fr,fc,er,ec=args
            if mutation=='ID':c['rows'][0]['generated_ids'][2][51]+=1;fc['rows'][2]['generated_ids'][2][51]+=1
            elif mutation=='key':r['rows'][0]['keys']=[s['groups'][2]['keys'][0]]
            else:c['rows'][0]['post_step_states'][1]['step']=51
            with self.assertRaises((AssertionError,KeyError)):self.run_gate(*args)
    def test_step7_other_row_drift_parent_ID_drift_and_nan_rejected(self):
        for mutation in ('step7','parent','NaN'):
            args=self.fixture();s,key,r,c,f,fr,fc,er,ec=args
            if mutation=='step7':c['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][0]='f'*64
            elif mutation=='parent':c['rows'][0]['generated_ids'][2][52]+=1
            else:capture=c['rows'][0]['post_step_states'][1]['capture'];entry=capture['vectors']['hidden'];raw=bytearray(f[entry['file']]);raw[-4:]=struct.pack('<f',float('nan'));entry['file']='only-nonfinite-step52.f32';f[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest();c['rows'][0]['final_state']=copy.deepcopy(capture)
            with self.assertRaises(AssertionError):self.run_gate(*args)
if __name__=='__main__':unittest.main()
