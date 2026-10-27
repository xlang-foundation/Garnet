import copy,unittest
from . import test_b4_full4k_exact_gate as fixtures
from .b4_full4k_decode_prefix import observe,same_profile_parent,STATUS

class Prefix(unittest.TestCase):
    def fixture(self):
        s,r,c,f=fixtures.FullContextGate().fixture();oldr,oldc=copy.deepcopy(r),copy.deepcopy(c)
        keys=s['groups'][2]['keys'];by_key={row['keys'][0]:row for row in r['rows']}
        r['rows']=[by_key[k] for k in keys];c['rows']=[c['rows'][2]]
        for report in (r,c):
            report['status']=STATUS
            for row in report['rows']:
                row['generated_ids']=[v[:8] for v in row['generated_ids']]
                row['post_step_states']=[]
                for step in range(1,8):
                    capture=copy.deepcopy(row['final_state']);capture['logical']['slots']=[3968+step]*report['physical_batch']
                    row['post_step_states'].append(dict(step=step,capture=capture))
                row['final_state']=copy.deepcopy(row['post_step_states'][-1]['capture'])
        return s,r,c,f,oldr,oldc
    def run_gate(self,s,r,c,f):return observe(s,r,c,lambda k,n:f[n],lambda n:f[n])
    def test_all_128_components_and_all_seven_steps_observed(self):
        s,r,c,f,pr,pc=self.fixture();q=self.run_gate(s,r,c,f)
        self.assertIsNone(q['first_nonexact_step']);self.assertTrue(q['original_full4k_gate_still_failed']);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
        for a,b in ((r,pr),(c,pc)):same_profile_parent(a,b,lambda n:f[n],lambda n:f[n])
    def test_first_drift_component_127_at_step_three_retained(self):
        s,r,c,f,_,_=self.fixture();c['rows'][0]['post_step_states'][2]['capture']['logical']['components'][127]['row_sha256'][3]='f'*64
        q=self.run_gate(s,r,c,f);self.assertEqual(q['first_nonexact_step'],3);self.assertEqual(q['steps'][2]['comparison']['rows'][3]['different_component_indices'],[127])
    def test_missing_step_wrong_slot_last_ID_or_permutation_rejected(self):
        for mutation in ('step','slot','ID','keys'):
            s,r,c,f,_,_=self.fixture();row=c['rows'][0]
            if mutation=='step':row['post_step_states'].pop()
            elif mutation=='slot':row['post_step_states'][0]['capture']['logical']['slots'][0]=3968
            elif mutation=='ID':row['generated_ids'][3][7]+=1
            else:row['keys'].reverse()
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_parent_endpoint_or_first_ID_change_rejected(self):
        for mutation in ('state','ID','raw'):
            _,r,_,f,pr,_=self.fixture()
            if mutation=='state':r['rows'][0]['prefill_state']['logical']['components'][0]['row_sha256'][0]='f'*64
            elif mutation=='ID':r['rows'][0]['generated_ids'][0][0]+=1
            else:entry=r['rows'][0]['prefill_state']['vectors']['hidden'];entry['sha256']='f'*64
            with self.assertRaises(AssertionError):same_profile_parent(r,pr,lambda n:f[n],lambda n:f[n])
if __name__=='__main__':unittest.main()
