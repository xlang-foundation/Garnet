import copy,unittest
from . import test_b4_full4k_exact_gate as full
from .b4_mtp_full4k_gate import admit
class MTPFullContext(unittest.TestCase):
    def fixture(self):
        s,r,c,f=full.FullContextGate().fixture();c['mtp_study_class']='CUDAMTPPrefill16B4StudySession';c['mtp_memory']=dict(physical_batch=4,target_prefill_chunk=32,draft_prefill_chunk=16,draft_decode_chunk=4,draft_decode_builder_workspace_bytes=256<<20,draft_prefill_builder_workspace_bytes=128<<20,extra_draft_engine_weights_duplicated=True)
        iterations=[dict(processed_target_counts=[4]*4,sampled_so_far=[4*(i+1)]*4) for i in range(31)]+[dict(processed_target_counts=[3]*4,sampled_so_far=[128]*4)]
        for row in c['rows']:row['speculative_metrics']=dict(target_verification_calls=32,proposed_draft_tokens=300,accepted_draft_tokens=100,mtp_prefill_calls=249,target_prefix_recomputes=0,target_prefix_replays=1,iterations=copy.deepcopy(iterations))
        return s,r,c,f
    def run_gate(self,s,r,c,f):return admit(s,r,c,lambda key,name:f[name],lambda label,name:f[name])
    def test_actual_MTP_counts_and_unchanged_exact_target_gate(self):
        s,r,c,f=self.fixture();q=self.run_gate(s,r,c,f);self.assertEqual(q['actual_MTP_routes'][0]['committed_target_tokens'],[127]*4);self.assertFalse(q['short_prefix_permutation_tail_repeat_quality_HTTP_or_performance_qualified'])
    def test_greedy_or_wrong_workspace_geometry_cannot_be_labeled_MTP(self):
        for change in ('class','workspace','batch','prefill'):
            s,r,c,f=self.fixture()
            if change=='class':c['mtp_study_class']='greedy'
            elif change=='workspace':c['mtp_memory']['draft_prefill_builder_workspace_bytes']=256<<20
            elif change=='batch':c['mtp_memory']['physical_batch']=1
            else:c['mtp_memory']['draft_prefill_chunk']=4
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_missing_proposals_verification_or_shifted_prefill_and_prefix_counts_rejected(self):
        for change in ('proposed_draft_tokens','target_verification_calls','mtp_prefill_calls','target_prefix_recomputes','counts','sampled'):
            s,r,c,f=self.fixture();m=c['rows'][0]['speculative_metrics']
            if change=='counts':m['iterations'][-1]['processed_target_counts'][0]=2
            elif change=='sampled':m['iterations'][-1]['sampled_so_far'][0]=True
            else:m[change]=1 if change=='target_prefix_recomputes' else 0
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_MTP_metrics_do_not_waive_last_state_component_or_last_ID(self):
        for kind in ('state','ID'):
            s,r,c,f=self.fixture()
            if kind=='state':c['rows'][-1]['final_state']['logical']['components'][127]['row_sha256'][3]='f'*64
            else:c['rows'][-1]['generated_ids'][3][127]+=1
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
if __name__=='__main__':unittest.main()
