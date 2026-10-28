"""Synthetic counter/transport controls only; never execute model or GPU code."""
import copy,json,unittest
from .b4_resident_serial_graph_gate import admit,NAMES

class SerialGraphEvidenceControls(unittest.TestCase):
    def event(self,key='1'*32,value=2):
        counter=dict(contexts=1,graph_captures=1,graph_replays=value,graph_capture_failures=0,serial_graph_reuses=value,pending_graph_reuses=1,serial_graph_warmup_waits=1)
        return dict(event='study_graph_snapshot',request_ids=[key],real_rows=1,physical_batch=4,CUDA_kernel_elapsed_claimed=False,clean_throughput_result_claimed=False,
                    graphs={name:dict(counter,cuda_graph_requested=True,serial_graph_reuse_requested=True) for name in NAMES},io=dict(embedding_mode='host_mapped_bf16',head_precision='modelopt_nvfp4_w4a4'))
    def raw(self,values):return b'\n'.join(json.dumps(dict(line=json.dumps(v))).encode() for v in values)
    def test_disjoint_groups_and_cumulative_actual_reuse(self):
        values=[self.event(),self.event('2'*32,3)];r=admit(self.raw(values),{'1'*32,'2'*32});self.assertEqual((r['snapshot_groups'],r['real_control_requests']),(2,2));self.assertFalse(r['CUDA_kernel_time_or_clean_throughput_qualified'])
    def test_requested_flags_without_reuse_are_insufficient(self):
        value=self.event();value['graphs']['target32'].update(graph_replays=0,serial_graph_reuses=0,pending_graph_reuses=0)
        with self.assertRaises(AssertionError):admit(self.raw([value]),{'1'*32})
    def test_duplicate_omitted_or_unknown_real_IDs_rejected(self):
        value=self.event()
        for values,expected in (([value,value],{'1'*32}),([value],{'1'*32,'2'*32}),([self.event('2'*32)],{'1'*32})):
            with self.assertRaises(AssertionError):admit(self.raw(values),expected)
    def test_capture_failure_disabled_policy_or_wrong_precision_rejected(self):
        for field,v in (('graph_capture_failures',1),('serial_graph_reuse_requested',False),('cuda_graph_requested',False)):
            value=self.event();value['graphs']['draft16'][field]=v
            with self.assertRaises(AssertionError):admit(self.raw([value]),{'1'*32})
        value=self.event();value['io']['head_precision']='bf16'
        with self.assertRaises(AssertionError):admit(self.raw([value]),{'1'*32})
    def test_decreasing_or_inconsistent_counters_rejected(self):
        values=[self.event(),self.event('2'*32,1)]
        with self.assertRaises(AssertionError):admit(self.raw(values),{'1'*32,'2'*32})
        value=self.event();value['graphs']['target4']['pending_graph_reuses']=3
        with self.assertRaises(AssertionError):admit(self.raw([value]),{'1'*32})
    def test_boolean_negative_counter_or_false_timing_claim_rejected(self):
        for mutation in ('bool','negative','claim'):
            value=self.event()
            if mutation=='bool':value['graphs']['target4']['contexts']=True
            elif mutation=='negative':value['graphs']['target4']['contexts']=-1
            else:value['CUDA_kernel_elapsed_claimed']=True
            with self.assertRaises(AssertionError):admit(self.raw([value]),{'1'*32})
