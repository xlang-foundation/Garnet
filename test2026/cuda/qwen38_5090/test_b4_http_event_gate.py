import copy,json,unittest
from .b4_http_event_gate import events,match_native,dispatches
from .test_b4_http_protocol import body

class HTTPEventGateControls(unittest.TestCase):
    def fixture(self):
        key='1'*32;b=body(2,True);r=dict(event='ready',batch=4)
        output=dict(event='completed',id=key,prompt_ids=[1],generated_ids=[2,3],finish_reason='length',ignore_eos=True,image_sha256=None,image_grid=None,
                    physical_batch=4,real_rows=1,padded_rows=3,generation_mode='verified_mtp',padding_policy='duplicate_first_real_row_no_public_dummy_events',native_elapsed_seconds=1.,speculative_metrics=dict(iterations=[{}]))
        stream=[r]+[dict(event='token',id=key,token_id=t,sampled_tokens=i+1,native_monotonic_seconds=float(i)) for i,t in enumerate([2,3])]+[output]
        d=dict(request_ids=[key],physical_capacity=4,real_rows=1,grouping_clock='perf_counter',grouping_started_perf_counter=1.,grouping_deadline_perf_counter=1.01,dispatch_perf_counter=1.02,request_created_perf_counter={key:.9},queue_put_started_perf_counter={key:.95})
        return key,b,r,output,stream,d
    def raw(self,values):return b'\n'.join(json.dumps(dict(line=json.dumps(v))).encode() for v in values)
    def test_actual_ID_chain_and_dispatch_fixture(self):
        key,b,r,o,s,d=self.fixture();got=events(self.raw(s),r);match_native(got[key],b,[1],[2,3],'length');self.assertEqual(dispatches(json.dumps(d).encode(),{key:b},got)['real_rows'],[1])
    def test_missing_token_duplicate_completion_and_dummy_ID_rejected(self):
        key,b,r,o,s,d=self.fixture()
        for bad in (s[:1]+s[2:],s+[o],s+[dict(s[1],id='2'*32)]):
            with self.assertRaises((AssertionError,KeyError)):events(self.raw(bad),r)
    def test_incorrect_padding_nonfinite_clock_and_wrong_ready_rejected(self):
        key,b,r,o,s,d=self.fixture()
        for attr,value in [('padded_rows',0),('native_elapsed_seconds',float('nan')),('real_rows',5)]:
            bad=copy.deepcopy(s);bad[-1][attr]=value
            with self.assertRaises(AssertionError):events(self.raw(bad),r)
        with self.assertRaises(AssertionError):events(self.raw(s),dict(r,batch=1))
    def test_response_prompt_output_stop_and_image_drift_rejected(self):
        key,b,r,o,s,d=self.fixture()
        for attr,value in [('prompt_ids',[8]),('generated_ids',[8]),('finish_reason','stop'),('ignore_eos',False),('image_sha256','0'*64)]:
            bad=dict(o);bad[attr]=value
            with self.assertRaises(AssertionError):match_native(bad,b,[1],[2,3],'length')
    def test_dispatch_omission_duplicate_and_wrong_real_rows_rejected(self):
        key,b,r,o,s,d=self.fixture()
        for raw in (b'',json.dumps(d).encode()+b'\n'+json.dumps(d).encode(),json.dumps(dict(d,real_rows=2)).encode()):
            with self.assertRaises(AssertionError):dispatches(raw,{key:b},{key:o})
    def test_large_joined_frame_reconstructed_from_real_requests(self):
        key,b,r,o,s,d=self.fixture();keys=[str(i)*32 for i in range(1,5)];b['study_transport_padding']='x'*600000
        d.update(request_ids=keys,real_rows=4,request_created_perf_counter={k:.9 for k in keys},queue_put_started_perf_counter={k:.95 for k in keys})
        outputs={k:dict(o,id=k,real_rows=4,padded_rows=0) for k in keys}
        got=dispatches(json.dumps(d).encode(),{k:b for k in keys},outputs);self.assertEqual(got['joined_frames_over_default_2MiB'],1);self.assertLess(got['max_joined_frame_bytes'],16*1024*1024)
