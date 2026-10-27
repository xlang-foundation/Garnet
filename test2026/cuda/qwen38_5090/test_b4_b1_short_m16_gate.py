import copy,json,unittest
from .test_b4_b1_exact_gate import Gate
from .b4_b1_short_m16_gate import observe,admit,route_trace,COUNTS,FACTORY_APIS
class Admission(unittest.TestCase):
    def fixture(self):
        a,b,files=Gate().fixture();inputs=[dict(prompt_ids=[1]*n) for n in (41,104,93,93)];reports=[]
        for mode,batch,rows in [('reference',1,b),('candidate',4,[a])]:
            report=dict(status='four_B1_short_oracles_collected' if mode=='reference' else 'B4_short_mixed_tail_raw_capture_needs_CPython_admission',physical_batch=batch,capacity=4096,prefill_chunk=32,decode_chunk=4,inputs=inputs,performance_admitted=False,full4k_MTP_quality_qualified=False,graph_stats={str(t):{k:(v if mode=='candidate' and t==32 else 0) for k,v in COUNTS.items()} for t in (4,32)},vision_stats={k:0 for k in COUNTS},rows=[])
            for i,capture in enumerate(rows):report['rows'].append(dict(physical_rows=[i] if batch==1 else [0,1,2,3],generated_ids=[[r+1]*8 for r in ([i] if batch==1 else range(4))],**{phase:copy.deepcopy(capture) for phase in ('reset_state','prefill_state','final_state','final_reset')}))
            reports.append(report)
        return *reports,files
    def test_original_all_phase_words_and_ID_gates_preserved(self):
        r,c,f=self.fixture();self.assertEqual(admit(r,c,f.__getitem__,f.__getitem__)['status'],'B4_M16_short_complete_state_words_and_IDs_exact')
        c['rows'][0]['final_state']['logical']['components'][127]['row_sha256'][3]='f'*64;self.assertFalse(observe(r,c,f.__getitem__,f.__getitem__)['all_states_and_vectors_exact'])
        with self.assertRaises(AssertionError):admit(r,c,f.__getitem__,f.__getitem__)
        r,c,f=self.fixture();c['rows'][0]['generated_ids'][3][7]=9
        with self.assertRaises(AssertionError):admit(r,c,f.__getitem__,f.__getitem__)
    def test_wrong_geometry_count_or_incomplete_capture_rejected(self):
        for field in ('capacity','physical_batch','status'):
            r,c,f=self.fixture();c[field]=0
            with self.assertRaises(AssertionError):observe(r,c,f.__getitem__,f.__getitem__)
        r,c,f=self.fixture();c['graph_stats']['32']['native_fp8_m128_canonical_projections']=0
        with self.assertRaises(AssertionError):observe(r,c,f.__getitem__,f.__getitem__)
    def test_live_configuration_workspace_and_route_trace_guards(self):
        lines=[]
        for context in (1,2):
            rows=[]
            def row(event,name,a=0,b=0,c=0,opaque=''):
                rows.append(dict(id=context,seq=len(rows)+1,event=event,name=name,a=a,b=b,c=c,opaque=opaque))
            row('begin','factory')
            for api in FACTORY_APIS:row('api',api,130400 if api=='cublasLtGetVersion' else 0)
            row('candidate','heuristic',0,0,100,'ab'*64)
            for name,value in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING','CUSTOM_OPTION'),(35,13,17,3,2,0,1)):row('configuration',name,value,4,value)
            row('selection','observed_M4_control',0,opaque='ab'*64);row('check','M4',0,100);row('check','M16',0,100);row('end','created')
            lines.extend('GARNET_GDNM16_FACTORY '+json.dumps(r) for r in rows)
        for i in range(96):
            for phase in ('begin','end'):lines.append('GARNET_MODELOPT_INIT '+json.dumps(dict(id=i+1,phase=phase,route='fp8_gdn_m16',instance=str(i),rows=10240 if i<48 else 6144,columns=5120,policy=16,context_present=phase=='end',factory_called=phase=='end',result=0 if phase=='end' else -1,exception=False)))
        raw='\n'.join(lines);self.assertEqual(route_trace(raw)['created_contexts'],2)
        for bad in (raw.replace('"a": 35','"a": 67',1),raw.replace('"name": "M16", "a": 0, "b": 100','"name": "M16", "a": 0, "b": 4194305',1),raw.replace('"policy": 16','"policy": 4',1)):
            with self.assertRaises(AssertionError):route_trace(bad)
if __name__=='__main__':unittest.main()
