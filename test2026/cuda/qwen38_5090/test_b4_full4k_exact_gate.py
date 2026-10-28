import copy,hashlib,struct,unittest
from . import test_b4_b1_exact_gate as original_fixtures
from .b4_b1_short_m16_gate import COUNTS,PHASES
from .b4_full4k_exact_gate import observe,admit
sha=lambda raw:hashlib.sha256(raw).hexdigest()
class FullContextGate(unittest.TestCase):
    def fixture(self):
        _,single,files=original_fixtures.Gate().fixture();files=dict(files);inputs={key:dict(key=key,case='text' if i<4 else 'image',prompt_ids=[i+1]*3968) for i,key in enumerate('abcdefgh')}
        groups=[dict(label='full4k-text',keys=list('abcd'),generated_IDs=128,total_context_capacity=4096),dict(label='full4k-image',keys=list('efgh'),generated_IDs=128,total_context_capacity=4096),dict(label='full4k-mixed',keys=list('aebf'),generated_IDs=128,total_context_capacity=4096)]
        selection=dict(status='retained_fixed4k_B4_correctness_inputs_selected_not_executed',inputs=inputs,groups=groups)
        def report(mode):return dict(status='eight_B1_full4k_oracles_collected' if mode=='reference' else 'B4_full4k_raw_capture_needs_CPython_admission',physical_batch=1 if mode=='reference' else 4,capacity=4096,prefill_chunk=32,decode_chunk=4,performance_admitted=False,MTP_quality_qualified=False,inputs=copy.deepcopy(inputs),graph_stats={str(t):{k:(v if mode=='candidate' and t==32 else 0) for k,v in COUNTS.items()} for t in (4,32)},vision_stats={k:0 for k in COUNTS},rows=[])
        r,c=report('reference'),report('candidate')
        for i,key in enumerate(inputs):
            row=dict(keys=[key],generated_ids=[[i+1]*128])
            for phase,slot in zip(PHASES,(0,3968,4095,0)):
                capture=copy.deepcopy(single[i%4]);capture['logical']['slots']=[slot];row[phase]=capture
            r['rows'].append(row)
        by_key={row['keys'][0]:row for row in r['rows']}
        for group in groups:
            refs=[by_key[key] for key in group['keys']];row=dict(label=group['label'],keys=list(group['keys']),generated_ids=[copy.deepcopy(x['generated_ids'][0]) for x in refs])
            for phase in PHASES:
                capture=copy.deepcopy(refs[0][phase]);logic=capture['logical']
                for field in ('slots','physical_row_sha256','logical_row_sha256'):logic[field]=[x[phase]['logical'][field][0] for x in refs]
                for i,component in enumerate(logic['components']):component['row_sha256']=[x[phase]['logical']['components'][i]['row_sha256'][0] for x in refs]
                for kind,entry in capture['vectors'].items():
                    raw=b''.join(files[x[phase]['vectors'][kind]['file']] for x in refs);name=group['label']+'-'+phase+'-'+kind+'.f32';files[name]=raw;capture['vectors'][kind]=dict(file=name,bytes=len(raw),sha256=sha(raw))
                row[phase]=capture
            c['rows'].append(row)
        return selection,r,c,files
    def run_gate(self,selection,r,c,files,method=admit):return method(selection,r,c,lambda key,name:files[name],lambda group,name:files[name])
    def test_all_eight_oracles_three_groups_four_phases_and_128_IDs(self):
        s,r,c,f=self.fixture();q=self.run_gate(s,r,c,f);self.assertEqual(q['full_phase_row_comparisons'],48);self.assertEqual(q['components_per_row'],128);self.assertTrue(q['all_128_IDs_exact']);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
    def test_last_state_component_and_last_output_ID_cannot_be_skipped(self):
        for mutation in ('component','ID'):
            s,r,c,f=self.fixture()
            if mutation=='component':c['rows'][-1]['final_state']['logical']['components'][127]['row_sha256'][3]='f'*64
            else:c['rows'][-1]['generated_ids'][3][127]+=1
            q=self.run_gate(s,r,c,f,observe);self.assertFalse(q['all_states_and_vectors_exact'] if mutation=='component' else q['all_128_IDs_exact'])
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_missing_oracle_row_permutation_short_context_or_127_outputs_rejected(self):
        for mutation in ('oracle','permutation','context','outputs'):
            s,r,c,f=self.fixture()
            if mutation=='oracle':r['rows'].pop()
            elif mutation=='permutation':c['rows'][0]['keys'].reverse()
            elif mutation=='context':s['inputs']['a']['prompt_ids'].pop()
            else:c['rows'][0]['generated_ids'][0].pop()
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_nonfinite_and_signed_zero_vectors_rejected_without_tolerance(self):
        for replacement in (struct.pack('<f',float('nan')),struct.pack('<f',-0.0)):
            s,r,c,f=self.fixture()
            for name in f:f[name]=b'\0'*len(f[name])
            for report in (r,c):
                for row in report['rows']:
                    for phase in PHASES:
                        for entry in row[phase]['vectors'].values():entry['sha256']=sha(f[entry['file']])
            self.run_gate(s,r,c,f)
            entry=c['rows'][-1]['final_state']['vectors']['hidden'];raw=bytearray(f[entry['file']]);raw[-4:]=replacement;f[entry['file']]=bytes(raw);entry['sha256']=sha(raw)
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
    def test_full_context_slot_and_reset_restoration_enforced(self):
        for mutation in ('slot','reset','profile'):
            s,r,c,f=self.fixture()
            if mutation=='slot':c['rows'][0]['final_state']['logical']['slots'][0]=4096
            elif mutation=='reset':c['rows'][0]['final_reset']['logical']['components'][127]['row_sha256'][3]='f'*64
            else:c['graph_stats']['32']['native_fp8_q128_canonical_projections']=0
            with self.assertRaises(AssertionError):self.run_gate(s,r,c,f)
if __name__=='__main__':unittest.main()
