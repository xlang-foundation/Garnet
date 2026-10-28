import copy,hashlib,struct,unittest
from . import test_b4_b1_short_m16_gate as old
from .decode_boundary_analysis import STATUS,primary_parity,validate_observations,analyze,blob,difference
from .decode_boundary_layout import LAYERS,STEPS
class Analysis(unittest.TestCase):
    def fixture(self):
        r,c,files=old.Admission().fixture();parents=[]
        def entry(raw,name):
            files[name]=raw;return dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        for mode,report in (('reference',r),('candidate',c)):
            batch=report['physical_batch'];report.update(mode=mode,library_sha256=('1' if batch==1 else '2')*64,pool=dict(embedding_mode='host_mapped_bf16',head_precision='modelopt_nvfp4_w4a4',compressed_projection_pool_entries=336,compressed_nvfp4_entries=128,compressed_fp8_entries=208))
            for row in report['rows']:row['prompt_cache']=dict(page_size=128)
            parents.append(copy.deepcopy(report));report['status']=STATUS;report['observer']=dict(layers=list(LAYERS),tokens=4,hidden=5120,primary_width=17,wide_width=17+8*20480,hidden_width=20480);report['boundary_runs']=len(report['rows'])*STEPS;report['normalizer_stats']={str(k):dict(**{key:0 for key in report['vision_stats']},native_fp8_projection_weights=0,native_nvfp4_projection_weights=0) for k in (3,11)}
            for group,row in enumerate(report['rows']):
                row['decode_observations']=[]
                for step in range(1,STEPS+1):
                    stem=mode+'-g'+str(group)+'-s'+str(step);e=dict(step=step,physical_rows=row['physical_rows'][:],consumed_tokens=[v[step-1] for v in row['generated_ids']],program_runs=group*STEPS+step,boundaries={},normalizations={},value_cache={});row['decode_observations'].append(e)
                    for layer in LAYERS:
                        raw=b''.join(struct.pack('<f',float(step+layer+i))*5120+b'\0'*(3*5120*4) for i in row['physical_rows']);e['boundaries'][str(layer)]=entry(raw,stem+'-l'+str(layer)+'.f32')
                    for layer in (3,11):
                        raw=files[e['boundaries'][str(layer-1)]['file']];e['normalizations'][str(layer)]=dict(repeats=[entry(raw,stem+'-n'+str(layer)+'-r'+str(k)+'.f32') for k in range(2)],input_after_sha256=[hashlib.sha256(raw).hexdigest()]*2);items=[]
                        for i in row['physical_rows']:
                            raw=struct.pack('<H',0x3f80+i)*1024;items.append(dict(physical_row=i,slot=len(report['inputs'][i]['prompt_ids'])+step-1,heads=4,values_per_head=256,repeats=[entry(raw,stem+'-v'+str(layer)+'-row'+str(i)+'-r'+str(k)+'.bf16') for k in range(2)]))
                        e['value_cache'][str(layer)]=items
        return r,c,parents,files
    def change(self,entry,files,raw):files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
    def test_complete_primary_snapshots_required_for_both_batches(self):
        r,c,parents,f=self.fixture()
        for report,parent in zip((r,c),parents):self.assertEqual(primary_parity(report,parent,f.__getitem__,f.__getitem__)['complete_snapshots'],16 if report['physical_batch']==1 else 4)
        for phase in ('reset_state','prefill_state','final_state','final_reset'):
            bad=copy.deepcopy(c);bad['rows'][0][phase]['logical']['components'][127]['row_sha256'][3]='f'*64
            with self.assertRaises(AssertionError):primary_parity(bad,parents[1],f.__getitem__,f.__getitem__)
        for key,value in (('library_sha256','f'*64),('capacity',4095),('physical_batch',1)):
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(AssertionError):primary_parity(bad,parents[1],f.__getitem__,f.__getitem__)
        bad=copy.deepcopy(c);bad['rows'][0]['generated_ids'][3][7]=0
        with self.assertRaises(AssertionError):primary_parity(bad,parents[1],f.__getitem__,f.__getitem__)
    def test_late_primary_signedzero_vector_and_nonfinite_are_not_waived(self):
        r,c,parents,f=self.fixture();e=c['rows'][0]['final_state']['vectors']['hidden'];raw=f[e['file']];self.change(e,f,raw[:-4]+struct.pack('<f',-0.0))
        with self.assertRaises(AssertionError):primary_parity(c,parents[1],f.__getitem__,f.__getitem__)
        self.change(e,f,raw[:-4]+struct.pack('<f',float('nan')))
        with self.assertRaises(AssertionError):primary_parity(c,parents[1],f.__getitem__,f.__getitem__)
    def test_all_seven_full_arrays_and_actual_cache_are_compared(self):
        r,c,parents,f=self.fixture();a=analyze(r,c,f.__getitem__,f.__getitem__);self.assertEqual(len(a['rows']),28);self.assertTrue(all(v['all_consumed_inputs_so_far_exact'] and v['first_differing_observed_boundary'] is None for v in a['rows']))
        e=c['rows'][0]['decode_observations'][6]['boundaries']['12'];raw=f[e['file']];self.change(e,f,raw[:-4]+struct.pack('<f',-0.0));a=analyze(r,c,f.__getitem__,f.__getitem__);last=a['rows'][-1]['boundaries']['12'];self.assertEqual((last['different_words'],last['active_different_words'],last['inactive_different_words'],last['first_different_word']),(1,0,1,20479))
    def test_repeats_input_mutation_slots_counts_and_row_permutation_rejected(self):
        r,c,parents,f=self.fixture()
        def mutations(q):
            q['rows'][0]['decode_observations'][6]['program_runs']=6
        bad=copy.deepcopy(c);mutations(bad)
        with self.assertRaises(AssertionError):validate_observations(bad,f.__getitem__)
        bad=copy.deepcopy(c);bad['rows'][0]['decode_observations'][0]['physical_rows']=[1,0,2,3]
        with self.assertRaises(AssertionError):validate_observations(bad,f.__getitem__)
        bad=copy.deepcopy(c);bad['rows'][0]['decode_observations'][0]['value_cache']['3'][3]['slot']+=1
        with self.assertRaises(AssertionError):validate_observations(bad,f.__getitem__)
        bad=copy.deepcopy(c);bad['rows'][0]['decode_observations'][0]['normalizations']['3']['input_after_sha256'][1]='f'*64
        with self.assertRaises(AssertionError):validate_observations(bad,f.__getitem__)
        bad=copy.deepcopy(c);e=bad['rows'][0]['decode_observations'][0]['normalizations']['3']['repeats'][1];raw=f[e['file']];self.change(e,f,b'\0'*len(raw))
        with self.assertRaises(AssertionError):validate_observations(bad,f.__getitem__)
    def test_nonfinite_inactive_float_and_low_packed_bf16_half_rejected(self):
        r,c,parents,f=self.fixture();e=c['rows'][0]['decode_observations'][6]['boundaries']['12'];raw=f[e['file']];self.change(e,f,raw[:-4]+struct.pack('<f',float('inf')))
        with self.assertRaises(AssertionError):validate_observations(c,f.__getitem__)
        raw=struct.pack('<HH',0x7fc0,0x3f80);e=dict(file='packed.bf16',bytes=4,sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(AssertionError):blob(e,lambda _:raw,4,'bf16')
    def test_conditioning_divergence_explicit_and_bf16_word_exact(self):
        r,c,parents,f=self.fixture();row=c['rows'][0];row['generated_ids'][3][5]=999;row['decode_observations'][5]['consumed_tokens'][3]=999;a=analyze(r,c,f.__getitem__,f.__getitem__)
        self.assertTrue(a['rows'][19]['all_consumed_inputs_so_far_exact']);self.assertFalse(a['rows'][23]['all_consumed_inputs_so_far_exact']);self.assertFalse(a['rows'][27]['all_consumed_inputs_so_far_exact']);self.assertTrue(a['rows'][26]['all_consumed_inputs_so_far_exact'])
        self.assertEqual(difference(struct.pack('<H',0),struct.pack('<H',0x8000),'bf16')['different_words'],1)
if __name__=='__main__':unittest.main()
