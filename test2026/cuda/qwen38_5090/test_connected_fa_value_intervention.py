import copy,hashlib,struct,unittest
import numpy as np
from . import test_connected_fa_value_gate as baseline_fixtures
from .connected_fa_value_intervention import admit,match_prior_baseline

class Intervention(unittest.TestCase):
    def fixture(self):
        reference,off,files,fixture=baseline_fixtures.Gate().fixture()
        # Distinct finite full-row hidden patterns catch permutation/inactive-word faults.
        for report in (reference,off):
            for record in report['cases']:
                raw=b''.join(struct.pack('<f',row+.25)*20480 for row in record['physical_rows'])
                for entry in record['repeats']:self.change(entry['component_hidden'],files,raw)
        on=copy.deepcopy(off)
        for record in on['cases']:
            entries=[record['input']]+[entry[key] for entry in record['repeats'] for key in ('unmasked_normalized','component_hidden','component_V')]
            for entry in entries:
                original=entry['file'];entry['file']='on-'+original;files[entry['file']]=files[original]
            source=fixture[record['label']+'-step0-garnet-m4-active.bf16'];raw=(np.frombuffer(source,dtype='<u2').astype('<u4')<<16).tobytes()
            for entry in record['repeats']:self.change(entry['component_V'],files,raw)
        return reference,off,on,files,fixture
    def change(self,entry,files,raw):
        files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
    def run_gate(self,r,off,on,files,f):return admit(r,off,on,files.__getitem__,files.__getitem__,files.__getitem__,f.__getitem__)
    def test_complete_exact_component_and_failure_scope_retained(self):
        r,off,on,a,f=self.fixture();q=self.run_gate(r,off,on,a,f)
        self.assertEqual(len(q['hidden']),16);self.assertEqual(sum(v['B4_component_vs_four_B1_different_words'] for v in q['baseline']['rows']),2)
        self.assertTrue(all(v['on_vs_four_B1_different_words']==0 for v in q['hidden']))
        self.assertFalse(q['full_model_or_quality_or_performance_qualified']);self.assertFalse(q['original_failed_SDK_or_model_job_promoted'])
    def test_prior_baseline_all_arrays_and_metadata_required(self):
        for use_reference in (True,False):
            r,off,on,a,f=self.fixture();current=r if use_reference else off
            for row in current['cases']:row['weight_metadata']={'actual_pinned_weight':'fixture'}
            prior=copy.deepcopy(current);self.assertEqual(match_prior_baseline(current,prior,a.__getitem__,a.__getitem__)['arrays'],56 if use_reference else 14)
            for mutation in ('hidden','metadata'):
                changed=copy.deepcopy(current)
                if mutation=='metadata':changed['cases'][0]['weight_metadata']={}
                else:
                    entry=changed['cases'][0]['repeats'][1]['component_hidden'];old=entry['file'];entry['file']='changed-'+old;raw=a[old];self.change(entry,a,raw[:-4]+struct.pack('<f',99.))
                with self.assertRaises(AssertionError):match_prior_baseline(changed,prior,a.__getitem__,a.__getitem__)
    def test_original_off_baseline_cannot_be_replaced_with_corrected_values(self):
        r,off,on,a,f=self.fixture()
        for before,after in zip(off['cases'],on['cases']):
            for x,y in zip(before['repeats'],after['repeats']):self.change(x['component_V'],a,a[y['component_V']['file']])
        with self.assertRaises(AssertionError):self.run_gate(r,off,on,a,f)
    def test_uncorrected_V_intervention_rejected(self):
        r,off,on,a,f=self.fixture()
        with self.assertRaises(AssertionError):self.run_gate(r,off,copy.deepcopy(off),a,f)
    def test_complete_inactive_hidden_signed_zero_and_finite_faults_rejected(self):
        for word in (0x80000000,0x7fc00000):
            r,off,on,a,f=self.fixture()
            for entry in on['cases'][0]['repeats']:
                target=entry['component_hidden'];raw=a[target['file']];self.change(target,a,raw[:-4]+struct.pack('<I',word))
            with self.assertRaises(AssertionError):self.run_gate(r,off,on,a,f)
    def test_V_and_hidden_row_permutations_rejected(self):
        for key,width in [('component_V',4096),('component_hidden',81920)]:
            r,off,on,a,f=self.fixture()
            for entry in on['cases'][0]['repeats']:
                target=entry[key];raw=a[target['file']];self.change(target,a,raw[width:2*width]+raw[:width]+raw[2*width:])
            with self.assertRaises(AssertionError):self.run_gate(r,off,on,a,f)
    def test_original_repeat_input_and_full_cache_guards_apply_to_on(self):
        for mutation in ('repeat','input','cache'):
            r,off,on,a,f=self.fixture()
            if mutation=='repeat':
                target=on['cases'][0]['repeats'][1]['component_hidden'];raw=a[target['file']];self.change(target,a,struct.pack('<f',.5)+raw[4:])
            elif mutation=='input':on['cases'][0]['input_after_sha256']='f'*64
            else:on['cases'][0]['input_cache_sha256_after'][7]='f'*64
            with self.assertRaises(AssertionError):self.run_gate(r,off,on,a,f)
if __name__=='__main__':unittest.main()
