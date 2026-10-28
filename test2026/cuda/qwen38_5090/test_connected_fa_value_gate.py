import copy,hashlib,struct,unittest
import numpy as np
from .connected_fa_value_gate import observe,admit,COUNTS
from .fa_value_sdk_control import CASES
class Gate(unittest.TestCase):
    def fixture(self):
        f={};files={};reports=[];sha=lambda raw:hashlib.sha256(raw).hexdigest()
        def entry(name,raw):files[name]=raw;return dict(file=name,bytes=len(raw),sha256=sha(raw))
        for label,_,_ in CASES:
            f[label+'-step0-hidden.f32']=f[label+'-step0-normalized.f32']=bytes(16*5120*4);base=np.repeat(np.arange(4,dtype='<u2')+0x3f00,1024);f[label+'-step0-garnet-m4-active.bf16']=base.tobytes();base[1024+17]^=1;f[label+'-step0-garnet-m16-active.bf16']=base.tobytes()
        for mode,batch in (('reference',1),('candidate',4)):
            q=dict(status='connected_FA_value_components_collected',mode=mode,physical_batch=batch,tokens=4,capacity=4096,model_or_performance_qualified=False,internal_fused_normalization_directly_captured=False,cache_mode='ordinary_zero_FP32_inputs_component_only',position_and_slot_controls='zero_component_controls_V_projection_independent',cases=[])
            for label,layer,step in CASES:
                for physical in ([[r] for r in range(4)] if batch==1 else [[0,1,2,3]]):
                    stem=mode+'-'+label+'-'+str(physical[0]);record=dict(label=label,layer=layer,actual_decode_step=step,physical_rows=physical,input=entry(stem+'-input.f32',bytes(batch*81920)),graph_stats=dict(**{key:0 for key in COUNTS},native_fp8_projection_weights=4,native_nvfp4_projection_weights=2,external_compressed_weight_inputs=6),norm_stats=dict(**{key:0 for key in COUNTS},native_fp8_projection_weights=0,native_nvfp4_projection_weights=0),input_cache_sha256_before=[sha(bytes(4194304*4))]*(2*batch),repeats=[]);record['input_cache_sha256_after']=record['input_cache_sha256_before'][:];record['input_after_sha256']=record['input']['sha256']
                    expected=f[label+'-step0-garnet-'+('m4' if batch==1 else 'm16')+'-active.bf16'];raw=b''.join(expected[r*2048:(r+1)*2048] for r in physical);raw=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).tobytes()
                    for repeat in range(2):record['repeats'].append(dict(repeat=repeat,unmasked_normalized=entry(stem+'-n'+str(repeat)+'.f32',bytes(batch*81920)),component_hidden=entry(stem+'-h'+str(repeat)+'.f32',bytes(batch*81920)),component_V=entry(stem+'-v'+str(repeat)+'.f32',raw),input_after_sha256=record['input']['sha256']))
                    q['cases'].append(record)
            reports.append(q)
        return *reports,files,f
    def run_gate(self,r,c,a,f,strict=True):return (admit if strict else observe)(r,c,a.__getitem__,a.__getitem__,f.__getitem__)
    def change(self,entry,files,raw):files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
    def test_all_real_cache_baselines_required_and_cross_profile_differences_retained(self):
        r,c,a,f=self.fixture();q=self.run_gate(r,c,a,f);self.assertEqual(len(q['rows']),8);self.assertEqual(sum(row['B4_component_vs_four_B1_different_words'] for row in q['rows']),2)
        for entry in c['cases'][1]['repeats']:
            e=entry['component_V'];raw=a[e['file']];self.change(e,a,raw[:-4]+struct.pack('<I',0x80000000))
        self.assertFalse(self.run_gate(r,c,a,f,False)['all_original_cache_baselines_exact'])
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
    def test_original_normalizer_and_full_inactive_words_cannot_be_waived(self):
        r,c,a,f=self.fixture();e=c['cases'][0]['repeats'][0]['unmasked_normalized'];raw=a[e['file']];self.change(e,a,raw[:-4]+struct.pack('<f',-0.0))
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
    def test_finite_BF16_cache_and_repeat_stability(self):
        for bad_word in (0x7fc00000,0x3f800001):
            r,c,a,f=self.fixture();e=c['cases'][0]['repeats'][0]['component_V'];raw=a[e['file']];self.change(e,a,struct.pack('<I',bad_word)+raw[4:])
            with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
        r,c,a,f=self.fixture();e=c['cases'][1]['repeats'][1]['component_hidden'];raw=a[e['file']];self.change(e,a,raw[:-4]+struct.pack('<f',-0.0))
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
    def test_complete_input_caches_and_input_readonly_required(self):
        for field in ('input_cache_sha256_after','input_cache_sha256_before'):
            r,c,a,f=self.fixture();c['cases'][0][field][7]='f'*64
            with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
        r,c,a,f=self.fixture();c['cases'][0]['repeats'][1]['input_after_sha256']='f'*64
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
    def test_missing_row_duplicate_row_wrong_geometry_and_routes_rejected(self):
        for mutation in ('duplicate','batch','tokens','counts'):
            r,c,a,f=self.fixture()
            if mutation=='duplicate':r['cases'][3]['physical_rows']=[2]
            elif mutation=='batch':c['physical_batch']=True
            elif mutation=='tokens':c['tokens']=16
            else:c['cases'][0]['graph_stats']['native_fp8_projection_weights']=3
            with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
    def test_truncated_input_and_cache_rejected(self):
        r,c,a,f=self.fixture();e=r['cases'][7]['input'];a[e['file']]=a[e['file']][:-4]
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
        r,c,a,f=self.fixture();f['v11-step0-garnet-m16-active.bf16']=f['v11-step0-garnet-m16-active.bf16'][:-2]
        with self.assertRaises(AssertionError):self.run_gate(r,c,a,f)
if __name__=='__main__':unittest.main()
