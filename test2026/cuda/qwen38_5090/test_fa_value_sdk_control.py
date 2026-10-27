from pathlib import Path
import json,struct,unittest
import numpy as np
from .fa_value_sdk_control import CASES,K,N,active_values,observe,replay,parse_log,verify_derivations
M=Path(__file__).resolve().parent
class Controls(unittest.TestCase):
    def fixture(self):
        f={};a={}
        for label,layer,step in CASES:
            stem=label+'-step0';f[label+'-input-scale.f32']=struct.pack('<f',1.);f[stem+'-normalized.f32']=bytes(16*K*4);base=np.arange(16*N,dtype='<u2')%1024;base+=0x3f00+np.repeat(np.arange(16,dtype='<u2')*4,N);zero=base.tobytes();changed=bytearray(zero);changed[(4*N+17)*2]^=1;changed=bytes(changed)
            f[stem+'-garnet-m4-active.bf16']=active_values(zero);f[stem+'-garnet-m16-active.bf16']=active_values(changed);a[stem+'-cast.bf16']=bytes(16*K*2);a[stem+'-packed4.e4m3']=a[stem+'-packed16.e4m3']=bytes(16*K)
            for repeat in range(2):
                for mode in ('m4','default16','shared16'):a[stem+'-'+mode+'-r'+str(repeat)+'.bf16']=changed if mode=='default16' else zero
        return f,a
    def test_full_output_intervention_and_native_active_baselines(self):
        f,a=self.fixture();r=replay(f.__getitem__,a.__getitem__);self.assertEqual(len(r['cases']),4);self.assertTrue(all(v['default_M16_vs_four_M4_active_different_words']==1 for v in r['cases']));self.assertFalse(r['full_Garnet_inactive_V_projection_outputs_captured'])
    def test_active_stride_signedzero_roworder_and_full_padding_not_waived(self):
        f,a=self.fixture();raw=np.zeros((4,4,N),dtype='<u2');raw[3,0,-1]=0x8000;self.assertEqual(active_values(raw.tobytes())[-2:],b'\x00\x80')
        for repeat in range(2):
            name='v11-step0-shared16-r'+str(repeat)+'.bf16';a[name]=a[name][:-2]+b'\x00\x80'
        r=observe(f.__getitem__,a.__getitem__);self.assertTrue(r['four_M4_and_default_M16_reproduce_actual_Garnet_active_cache']);self.assertFalse(r['same_M4_algorithm_at_M16_full_words_exact'])
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        f,a=self.fixture();name='v3-step0-garnet-m4-active.bf16';f[name]=f[name][2*N:]+f[name][:2*N]
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
    def test_actual_baseline_quantization_nonfinite_and_repeat_failures(self):
        f,a=self.fixture();f['v3-step0-garnet-m16-active.bf16']=f['v3-step0-garnet-m4-active.bf16']
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        for name in ('v3-step0-packed4.e4m3','v11-step0-m4-r1.bf16'):
            f,a=self.fixture();a[name]=b'\x01'+a[name][1:]
            with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        f,a=self.fixture();name='v11-step0-default16-r0.bf16';a[name]=a[name][:-2]+b'\x80\x7f'
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
    def log(self):
        lines=['DEVICE synthetic SM120','CUBLASLT_VERSION 130400']
        for label,_,_ in CASES:
            for mode in ('m4','default16','shared16'):
                for repeat in range(2):
                    if mode=='shared16':lines.append(f'ALGO_CHECK projection={label} step=0 repeat={repeat} m=16 status=0 state=0 workspace=100 capacity=4194304')
                    for row in range(4 if mode=='m4' else 1):lines.append(f'ALGO projection={label} step=0 repeat={repeat} row={row} mode={mode} m={4 if mode=="m4" else 16} status=0 id={67 if mode=="default16" else 35} tile=1 stages=2 split=3 reduction=4 swizzle=5 custom=6 workspace=4194304')
        return '\n'.join(lines+['ACTUAL_M4_M16_SDK_CAPTURE_COMPLETE_NO_MODEL_OR_PERFORMANCE_ADMISSION'])
    def test_live_seven_attributes_full_call_coverage_and_actual_M16_checks(self):
        raw=self.log();r=parse_log(raw);self.assertEqual(len(r['calls']),24);self.assertEqual(r['actual_M16_AlgoChecks'],4)
        for bad in (raw.replace('mode=shared16 m=16 status=0 id=35','mode=shared16 m=16 status=0 id=67',1),raw.replace('workspace=100 capacity=4194304','workspace=4194305 capacity=4194304',1),raw.replace('mode=m4 m=4','mode=m4 m=8',1),raw.replace('state=0 workspace=100','state=1 workspace=100',1),raw.replace('tile=1 stages=2','tile=7 stages=2',1),raw+'\n'+raw.splitlines()[2]):
            with self.assertRaises(AssertionError):parse_log(bad)
    def test_exact_native_math_and_parser_derivation(self):
        record=json.loads((M/'evidence/fa-value-sdk-derivation.json').read_bytes());verify_derivations(record,lambda name:(M/name).read_bytes())
        self.assertIn(b'const int n=1024',(M/'actual_decode_fa_value_m4_m16.cu').read_bytes())
    def test_shapes_and_nonfinite_input_are_rejected(self):
        with self.assertRaises(AssertionError):active_values(bytes(16*N*2-2))
        f,a=self.fixture();name='v3-step0-normalized.f32';f[name]=struct.pack('<f',float('nan'))+f[name][4:]
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
if __name__=='__main__':unittest.main()
