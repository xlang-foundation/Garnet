import copy,struct,unittest
from .decode_fp8_sdk_control import replay,observe,parse_log,PROJECTIONS,K
class Controls(unittest.TestCase):
    def fixture(self):
        fixture={};actual={}
        for step in range(5):fixture['step'+str(step)+'-normalized.f32']=bytes(16*K*4)
        for label,n in PROJECTIONS.items():
            fixture[label+'-input-scale.f32']=struct.pack('<f',1.)
            for step in range(5):
                stem=label+'-step'+str(step);zero=bytes(16*n*2);changed=b'\x80\x3f'+zero[2:];fixture[stem+'-garnet-m4.bf16']=zero;fixture[stem+'-garnet-m16.bf16']=changed
                actual[stem+'-cast.bf16']=bytes(16*K*2);actual[stem+'-packed4.e4m3']=actual[stem+'-packed16.e4m3']=bytes(16*K)
                for repeat in range(2):
                    for mode in ('m4','default16','shared16'):actual[stem+'-'+mode+'-r'+str(repeat)+'.bf16']=changed if mode=='default16' else zero
        return fixture,actual
    def test_positive_full_word_and_quantization_controls(self):
        f,a=self.fixture();r=replay(f.__getitem__,a.__getitem__);self.assertEqual(len(r['cases']),20);self.assertTrue(r['same_M4_algorithm_at_M16_exact'])
    def test_last_word_shared_control_and_baseline_corruption(self):
        f,a=self.fixture()
        for repeat in range(2):
            name='z-step4-shared16-r'+str(repeat)+'.bf16';a[name]=a[name][:-2]+b'\x80\x3f'
        r=observe(f.__getitem__,a.__getitem__);self.assertFalse(r['same_M4_algorithm_at_M16_exact'])
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        f,a=self.fixture();a['qkv-step0-default16-r0.bf16']=a['qkv-step0-m4-r0.bf16']
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
    def test_quantization_finiteness_and_repeat_corruption(self):
        f,a=self.fixture();name='qkv-step0-packed4.e4m3';a[name]=b'\x01'+a[name][1:]
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        f,a=self.fixture();name='z-step0-m4-r0.bf16';a[name]=b'\x80\x7f'+a[name][2:]
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
        f,a=self.fixture();name='qkv-step0-m4-r1.bf16';a[name]=b'\x80\x3f'+a[name][2:]
        with self.assertRaises(AssertionError):replay(f.__getitem__,a.__getitem__)
    def log(self):
        lines=['DEVICE synthetic SM120','CUBLASLT_VERSION 130400']
        for p in PROJECTIONS:
            for s in range(5):
                for mode in ('m4','default16','shared16'):
                    for r in range(2):
                        if mode=='shared16':lines.append(f'ALGO_CHECK projection={p} step={s} repeat={r} m=16 status=0 state=0 workspace=100 capacity=4194304')
                        for row in range(4 if mode=='m4' else 1):lines.append(f'ALGO projection={p} step={s} repeat={r} row={row} mode={mode} m={4 if mode=="m4" else 16} status=0 id={67 if mode=="default16" else 35} tile=1 stages=2 split=3 reduction=4 swizzle=5 custom=6 workspace=4194304')
        return '\n'.join(lines+['ACTUAL_M4_M16_SDK_CAPTURE_COMPLETE_NO_MODEL_OR_PERFORMANCE_ADMISSION'])
    def test_actual_shape_config_and_check_log_guards(self):
        raw=self.log();self.assertEqual(parse_log(raw)['actual_M16_AlgoChecks'],20)
        for bad in (raw.replace('mode=shared16 m=16 status=0 id=35','mode=shared16 m=16 status=0 id=67',1),raw.replace('workspace=100 capacity=4194304','workspace=4194305 capacity=4194304',1),raw.replace('mode=m4 m=4','mode=m4 m=8',1)):
            with self.assertRaises(AssertionError):parse_log(bad)
if __name__=='__main__':unittest.main()
