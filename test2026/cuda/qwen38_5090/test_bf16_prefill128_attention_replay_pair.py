import copy,unittest
from .compare_bf16_prefill128_attention_replays import compare_replays

class Controls(unittest.TestCase):
    def proofs(self):
        return {'status':'prefill128_attention_component_full_fp64_replayed','full_output_scalars_replayed':28311552,'numerical_failures':[],
                'archive_sha256':'same','replay_verifier_sha256':'same','atol':3e-5,'rtol':2e-5,
                'cases':[{'id':i,'full_output_scalars':786432,'active_output_scalars':i*6144,'inactive_output_exact_zero':True,
                          'max_absolute_error':1e-8,'max_tolerance_fraction':.001} for i in range(36)]}
    def test_only_bounded_error_summary_differences_allowed(self):
        a=self.proofs();b=copy.deepcopy(a);b['cases'][3]['max_absolute_error']+=1e-16
        r=compare_replays(a,b);self.assertEqual(len(r['summary_differences']),1);self.assertFalse(r['summary_values_identical'])
        self.assertTrue(compare_replays(a,a)['summary_values_identical'])
    def test_structural_failure_nonfinite_and_large_summary_differences_rejected(self):
        a=self.proofs()
        for key,value in (('archive_sha256','different'),('status','failure'),('full_output_scalars_replayed',786432),('atol',1)):
            b=copy.deepcopy(a);b[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):compare_replays(a,b)
        for key,value in (('id',99),('full_output_scalars',1),('inactive_output_exact_zero',False),('max_absolute_error',float('nan')),('max_absolute_error',1e-4),('max_tolerance_fraction',1.1),('max_tolerance_fraction',.002)):
            b=copy.deepcopy(a);b['cases'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):compare_replays(a,b)

if __name__=='__main__':unittest.main()
