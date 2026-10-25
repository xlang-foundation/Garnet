"""Small analytical controls for the independent FP64 reference itself."""
import unittest
import numpy as np
from bf16_kv_overlay_oracle import attention,bf16,case_plan,compare_witness,reference_case,sample_triples

class OracleTests(unittest.TestCase):
    def test_bf16_signed_ties_choose_even(self):
        actual=bf16(np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],dtype=np.uint32))
        np.testing.assert_array_equal(actual,np.array([1,1.015625,-1,-1.015625]))
    def test_explicit_two_key_softmax_and_gate(self):
        actual=attention([[np.sqrt(2),0]],[[0,1],[1,1]],[[2,4],[6,8]],[[0,0]],np.array([2]))
        np.testing.assert_allclose(actual,[(np.array([2,4])+np.e*np.array([6,8]))/(1+np.e)/2],rtol=1e-14)
    def test_future_value_is_masked(self):
        actual=attention([[2,3]],[[1,2],[1e5,1e5]],[[4,-8],[1e6,1e6]],[[0,0]],np.array([1]))
        np.testing.assert_array_equal(actual,[[2,-4]])
    def test_inactive_case_has_no_samples(self):
        case=case_plan()[0]
        self.assertEqual(reference_case(case),{});self.assertEqual(sample_triples(case),[])
        self.assertEqual(compare_witness(case,b'')['scalars'],0)
    def test_first_token_known_current_value_ties_and_gate(self):
        case=case_plan()[1];ref=reference_case(case)
        # The only visible value is current-token BF16, so softmax is exactly1.
        from bf16_kv_overlay_oracle import fp32,raw_bits
        gate=fp32(raw_bits(np.arange(256,dtype=np.uint32),4127,0x3f800000))
        np.testing.assert_allclose(ref[(0,0,0)][:4],np.array([1,1.015625,-1,-1.015625])/(1+np.exp(-gate[:4])),rtol=1e-14)
    def test_corrupt_witness_is_rejected(self):
        case=case_plan()[1];ref=reference_case(case)
        raw=np.stack([ref[k] for k in sample_triples(case)]).astype('<f4')
        self.assertLess(compare_witness(case,raw.tobytes())['max_tolerance_fraction'],1)
        raw[0,0]+=.01
        with self.assertRaises(ValueError):compare_witness(case,raw.tobytes())
    def test_invalid_oracle_geometry_and_nonfinite_reject(self):
        for visible in (np.array([0]),np.array([2]),np.array([1.5])):
            with self.assertRaises(ValueError):attention([[1]],[[1]],[[1]],[[0]],visible)
        with self.assertRaises(ValueError):attention([[np.nan]],[[1]],[[1]],[[0]],np.array([1]))
    def test_case_plan_covers_all_attention_offsets_and_boundary_profiles(self):
        plan=case_plan();self.assertEqual(len(plan),70)
        self.assertEqual({c['layer'] for c in plan},set(range(3,64,4)))
        for b in (4,8):
            for t in (4,32):
                self.assertEqual({c['end'] for c in plan if c['batch']==b and c['chunk']==t},{0,1,511,512,513,4095,4096})

if __name__=='__main__':unittest.main()
