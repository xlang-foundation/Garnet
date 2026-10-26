"""CPU-only complete-output oracle and probe provenance controls."""
from pathlib import Path
import hashlib,json,unittest
import numpy as np
from .bf16_prefill128_attention_oracle import case_plan,reference_full,compare_full,spans,fp32,bf16,raw_bits,current_bits
from .bf16_kv_overlay_oracle import reference_case

class Controls(unittest.TestCase):
    def test_complete_case_plan_includes_full128_and_all_attention_layers(self):
        cases=case_plan();self.assertEqual(len(cases),36);self.assertEqual(len({r['id'] for r in cases}),36)
        self.assertEqual({r['layer'] for r in cases},set(range(3,64,4)))
        for identifier,end in ((127,128),(255,4096)):
            case=next(r for r in cases if r['id']==identifier);self.assertEqual(spans(case),[(end-128,128)])
        self.assertEqual({r['id'] for r in cases if r['id'] in (1,15,127)},{1,15,127})

    def test_inactive_full_output_and_malformed_buffers(self):
        case=case_plan()[0];zeros=np.zeros((1,128,24,256),dtype='<f4');result=compare_full(case,zeros.tobytes())
        self.assertEqual(result['full_output_scalars'],786432);self.assertEqual(result['active_output_scalars'],0)
        for raw in (zeros.tobytes()[:-4],np.full(zeros.shape,np.nan,dtype='<f4').tobytes()):
            with self.assertRaises(ValueError):compare_full(case,raw)
        with self.assertRaises(ValueError):reference_full({**case,'batch':True})

    def test_single_visible_key_analytic_all24heads_and_strict_inactive_zero(self):
        case=case_plan()[1];wanted=reference_full(case);dims=np.arange(256,dtype=np.uint32)
        for h in range(24):
            value=bf16(current_bits((h//6)*256+dims,True));gate=fp32(raw_bits(h*256+dims,4127,0x3f800000))
            np.testing.assert_allclose(wanted[0,0,h],value/(1+np.exp(-gate)),atol=1e-12,rtol=1e-12)
        actual=wanted.astype('<f4');compare_full(case,actual.tobytes())
        actual[0,0,23,255]+=.01
        with self.assertRaises(ValueError):compare_full(case,actual.tobytes())
        actual=wanted.astype('<f4');actual[0,127,0,0]=1e-9
        with self.assertRaises(ValueError):compare_full(case,actual.tobytes())

    def test_complete_reference_agrees_with_previous_independent_sample_reference(self):
        case=case_plan()[2];wanted=reference_full(case)
        for (b,t,h),row in reference_case(case).items():np.testing.assert_allclose(wanted[b,t,h],row,atol=1e-12,rtol=1e-12)

    def test_exact_probe_derivation_and_full_witness_shape(self):
        m=Path(__file__).resolve().parent;d=json.loads((m/'evidence/prefill128-attention-probe-derivation.json').read_bytes())
        raw=(m/d['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),d['source_sha256']);text=raw.decode()
        for step in d['replacements']:
            self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
        actual=(m/d['target']).read_bytes();self.assertEqual(actual,text.encode());self.assertEqual(hashlib.sha256(actual).hexdigest(),d['target_sha256'])
        self.assertIn('nq*4);witnessOffset+=nq*4',text);self.assertIn('graphWorkspace==originalWorkspace',text)
        self.assertNotIn('bf16Layers4',text)

if __name__=='__main__':unittest.main()
