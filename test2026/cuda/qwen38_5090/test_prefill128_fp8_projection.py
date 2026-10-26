"""Synthetic input/word/provenance faults only, never real GPU measurements."""
from pathlib import Path
import hashlib,json,struct,unittest
from .prefill128_fp8_projection import input_oracle,replay_projection

def fixture():
    x=struct.pack('<f',1.0)*(16*6144);y=struct.pack('<H',0x3f80)*(16*5120)
    source={'input-m16-k6144.f32':x,'input-m32-k6144.f32':x*2,'expected-m16-n5120.bf16':y,'expected-m32-n5120.bf16':y*2,'input-scale.f32':struct.pack('<f',1.)};outputs={}
    for m in (16,32,128):
        cast,quant=input_oracle(x*(m//16),1.);outputs['cast-m'+str(m)+'.bf16']=cast;outputs['quantized-m'+str(m)+'.e4m3']=quant
        for repeat in (0,1):outputs['actual-m'+str(m)+'-r'+str(repeat)+'.bf16']=y*(m//16)
    return source,outputs

class Controls(unittest.TestCase):
    def test_all_complete_output_words_and_input_casts(self):
        source,outputs=fixture()
        for mode in ('control','shared'):
            result=replay_projection(source.__getitem__,outputs.__getitem__,mode);self.assertEqual(result['complete_bf16_output_words_replayed'],1802240);self.assertFalse(result['model_or_performance_qualified'])

    def test_default128_difference_is_recorded_but_shared_last_word_cannot_drift(self):
        source,outputs=fixture()
        for repeat in (0,1):
            name='actual-m128-r'+str(repeat)+'.bf16';outputs[name]=outputs[name][:-2]+struct.pack('<H',0x4000)
        report=replay_projection(source.__getitem__,outputs.__getitem__,'control');self.assertEqual(report['cases'][-1]['different_from_repeated_m16_output_words'],1)
        with self.assertRaises(AssertionError):replay_projection(source.__getitem__,outputs.__getitem__,'shared')

    def test_cast_quantized_input_duplicate_rows_and_nonfinite_outputs_rejected(self):
        for fault in ('cast','quantized','rows','output'):
            source,outputs=fixture()
            if fault=='rows':source['input-m32-k6144.f32']=source['input-m32-k6144.f32'][:-4]+struct.pack('<f',2.)
            elif fault=='output':outputs['actual-m128-r1.bf16']=outputs['actual-m128-r1.bf16'][:-2]+struct.pack('<H',0x7fc0)
            else:
                name=('cast-m128.bf16' if fault=='cast' else 'quantized-m128.e4m3');outputs[name]=outputs[name][:-1]+bytes([outputs[name][-1]^1])
            with self.subTest(fault=fault),self.assertRaises(AssertionError):replay_projection(source.__getitem__,outputs.__getitem__,'control')

    def test_exact_recorded_native_source_transformation(self):
        m=Path(__file__).resolve().parent;record=json.loads((m/'evidence/prefill128-fp8-projection-derivation.json').read_bytes());raw=(m/record['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),record['source_sha256']);text=raw.decode()
        for step in record['replacements']:
            self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
        self.assertEqual(text.encode(),(m/record['target']).read_bytes());self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),record['target_sha256'])

if __name__=='__main__':unittest.main()
