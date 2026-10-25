"""CPU coverage and admission isolation; no model/GPU import."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bf16_low_batch_attention_oracle import case_plan,sample_triples,spans
from xModel.qwen35.vl_0_8b.bf16_kv_profile import validate_bf16_kv_program

class LowBatchCandidateTests(unittest.TestCase):
    def test_full_chunk_and_boundary_coverage_for_single_request(self):
        plan=case_plan();self.assertEqual(len(plan),44);self.assertEqual(len({c['id'] for c in plan}),44)
        self.assertEqual({c['batch'] for c in plan},{1})
        self.assertEqual({c['layer'] for c in plan},set(range(3,64,4)))
        for chunk in (4,32):
            for layer in (3,63):
                self.assertTrue({0,1,511,512,513,4095,4096}.issubset({c['end'] for c in plan if c['chunk']==chunk and c['layer']==layer}))
        full=[c for c in plan if c['chunk']==32 and spans(c)[0][1]==32]
        self.assertEqual({c['end'] for c in full},{32,4096})
        self.assertTrue(all(spans(c)[0][0] in (0,4064) for c in full))
        self.assertEqual(sum(len(sample_triples(c))*256 for c in plan),119040)
        self.assertTrue(all(b==0 for c in plan for b,t,h in sample_triples(c)))

    def test_existing_full_decoder_admission_still_rejects_b1(self):
        config=json.loads((Path(__file__).with_name('evidence')/'nvfp4-config.json').read_bytes())['text_config']
        for chunk in (4,32):
            with self.assertRaises(ValueError):
                validate_bf16_kv_program(True,1,4096,chunk,chunk==4,config,external_compressed_weights=True,joined_mlp_projections=True)

if __name__=='__main__':unittest.main()
