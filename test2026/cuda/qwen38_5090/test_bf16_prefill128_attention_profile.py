"""CPU geometry/provenance checks only; never imply numerical qualification."""
from pathlib import Path
import ast,hashlib,json,unittest
from .bf16_prefill128_attention_profile import profile
from .bf16_prefill128_layout import build_bf16_prefill128_layout
from xModel.qwen35.vl_0_8b.bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_program

class Controls(unittest.TestCase):
    def test_geometry_matches_packed_canonical_layout_and_exact_workspace(self):
        p=profile();self.assertEqual(p['workspace_bytes'],50331648);self.assertEqual(p['canonical_bytes'],427294720)
        self.assertEqual(p['query_bytes'],3145728);self.assertEqual(p['chunk_kv_bytes'],524288)
        layout=build_bf16_prefill128_layout();self.assertEqual(len(p['full_attention_offsets']),16)
        self.assertEqual(p['full_attention_offsets'],[(r['offset'],r['offset']+r['state1_width']) for r in layout['layers'] if r['kind']=='full_attention'])
        self.assertFalse(p['runtime_qualified']);self.assertFalse(p['performance_admitted'])

    def test_malformed_dimensions_and_old_profiles_do_not_enter_new_component(self):
        for key,values in (('batch',(0,2,4,8,True)),('tokens',(4,32,64,127,129,128.0)),('heads',(0,16,32)),
                           ('kv',(0,8)),('capacity',(4095,4097,True)),('dim',(128,512)),('strategy',(-1,2,True,1.0))):
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):profile(**{key:value})
        self.assertEqual(profile(strategy=0)['strategy'],0)

    def test_existing_python_runtime_still_rejects_unqualified128_profile(self):
        m=Path(__file__).resolve().parent;config=json.loads((m/'evidence/nvfp4-config.json').read_bytes())['text_config']
        with self.assertRaises(ValueError):validate_bf16_low_batch_kv_program(True,1,4096,128,False,config,external_compressed_weights=True,joined_mlp_projections=True)

    def test_exact_derivation_preserves_all_attention_kernel_math(self):
        m=Path(__file__).resolve().parent;repo=m.parents[2];d=json.loads((m/'evidence/prefill128-attention-component-derivation.json').read_bytes())
        raw=(repo/d['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),d['source_sha256']);text=raw.decode()
        for step in d['replacements']:
            self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
        actual=(m/d['target']).read_bytes();self.assertEqual(actual,text.encode());self.assertEqual(hashlib.sha256(actual).hexdigest(),d['target_sha256'])
        self.assertNotIn('GarnetQwen38BF16LowBatchTiledScoresEnabled',text)

if __name__=='__main__':unittest.main()
