from pathlib import Path
import hashlib,json,unittest
from . import full_chunk_stage_vendor as original
from . import full_chunk_stage_vendor_optional_fields as fixed
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class OptionalFieldsTests(unittest.TestCase):
    def test_exact_reader_only_derivation_preserves_every_numeric_predicate(self):
        r=json.loads((M/'evidence/full-chunk-stage-vendor-optional-fields-derivation.json').read_bytes());raw=(M/r['source']).read_bytes();self.assertEqual(sha(raw),r['source_sha256'])
        for c in r['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
        self.assertEqual(raw,(M/r['target']).read_bytes());self.assertEqual(sha(raw),r['target_sha256']);self.assertTrue(r['numeric_primary_state_vector_token_readonly_repeat_array_predicates_unchanged']);self.assertFalse(r['new_GPU_run'])
    def test_actual_omitted_optional_fields_shape_is_accepted_without_inventing_defaults(self):
        raw='[cublasLt][12368][Info] cuBLASLt v13.4 device #0:\n[cublasLt][12368][Trace][cublasLtMatmul] A=0X1954200000 Adesc=[type=R_8F_E4M3 rows=5120 cols=12288 ld=5120] B=0X1D79A00000 Bdesc=[type=R_8F_E4M3 rows=5120 cols=32 ld=5120] C=0X1D7A600000 Cdesc=[type=R_16BF rows=12288 cols=32 ld=12288] D=0X1D7A600000 Ddesc=[type=R_16BF rows=12288 cols=32 ld=12288] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T] algo=[algoId=67 tile=MATMUL_TILE_64x32 stages=MATMUL_STAGES_64x5 ctaSwizzling=1 customOption=18] workSpace=0X1D79A28000 workSpaceSizeInBytes=4194304 beta=0 outOfPlace=0 stream=0X2\n'
        with self.assertRaises(AssertionError):original.parse_qg(raw,32)
        result=fixed.parse_qg(raw,32);self.assertEqual(result['matmul_trace_calls'],1);self.assertEqual(result['unique_printed_configurations'],[dict(algoId='67',tile='MATMUL_TILE_64x32',stages='MATMUL_STAGES_64x5',ctaSwizzling='1',customOption='18')]);self.assertTrue(result['unprinted_attributes_not_assumed'])
        for text in (raw.replace('algoId=67 ',''),raw.replace('tile=MATMUL_TILE_64x32 ',''),raw.replace('stages=MATMUL_STAGES_64x5 ',''),raw.replace('R_16BF','R_32F'),raw.replace('cols=12288','cols=1024')):
            with self.assertRaises(AssertionError):fixed.parse_qg(text,32)
