import unittest
from .fa3_projection_vendor_log import parse
def fixture():
    lines=['[time][cublasLt][7][Info][cublasLtCreate] cuBLASLt v13.4 device #0:']
    for t in (32,128):
        for weight in ('0X1000','0X2000'):
            d=f'Adesc=[type=R_8F_E4M3 rows=5120 cols=1024 ld=5120] Bdesc=[type=R_8F_E4M3 rows=5120 cols={t} ld=5120] Cdesc=[type=R_16BF rows=1024 cols={t} ld=1024] Ddesc=[type=R_16BF rows=1024 cols={t} ld=1024]'
            lines+=['[time][cublasLt][7][Api][cublasLtMatmulAlgoGetHeuristic] '+d+' maxWorkspaceSizeinBytes=4194304','[time][cublasLt][7][Info][cublasLtMatmulAlgoGetHeuristic] heuristicResults=[35,35]',f'[time][cublasLt][7][Trace][cublasLtMatmul] A={weight} '+d+' computeType=COMPUTE_32F scaleType=R_32F transa=OP_T algo=[algoId=35 tile=MATMUL_TILE_64x64 stages=MATMUL_STAGES_64x'+('6 reductionScheme=REDUCTION_SCHEME_COMPUTE_TYPE numSplitsK=7 ctaSwizzling=1' if t==32 else '4 reductionScheme=REDUCTION_SCHEME_COMPUTE_TYPE numSplitsK=5')+'] workSpaceSizeInBytes=4194304 beta=0 outOfPlace=0']
    return '\n'.join(lines)
class Controls(unittest.TestCase):
    def test_typed_parameters_and_unprinted_attributes_are_not_invented(self):
        q=parse(fixture());self.assertEqual(len(q['calls']),4);a=q['profiles']['32']['unique_printed_configurations'][0];b=q['profiles']['128']['unique_printed_configurations'][0];self.assertEqual((a['numSplitsK'],b['numSplitsK']),(7,5));self.assertNotIn('ctaSwizzling',b);self.assertNotIn('CUSTOM_OPTION',a)
    def test_descriptor_candidate_query_sequence_and_process_mutations_fail(self):
        source=fixture()
        for a,b in [('cols=1024 ld=5120','cols=1023 ld=5120'),('cols=128 ld=5120','cols=127 ld=5120'),('beta=0','beta=1'),('heuristicResults=[35,35]','heuristicResults=[67,35]'),('maxWorkspaceSizeinBytes=4194304','maxWorkspaceSizeinBytes=0')]:
            with self.assertRaises(AssertionError):parse(source.replace(a,b,1))
        with self.assertRaises(AssertionError):parse(source+'\n[time][cublasLt][8][Api][other]')
if __name__=='__main__':unittest.main()
