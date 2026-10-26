import copy,unittest
from .z128_reconstruct_arguments import expected_initialization,check
class Controls(unittest.TestCase):
    def test_type_layout_transpose_and_width_mutations_fail(self):
        old=['cudaGetDevice','cudaGetDeviceProperties:0','cublasLtGetVersion','cublasLtCreate','cublasLtMatmulDescCreate:68:0','cublasLtMatmulDescSetAttribute:0:4:1','cublasLtMatmulDescSetAttribute:1:4:0','cublasLtMatrixLayoutCreate:28:5120:6144:5120','cublasLtMatrixLayoutCreate:28:5120:128:5120','cublasLtMatrixLayoutCreate:14:6144:128:6144','cublasLtMatrixLayoutCreate:28:5120:32:5120','cublasLtMatrixLayoutCreate:14:6144:32:6144']+['unused']*5
        expected=expected_initialization(old);self.assertEqual(len(expected),32);report={'mode':'success','calls':expected+['cleanup']};self.assertEqual(check(report,expected),32)
        for index in (5,7,8,10,12,13,20,21,28,31):
            bad=copy.deepcopy(report);bad['calls'][index]+='-wrong'
            with self.assertRaises(AssertionError):check(bad,expected)
        self.assertEqual(check({'mode':'fail_13','calls':expected[:13]+['cleanup']},expected),13)
        self.assertEqual(check({'mode':'unsupported','calls':[]},expected,True),0)
        with self.assertRaises(AssertionError):check({'mode':'unsupported','calls':['cudaGetDevice']},expected,True)
if __name__=='__main__':unittest.main()
