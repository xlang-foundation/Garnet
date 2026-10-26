"""Synthetic native-context log faults only; never hardware evidence."""
import unittest
from .test_gdn_z_algorithm_log import synthetic
from .fp8_z128_native_control import replay_algorithms


def native_synthetic():
    raw=synthetic('control').replace(b'02'*64,b'01'*64)
    context=b'NATIVE_Z128_CONTEXT algorithm=67 device=0 required_workspace=0 blas_workspace=4194304 opaque='+b'01'*64+b'\nNATIVE_Z128_REJECTIONS count=15 unexpected_matmul_calls=0\n'
    return raw.replace(b'CONTROL profile=m128 repeat=0 start=0',context+b'CONTROL profile=m128 repeat=0 start=0').replace(b'REAL_GDN_Z_BOTH_DEFAULT_PROFILES_PASSED',b'REAL_GDN_Z_TYPED_NATIVE_Z128_PASSED')


class NativeControlFaults(unittest.TestCase):
    def test_native_context_and_original_control(self):
        self.assertEqual(replay_algorithms(native_synthetic(),'native','01'*64)['native_context']['invalid_argument_rejections'],15)
        self.assertIsNone(replay_algorithms(synthetic('control'),'control','01'*64)['native_context'])

    def test_changed_context_algorithm_workspace_badargs_and_missing_call_fail(self):
        raw=native_synthetic()
        for old,new in ((b'algorithm=67 device=0',b'algorithm=68 device=0'),(b'required_workspace=0 blas_workspace',b'required_workspace=9000000 blas_workspace'),(b'blas_workspace=4194304',b'blas_workspace=8388608'),(b'count=15 unexpected_matmul_calls=0',b'count=15 unexpected_matmul_calls=1'),(b'opaque='+b'01'*64,b'opaque='+b'03'*64),(b'CONTROL profile=m128 repeat=1 start=0',b'CONTROL profile=m128 repeat=1 start=32')):
            with self.assertRaises(AssertionError):replay_algorithms(raw.replace(old,new,1),'native','01'*64)

if __name__=='__main__':unittest.main()
