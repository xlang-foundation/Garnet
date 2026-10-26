"""Synthetic observer-log faults only, never hardware results."""
import unittest
from .gdn_z_algorithm_log import replay_log


def synthetic(mode):
    rows=['DEVICE NVIDIA GeForce RTX 5090 SM120 CUBLASLT_VERSION 130400',
          'CUBLASLT_DLL D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll']
    for profile,t in (('four-m32',32),('m128',128)):
        for repeat in (0,1):
            for start in range(0,128,t):
                original='01'*64 if t==32 else '02'*64
                actual='01'*64 if t==32 or mode=='shared' else original
                rows += [f'CONTROL profile={profile} repeat={repeat} start={start}',f'OBSERVE m={t} original_workspace=4194304',f'DEFAULT_SELECTION id_query=0 written=4 id={1 if t==32 else 2} opaque={original}']
                if t==128 and mode=='shared':rows += ['M32_ALGORITHM_CHECK_M128 status=0 state=0 required_workspace=5000000 supplied_workspace=8388608']
                rows += [f'ACTUAL_SELECTION id_query=0 written=4 id={1 if t==32 or mode=="shared" else 2} opaque={actual}',f'ACTUAL_MATMUL m={t} status=0 workspace={8388608 if t==128 and mode=="shared" else 4194304}']
            rows += [f'COMPLETE_OUTPUT profile={profile} repeat={repeat} changed_words=0 total_words=786432']
    rows += ['REAL_GDN_Z_'+('SHARED_M32' if mode=='shared' else 'BOTH_DEFAULT_PROFILES')+'_PASSED']
    return ('\n'.join(rows)+'\n').encode()


class AlgorithmLogTests(unittest.TestCase):
    def test_complete_observation_and_intervention(self):
        for mode in ('control','shared'):
            self.assertEqual(replay_log(synthetic(mode),mode)['actual_matmul_calls'],10)

    def test_missing_call_over_workspace_and_algorithm_substitution(self):
        raw=synthetic('shared')
        for old,new in ((b'required_workspace=5000000',b'required_workspace=9000000'),
                        (b'ACTUAL_SELECTION id_query=0 written=4 id=1 opaque='+b'01'*64,b'ACTUAL_SELECTION id_query=0 written=4 id=1 opaque='+b'03'*64),
                        (b'ACTUAL_MATMUL m=128 status=0 workspace=8388608',b'ACTUAL_MATMUL m=128 status=1 workspace=8388608'),
                        (b'CONTROL profile=m128 repeat=1 start=0',b'CONTROL profile=m128 repeat=1 start=32'),
                        (b'changed_words=0 total_words=786432',b'changed_words=1 total_words=786432')):
            with self.assertRaises(AssertionError):replay_log(raw.replace(old,new,1),'shared')


if __name__=='__main__':unittest.main()
