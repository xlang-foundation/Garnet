import unittest
import numpy as np
from .analyze_fa3_projection_value_cache_link import projected_cache
class Controls(unittest.TestCase):
    def test_independent_token_head_dimension_indices_and_inactive_tail(self):
        source=np.zeros(128*1024,dtype='<u4');positions=[(0,0,0),(11,1,226),(127,3,255)]
        for token,head,dim in positions:source[token*1024+head*256+dim]=0x3fc00000
        actual=np.frombuffer(projected_cache(source.tobytes()),dtype='<u2')
        self.assertEqual(np.flatnonzero(actual).tolist(),sorted((head*4096+token)*256+dim for token,head,dim in positions))
        for token,head,dim in positions:self.assertEqual(int(actual[(head*4096+token)*256+dim]),0x3fc0)
    def test_non_BF16_representable_nonfinite_or_wrong_length_rejected(self):
        for value in (0x3f800001,0x7f800000,0x7fc00000):
            source=np.zeros(128*1024,dtype='<u4');source[1]=value
            with self.assertRaises(AssertionError):projected_cache(source.tobytes())
        with self.assertRaises(AssertionError):projected_cache(bytes(4))
