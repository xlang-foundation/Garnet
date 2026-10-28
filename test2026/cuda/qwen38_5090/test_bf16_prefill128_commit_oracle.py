"""CPU-only codec/rounding/layout controls for the isolated commit study."""
import struct,unittest
import numpy as np
from .bf16_prefill128_commit_oracle import MAGIC,GUARD,INITIAL,decode,round_bf16,fixture_word,fixture_words,expected_state

class Controls(unittest.TestCase):
    def test_lossless_repeated_block_decode_and_malformed_runs(self):
        block=list(range(32));raw=MAGIC+struct.pack('<Q',96)+struct.pack('<I32I',3,*block)
        self.assertEqual(decode(raw,96).tolist(),block*3)
        for bad in (raw[:-1],raw+b'x',b'BADMAGIC'+raw[8:],raw[:16]+struct.pack('<I32I',0,*block),
                    raw[:16]+struct.pack('<I32I',4,*block)):
            with self.subTest(raw=bad[:20]),self.assertRaises(ValueError):decode(bad,96)
        with self.assertRaises(ValueError):decode(raw,64)

    def test_independent_integer_bf16_ties_and_signed_zero(self):
        for source,result in ((0x3f808000,0x3f80),(0x3f818000,0x3f82),(0xbf808000,0xbf80),(0xbf818000,0xbf82),
                              (0x00008000,0),(0x80008000,0x8000),(0,0),(0x80000000,0x8000)):
            self.assertEqual(round_bf16(source),result)
        for value in (-1,0x100000000,True):
            with self.assertRaises(ValueError):round_bf16(value)

    def test_vector_fixture_matches_scalar_and_disambiguates_rows(self):
        actual=fixture_words(1024)
        self.assertEqual(actual.tolist(),[fixture_word(i) for i in range(1024)])
        self.assertFalse(np.array_equal(actual[:32],actual[256:288]))
        self.assertTrue(np.isfinite(actual.view(np.float32)).all())

    def test_logical_layout_oracle_preserves_unwritten_slots_and_guards(self):
        layout={'canonical_state_width':4+4194304,'layers':[
            {'kind':'linear_attention','offset':0,'target_offset':0,'state1_width':2,'state2_width':2},
            {'kind':'full_attention','offset':4,'target_offset':4,'state1_width':2097152,'state2_width':2097152}]}
        target=fixture_words(4+262144);actual=expected_state(layout,4095,1,target)
        self.assertTrue(np.all(actual[:32]==GUARD));self.assertTrue(np.all(actual[-32:]==GUARD))
        self.assertTrue(np.array_equal(actual[32:36],target[:4]))
        halves=actual[36:-32].view(np.uint16).reshape(2,4,4096,256)
        for which in range(2):
            for head in range(4):
                expected=[round_bf16(int(v)) for v in target[4+which*131072+head*256:4+which*131072+(head+1)*256]]
                self.assertEqual(halves[which,head,4095].tolist(),expected)
        self.assertTrue(np.all(halves[:,:,:4095,:]==0x3f00))
        empty=expected_state(layout,4096,0,target);self.assertTrue(np.all(empty[32:-32]==INITIAL))
        for start,count in ((-1,1),(4096,1),(3969,128),(0,129)):
            with self.assertRaises(ValueError):expected_state(layout,start,count,target)

if __name__=='__main__':unittest.main()
