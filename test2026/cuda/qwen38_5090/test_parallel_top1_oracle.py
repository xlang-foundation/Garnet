"""Control the independent replay's edge semantics before GPU execution."""
import struct,unittest
from test2026.cuda.qwen38_5090.parallel_top1_oracle import row_oracle,expected_rows


class ParallelTop1OracleTests(unittest.TestCase):
    def test_initial_sentinel_and_ignored_nonfinite_inputs(self):
        for dtype in ('fp32','bf16'):
            for pattern in (1,2,3,9):
                with self.subTest(dtype=dtype,pattern=pattern):self.assertEqual(row_oracle(0,257,pattern,dtype),(0,0xff7fffff))

    def test_ties_across_chunk_boundary(self):
        for dtype in ('fp32','bf16'):self.assertEqual(row_oracle(0,248320,7,dtype),(1023,0x45800000))

    def test_signed_zero_keeps_first_original_value(self):
        for dtype in ('fp32','bf16'):
            self.assertEqual(row_oracle(0,7,4,dtype),(0,0))
            self.assertEqual(row_oracle(1,7,4,dtype),(0,0x80000000))

    def test_bf16_rounding_changes_subnormal_tie(self):
        self.assertEqual(row_oracle(0,7,8,'fp32'),(6,1))
        self.assertEqual(row_oracle(0,7,8,'bf16'),(0,0x80000000))
        self.assertEqual(row_oracle(0,1,8,'bf16'),(0,0))

    def test_positive_infinity_and_single_finite_value(self):
        for dtype in ('fp32','bf16'):
            self.assertEqual(row_oracle(0,248320,5,dtype),(1024,0x7f800000))
            self.assertEqual(row_oracle(0,257,6,dtype),(256,0xbf800000))

    def test_seeded_fixture_matches_separate_serial_python_scan(self):
        for row in (0,1,7):
            best=-3.4028234663852886e38;index=0
            for col in range(1025):
                seed=((col*1664525+1013904223)&0xffffffff)^((row*2246822519)&0xffffffff)
                value=((seed%2047)-1023)/64
                if value>best:best=value;index=col
            bits=struct.unpack('<I',struct.pack('<f',best))[0]
            self.assertEqual(row_oracle(row,1025,0,'fp32'),(index,bits))

    def test_unqualified_shapes_and_types_rejected(self):
        for args in ((True,7,0,'fp32'),(0,7.0,4,'fp32'),(0,0,0,'fp32'),(0,7,11,'fp32'),(0,7,0,'fp16'),(128,7,0,'bf16')):
            with self.subTest(args=args),self.assertRaises(ValueError):row_oracle(*args)
        with self.assertRaises(ValueError):expected_rows('fp32',3,248320,0)


if __name__=='__main__':unittest.main()
