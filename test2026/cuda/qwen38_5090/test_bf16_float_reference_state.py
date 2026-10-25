"""CPU byte-representation controls; no model or CUDA imports."""
import struct,unittest
from test2026.cuda.qwen38_5090.bf16_float_reference_state import expand_bf16_carrier,require_bf16_roundtrip


class ReferenceStateBytes(unittest.TestCase):
    def test_known_values_and_signed_zero_preserved(self):
        raw=struct.pack('<6H',0x3f80,0xc000,0,0x8000,0x3f81,0x3f7f)
        expected=struct.pack('<6I',0x3f800000,0xc0000000,0,0x80000000,0x3f810000,0x3f7f0000)
        self.assertEqual(expand_bf16_carrier(raw),expected);self.assertEqual(require_bf16_roundtrip(expected),expected)

    def test_carrier_boundaries_do_not_swap_values(self):
        words=[0x3f00+i for i in range(1024)];raw=struct.pack('<1024H',*words)
        expected=struct.pack('<1024I',*[w<<16 for w in words])
        self.assertEqual(expand_bf16_carrier(raw),expected)
        self.assertEqual(b''.join(expand_bf16_carrier(raw[i:i+68]) for i in range(0,len(raw),68)),expected)

    def test_precision_corruption_rejected(self):
        for bit in (0,1,7,8,15):
            with self.subTest(bit=bit),self.assertRaises(ValueError):require_bf16_roundtrip(struct.pack('<I',0x3f800000|(1<<bit)))

    def test_malformed_representations_rejected(self):
        for raw in (b'\0',b'\0\0',b'\0\0\0',bytearray(4),None):
            for function in (expand_bf16_carrier,require_bf16_roundtrip):
                with self.subTest(function=function,raw=raw),self.assertRaises(ValueError):function(raw)

    def test_reference_chunk_keeps_full_state_copy_without_compact_commit(self):
        from test2026.cuda.qwen38_5090.test_kv_delta_transactions import decoder,controls
        d=decoder();d.kv_delta_state=False;d.bf16_kv_state=False;d.bf16_kv_float_reference=True;d.delta_commit=None
        self.assertEqual(d.chunk_hidden(*controls()),[7]*4)
        self.assertIn('long',d.events);self.assertNotIn('short',d.events)
        self.assertIn(('slice',('target',20480,'canonical')),d.events)
        self.assertFalse(any(isinstance(e,tuple) and e[0]=='commit' for e in d.events))
        self.assertEqual(d.slots,[1]*4)

    def test_reference_holes_reject_before_io(self):
        from test2026.cuda.qwen38_5090.test_kv_delta_transactions import decoder
        d=decoder();d.kv_delta_state=False;d.bf16_kv_state=False;d.bf16_kv_float_reference=True;d.delta_commit=None
        with self.assertRaisesRegex(ValueError,'contiguous'):
            d.chunk_hidden([['x']*4]*3,[[1]*4,[0]*4,[1]*4],[[[0,0,0]]*3 for _ in range(4)])
        self.assertEqual(d.events,[]);self.assertFalse(d.poisoned)

if __name__=='__main__':unittest.main()
