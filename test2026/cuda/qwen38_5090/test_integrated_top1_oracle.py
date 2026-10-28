"""Independent fixture guard and all original expected-output preservation."""
from pathlib import Path
import hashlib,json,unittest
from .integrated_top1_oracle import SHAPES,expected_rows,row_oracle

class IntegratedOracleTests(unittest.TestCase):
    def test_original_and_extended_expected_outputs(self):
        e=Path(__file__).with_name('evidence')
        old=json.loads((e/'parallel-top1-fixture-oracles.json').read_bytes())
        frozen=json.loads((e/'integrated-batch-top1-fixture-oracles.json').read_bytes())
        self.assertEqual(hashlib.sha256((e/'parallel-top1-fixture-oracles.json').read_bytes()).hexdigest(),frozen['original_fixture_sha256'])
        oldcases={(c['dtype'],c['rows'],c['vocab'],c['pattern']):(c['ids'],c['score_bits']) for c in old['cases']}
        keys=[]
        for c in frozen['cases']:
            key=tuple(c[n] for n in ('dtype','rows','vocab','pattern'));keys.append(key)
            expected=expected_rows(*key)
            self.assertEqual(expected,(c['ids'],c['score_bits']))
            if key in oldcases:self.assertEqual(expected,oldcases[key])
        self.assertEqual(keys,[(dtype,r,v,p) for dtype in ('fp32','bf16') for r,v in SHAPES for p in range(11)])
        self.assertEqual(len(keys),352)
        self.assertTrue(set(oldcases).issubset(keys))

    def test_row129_fallback_values(self):
        for dtype in ('fp32','bf16'):
            ids,bits=expected_rows(dtype,129,7,10)
            self.assertEqual(ids,[min((r*257)%7,6) for r in range(129)])
            self.assertEqual(bits,[0xc47a0000]*129)
            ids,bits=expected_rows(dtype,129,7,4)
            self.assertEqual(ids,[0]*129)
            self.assertEqual(bits,[0 if r%2==0 else 0x80000000 for r in range(129)])

    def test_reject_unqualified_shapes_and_types(self):
        for args in [('fp32',129,248320,0),('fp32',130,7,0),('fp32',True,7,0),('int4',129,7,0)]:
            with self.assertRaises(ValueError):expected_rows(*args)
        with self.assertRaises(ValueError):row_oracle(129,7,0,'fp32')

if __name__=='__main__':unittest.main()
