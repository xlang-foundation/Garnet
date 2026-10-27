from pathlib import Path
import copy,hashlib,json,unittest
from unittest.mock import patch
from .full_chunk_stage_layout import STAGES
from . import full_chunk_stage_vendor as v
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

class VendorTests(unittest.TestCase):
    def test_logger_only_derivation_and_same_native_model_inventory(self):
        record=json.loads((M/'evidence/full-chunk-stage-vendor-derivation.json').read_bytes())
        for row in record['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for c in row['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        bat=(M/'test_full_chunk_stage_vendor.bat').read_text();self.assertIn('set CUBLASLT_LOG_LEVEL=5',bat);self.assertIn('set CUBLASLT_LOG_MASK=31',bat);self.assertIn('if not "%~4"=="1" exit /b 2',bat)
        owner=(M/'run_full_chunk_stage_vendor.py').read_text();self.assertIn("assert not (out/(phase+'-cublaslt.log')).exists()",owner);self.assertIn("('stages32',32,1),('stages128',128,1)",owner)

    def test_complete_raw_stage_parity_rejects_finite_byte_change_and_readonly_change(self):
        frames=[dict(start=0,count=128,arrays={n:n for n,w in STAGES})];a=dict(tokens=128,readonly_observations=['complete-state-controls']);b=copy.deepcopy(a)
        with patch.object(v,'frames',return_value=frames),patch.object(v,'array',side_effect=lambda value,read:read(value)):
            result=v.stage_parity(a,b,lambda n:b'\x00'*4,lambda n:b'\x00'*4);self.assertEqual(result['complete_stage_entries'],10);self.assertEqual(result['complete_F32_words'],6553600)
            with self.assertRaises(AssertionError):v.stage_parity(a,b,lambda n:b'\x00'*4,lambda n:b'\x01\x00\x00\x00' if n=='qg' else b'\x00'*4)
            b['readonly_observations']=['mutated'];
            with self.assertRaises(AssertionError):v.stage_parity(a,b,lambda n:b'\x00'*4,lambda n:b'\x00'*4)

    def test_vendor_parser_keeps_only_actual_target_traces_and_never_guesses_missing_attributes(self):
        header='[cublasLt][42][Info] cuBLASLt v13.4 device #0:\n'
        line='[cublasLt][42][Trace][cublasLtMatmul] A=0XABC Adesc=[type=R_8F_E4M3 rows=5120 cols=12288 ld=5120] Bdesc=[type=R_8F_E4M3 rows=5120 cols=32 ld=5120] Cdesc=[type=R_16BF rows=12288 cols=32 ld=12288] Ddesc=[type=R_16BF rows=12288 cols=32 ld=12288] computeType=COMPUTE_32F scaleType=R_32F transa=OP_T workSpaceSizeInBytes=4194304 beta=0 outOfPlace=0 algo=[algoId=67 tile=MATMUL_TILE_64x64 stages=MATMUL_STAGES_64x6 reductionScheme=NONE numSplitsK=1]\n'
        parsed=v.parse_qg(header+line,32);self.assertEqual(parsed['matmul_trace_calls'],1);self.assertNotIn('ctaSwizzling',parsed['calls'][0]['printed_algorithm'])
        for text in (header,header+line.replace('cols=12288','cols=1024'),header+line.replace('R_16BF','R_32F'),header.replace('v13.4','v13.3')+line):
            with self.assertRaises(AssertionError):v.parse_qg(text,32)
        row=dict(logical_shape=[12288,5120],packed_shape=[12288,1280],packed_dtype='INT32',packed_bytes=62914560,recipe='modelopt_fp8_w8a8',scale_bytes=0,scale_shape=[],joined_mlp=False,weight_scale=.5,input_scale=.25);v.weight_metadata(row)
        for k,x in (('logical_shape',[1024,5120]),('weight_scale',float('nan')),('input_scale',0),('recipe','int4')):
            with self.assertRaises(AssertionError):v.weight_metadata({**row,k:x})
