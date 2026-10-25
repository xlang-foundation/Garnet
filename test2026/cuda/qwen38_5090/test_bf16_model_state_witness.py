"""Small independent host byte arrays check complete state observation and guards."""
import copy,hashlib,struct,unittest
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness


class IO:
    def __init__(self,width):self.width=width;self.reads=[];self.truncate=False
    def zeros(self,shape):return [bytearray(shape[1]*4) for _ in range(shape[0])]
    def slice_into(self,state,offset,out):
        words=len(out[0])//4;self.reads.append((offset,words))
        for row in range(len(out)):out[row][:]=state[row][offset*4:(offset+words)*4]
    def row(self,part,row):return part[row]
    def finite_f32_bytes(self,row):return bytes(row)[:-1] if self.truncate else bytes(row)


def fixture(packed):
    rows=[];expanded=[]
    for row in range(2):
        gdn=struct.pack('<5I',0x3f800001+row,0x40000001,0x00000000,0x80000000,0x3f000001)
        cache=[0x3f80+row,0xc000,0,0x8000,0x3f00,0x3f80,0x4000,0x4040]
        full=gdn+struct.pack('<8I',*[v<<16 for v in cache]);expanded.append(full)
        rows.append(gdn+struct.pack('<8H',*cache) if packed else full)
    width=9 if packed else 13;k=2 if packed else 4
    layout=[{'index':0,'offset':0,'kind':'linear_attention','state1_width':2,'state2_width':3},
            {'index':1,'offset':5,'kind':'full_attention','state1_width':k,'state2_width':k}]
    return rows,expanded,width,layout


class ModelStateWitnessTests(unittest.TestCase):
    def test_logical_hash_matches_independent_expanded_arrays_for_every_chunk_boundary(self):
        results=[]
        for packed in (False,True):
            rows,expected,width,layout=fixture(packed);saved=copy.deepcopy(rows)
            for block in (1,2,3,4,1048576):
                result=logical_state_witness(IO(width),rows,layout,2,width,[41,104],packed_bf16=packed,block_words=block)
                self.assertEqual(result['logical_row_sha256'],[hashlib.sha256(r).hexdigest() for r in expected])
                self.assertEqual(result['physical_row_sha256'],[hashlib.sha256(r).hexdigest() for r in rows]);self.assertEqual(result['logical_words_per_row'],13)
                self.assertEqual(rows,saved);results.append(result['components'])
        self.assertTrue(all(r==results[0] for r in results))

    def test_invalid_or_gapped_layout_rejected_before_io(self):
        rows,_,width,layout=fixture(True)
        for key,value in [('offset',6),('index',3),('kind','other'),('state1_width',True),('state2_width',0)]:
            bad=copy.deepcopy(layout);bad[1][key]=value;io=IO(width)
            with self.subTest(key=key),self.assertRaises(ValueError):logical_state_witness(io,rows,bad,2,width,[0,0],packed_bf16=True)
            self.assertEqual(io.reads,[])

    def test_profile_and_complete_coverage_guards(self):
        rows,_,width,layout=fixture(True)
        for kwargs in ({'width':width+1},{'batch':True},{'packed_bf16':1},{'block_words':0},{'block_words':1048577},{'slots':[True,0]}):
            good=dict(batch=2,width=width,slots=[0,0],packed_bf16=True);good.update(kwargs);io=IO(width)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):logical_state_witness(io,rows,layout,**good)
            self.assertEqual(io.reads,[])

    def test_reference_cache_low_precision_violation_rejected_without_gdn_rounding(self):
        rows,_,width,layout=fixture(False);bad=list(rows);data=bytearray(bad[0]);data[5*4]|=1;bad[0]=bytes(data)
        with self.assertRaisesRegex(ValueError,'outside BF16'):logical_state_witness(IO(width),bad,layout,2,width,[0,0],packed_bf16=False)
        self.assertEqual(logical_state_witness(IO(width),rows,layout,2,width,[0,0],packed_bf16=False)['logical_words_per_row'],13)

    def test_truncated_reader_rejected(self):
        rows,_,width,layout=fixture(True);io=IO(width);io.truncate=True
        with self.assertRaisesRegex(ValueError,'truncated'):logical_state_witness(io,rows,layout,2,width,[0,0],packed_bf16=True)

if __name__=='__main__':unittest.main()
