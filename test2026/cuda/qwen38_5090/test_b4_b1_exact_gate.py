import copy,hashlib,struct,unittest
from .b4_b1_exact_gate import compare_capture,require_ids

class Gate(unittest.TestCase):
    def fixture(self):
        files={};refs=[]
        def capture(batch,offset):
            hashes=[hashlib.sha256(str(r+offset).encode()).hexdigest() for r in range(batch)]
            logical=dict(physical_words_per_row=106823680,logical_words_per_row=173932544,reference_bf16_roundtrip_checked=False,slots=[32]*batch,physical_row_sha256=hashes[:],logical_row_sha256=hashes[:],components=[])
            for i in range(128):
                layer=i//2;role=i%2+1;full=layer%4==3
                logical['components'].append(dict(layer=layer,kind='full_attention' if full else 'linear_attention',role=role,logical_words=4194304 if full else (40960 if role==1 else 786432),row_sha256=hashes[:]))
            vectors={}
            for key in ('hidden','normalized'):
                name='b'+str(batch)+'r'+str(offset)+'-'+key+'.f32';raw=b''.join(struct.pack('<f',float(r+offset))*5120 for r in range(batch));files[name]=raw;vectors[key]=dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            return dict(logical=logical,vectors=vectors)
        actual=capture(4,0);refs=[capture(1,r) for r in range(4)];return actual,refs,files
    def replay(self,a,b,f):return compare_capture(a,b,f.__getitem__,lambda r,n:f[n])
    def test_exact_and_complete_component_corruption(self):
        a,b,f=self.fixture();self.assertTrue(self.replay(a,b,f)['all_exact'])
        a['logical']['components'][127]['row_sha256'][3]='f'*64
        q=self.replay(a,b,f);self.assertFalse(q['all_exact']);self.assertEqual(q['rows'][3]['different_component_indices'],[127])
    def test_row_permutation_and_raw_vector_bits(self):
        a,b,f=self.fixture();self.assertFalse(self.replay(a,[b[1],b[0],b[2],b[3]],f)['all_exact'])
        entry=a['vectors']['hidden'];raw=bytearray(f[entry['file']]);raw[3]=128;f[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest()
        self.assertFalse(self.replay(a,b,f)['all_exact']) # -0 and +0 must differ.
    def test_nonfinite_and_missing_component_rejected(self):
        a,b,f=self.fixture();bad=copy.deepcopy(a);bad['logical']['components'].pop()
        with self.assertRaises(AssertionError):self.replay(bad,b,f)
        e=a['vectors']['normalized'];raw=struct.pack('<f',float('nan'))+f[e['file']][4:];f[e['file']]=raw;e['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError):self.replay(a,b,f)
    def test_ids_and_slot_corruption(self):
        ids=[[1,2],[3,4],[5,6],[7,8]];self.assertTrue(require_ids(ids,copy.deepcopy(ids),2))
        bad=copy.deepcopy(ids);bad[3][1]=9
        with self.assertRaises(AssertionError):require_ids(ids,bad,2)
        a,b,f=self.fixture();a['logical']['slots'][2]=31;self.assertFalse(self.replay(a,b,f)['all_exact'])

if __name__=='__main__':unittest.main()
