import copy,hashlib,struct,unittest
from .decode_projection_diagnostic import STAGES,OFFSETS,TOTAL,compare,replay
class Controls(unittest.TestCase):
    def fixture(self):
        files={};refs=[{} for _ in range(4)];actual={}
        for key,width in STAGES.items():
            values=[]
            for row in range(4):
                raw=struct.pack('<f',float(row))*width+b'\0'*(3*width*4);values.append(raw);name=key+'-r'+str(row)+'.f32';files[name]=raw;refs[row][key]=dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            raw=b''.join(values);name=key+'-b4.f32';files[name]=raw;actual[key]=dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return actual,refs,files
    def run_compare(self,a,b,f):return compare(a,b,f.__getitem__,lambda r,n:f[n])
    def test_complete_geometry_and_exact(self):
        a,b,f=self.fixture();self.assertTrue(self.run_compare(a,b,f)['all_bytes_exact']);self.assertEqual(OFFSETS,dict(qkv=4*5120+40960+786432,z=4*5120+40960+786432+4*10240,beta=4*5120+40960+786432+4*(10240+6144),a=TOTAL-4*48))
    def test_active_signedzero_and_inactive_corruption(self):
        for index in (0,10240):
            a,b,f=self.fixture();e=a['qkv'];raw=bytearray(f[e['file']]);raw[4*index+3]=128;f[e['file']]=bytes(raw);e['sha256']=hashlib.sha256(raw).hexdigest();q=self.run_compare(a,b,f);row=q['stages']['qkv'][0];self.assertFalse(q['all_bytes_exact']);self.assertEqual(row['active_different_words'],int(index==0));self.assertEqual(row['inactive_different_words'],int(index!=0))
    def test_nonfinite_truncated_and_swapped_rows(self):
        a,b,f=self.fixture();self.assertFalse(self.run_compare(a,[b[1],b[0],b[2],b[3]],f)['all_bytes_exact'])
        e=a['z'];raw=struct.pack('<f',float('inf'))+f[e['file']][4:];f[e['file']]=raw;e['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError):self.run_compare(a,b,f)
    def test_collection_tokens_repeat_and_preserved_inputs(self):
        actual,refs,files=self.fixture();tokens=[[r+s for r in range(4)] for s in range(5)]
        def report(batch):
            cases=[]
            for s in range(5):
                for rows in ([[r] for r in range(4)] if batch==1 else [[0,1,2,3]]):
                    arrays=refs[rows[0]] if batch==1 else actual;cases.append(dict(step=s,physical_rows=rows,tokens=[tokens[s][r] for r in rows],input=arrays['input'],input_after_sha256=[arrays['input']['sha256']]*2,repeats=[copy.deepcopy(arrays),copy.deepcopy(arrays)]))
            return dict(status='actual_decode_projection_arrays_collected',physical_batch=batch,chunk=4,token_inputs=tokens,cases=cases,full_model_or_performance_qualified=False)
        b,c=report(1),report(4);self.assertTrue(replay(b,c,files.__getitem__,files.__getitem__,tokens)['all_projection_bytes_exact'])
        bad=copy.deepcopy(c);bad['cases'][0]['tokens'][0]=99
        with self.assertRaises(AssertionError):replay(b,bad,files.__getitem__,files.__getitem__,tokens)
        bad=copy.deepcopy(b);bad['cases'][0]['input_after_sha256'][1]='f'*64
        with self.assertRaises(AssertionError):replay(bad,c,files.__getitem__,files.__getitem__,tokens)
        bad=copy.deepcopy(b);entry=bad['cases'][0]['repeats'][1]['normalized'];raw=bytearray(files[entry['file']]);raw[3]=128;files['changed-normalized.f32']=bytes(raw);entry.update(file='changed-normalized.f32',sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(AssertionError,'Repeat instability'):replay(bad,c,files.__getitem__,files.__getitem__,tokens)
        a,b,f=self.fixture();e=a['beta'];f[e['file']]=f[e['file']][:-4];e['bytes']-=4;e['sha256']=hashlib.sha256(f[e['file']]).hexdigest()
        with self.assertRaises(AssertionError):self.run_compare(a,b,f)
if __name__=='__main__':unittest.main()
