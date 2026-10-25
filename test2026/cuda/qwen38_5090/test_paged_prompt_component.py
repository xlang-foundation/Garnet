"""Synthetic capture corruption controls; no native or GPU executions."""
import copy,hashlib,struct,tempfile,unittest
from pathlib import Path
from .paged_prompt_component import ROW_BYTES,HIDDEN,case_specs,token_rows,visual_row_bytes,expected_events,replay_case,RowCapture,embedding_oracle

def fixture():
    spec=case_specs()[0];f={'ids':list(range(16))};oracle={t:struct.pack('<f',t/16)*HIDDEN for t in range(16)}
    selected=token_rows(spec,f['ids']);blobs={};events=[]
    for kind,t in expected_events(spec):
        start=spec['visual_starts'][0];raw=visual_row_bytes(0,0,t-start) if start<=t<start+64 else oracle[selected[0][t]]
        digest=hashlib.sha256(raw).hexdigest();blobs[digest+'.f32']=raw;events.append({'kind':kind,'token':t,'rows':[digest]})
    record={'spec':spec,'closed':True,'post_close_rejected':True,'first_page_evicted':True,'read_events':len(events),
            'cache_stats':{'physical_batch':1,'tokens':191,'lengths':[191],'page_size':32,'cached_pages':[4,5],'peak_cached_packed_logical_bytes':2*32*ROW_BYTES}}
    return record,events,blobs,f,oracle,spec

class PagedComponentControls(unittest.TestCase):
    def test_bf16_byte_oracle_rejects_updated_hash_of_wrong_float_bytes(self):
        bf16=struct.pack('<H',0x3f80)*HIDDEN;f32=struct.pack('<f',1.)*HIDDEN
        files={'a.bf16':bf16,'a.f32':f32};item={'bf16':'a.bf16','f32':'a.f32','bf16_sha256':hashlib.sha256(bf16).hexdigest(),'f32_sha256':hashlib.sha256(f32).hexdigest()}
        f={'status':'checkpoint_embedding_rows_prepared_cpu_only','shape':[248320,HIDDEN],'ids':[0]*16,'rows':{'0':item}}
        self.assertEqual(embedding_oracle(f,files.__getitem__),{0:f32})
        files['a.f32']=struct.pack('<f',2.)*HIDDEN;item['f32_sha256']=hashlib.sha256(files['a.f32']).hexdigest()
        with self.assertRaises(AssertionError):embedding_oracle(f,files.__getitem__)
    def test_matrix_contains_full_b8_and_visual_epoch_permutation(self):
        specs=case_specs();self.assertEqual(len(specs),8);self.assertEqual(specs[-1]['lengths'],[3968]*8)
        self.assertEqual(len(expected_events(specs[-1])),7937);self.assertEqual(specs[-2]['row_ids'],[3,4,5,6,7,0,1,2])
    def test_synthetic_lossless_raw_reconstruction(self):
        record,events,blobs,f,oracle,spec=fixture();r=replay_case(record,events,blobs.__getitem__,f,oracle,spec)
        self.assertTrue(r['all_observed_bytes_equal_independent_oracle']);self.assertEqual(r['read_events'],383);self.assertFalse(r['performance_admitted'])
    def test_actual_blob_corruption_rejected_even_with_updated_digest(self):
        record,events,blobs,f,oracle,spec=fixture();digest=events[0]['rows'][0];raw=bytearray(blobs[digest+'.f32']);raw[-1]^=1;new=hashlib.sha256(raw).hexdigest();blobs[new+'.f32']=bytes(raw)
        events[0]['rows'][0]=new
        with self.assertRaises(AssertionError):replay_case(record,events,blobs.__getitem__,f,oracle,spec)
    def test_missing_shift_wrong_index_stale_visual_or_false_close_rejected(self):
        for kind in ('missing','index','visual','close','stats'):
            record,events,blobs,f,oracle,spec=fixture()
            if kind=='missing':events.pop()
            elif kind=='index':events[33]['token']+=1
            elif kind=='visual':next(v for v in events if v['kind']=='target' and v['token']==95)['rows']=events[0]['rows'][:]
            elif kind=='close':record['post_close_rejected']=False
            else:record['cache_stats']['peak_cached_packed_logical_bytes']+=ROW_BYTES
            with self.subTest(kind=kind),self.assertRaises(AssertionError):replay_case(record,events,blobs.__getitem__,f,oracle,spec)
    def test_capture_keeps_actual_row_bytes_once_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=RowCapture(Path(tmp)/'rows');a=struct.pack('<f',1.)*HIDDEN;b=struct.pack('<f',2.)*HIDDEN
            first=c.record(a+b+a,3);second=c.record(b+a,2)
            self.assertEqual(second,[first[1],first[0]]);self.assertEqual(len(c.rows),2)
            self.assertEqual(b''.join((c.directory/(s+'.f32')).read_bytes() for s in first),a+b+a)
            with self.assertRaises(ValueError):c.record(a[:-1],1)

if __name__=='__main__':unittest.main()
