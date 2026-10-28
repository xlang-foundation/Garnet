from pathlib import Path
import copy,hashlib,struct,tempfile,unittest
from .b4_k35_cache52 import key_layout,capture_key,checked_cache,compare_cache,primary_parity,analyze,CACHE_BYTES
from .bf16_float_reference_state import expand_bf16_carrier
from . import test_b4_full4k_selection52 as fixtures
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def layout():
    rows=[];offset=0
    for i in range(64):
        full=i%4==3;a,b=(2097152,2097152) if full else (40960,786432);rows.append(dict(index=i,kind='full_attention' if full else 'linear_attention',offset=offset,state1_width=a,state2_width=b));offset+=a+b
    return rows

class Cache(unittest.TestCase):
    def test_capture_reads_both_blocks_twice_selected_row_and_links_component70(self):
        raw=bytearray(CACHE_BYTES);raw[-2:]=struct.pack('<H',0x3f80);raw=bytes(raw)
        class IO:
            def __init__(self):self.starts=[]
            def zeros(self,shape):self.shape=shape;return object()
            def slice_into(self,state,start,part):self.starts.append(start);self.start=start
            def row(self,part,row):assert row==2;return raw[(self.start-55894016)*4:(self.start-55894016)*4+4194304]
            def finite_f32_bytes(self,value):return value
        component=dict(layer=35,kind='full_attention',role=1,logical_words=4194304,row_sha256=['a'*64,'b'*64,sha(expand_bf16_carrier(raw)),'c'*64]);io=IO()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);entry=capture_key(io,object(),layout(),4,2,out,'actual52',component)
            self.assertEqual(io.shape,[4,1048576]);self.assertEqual(io.starts,[55894016,56942592]*2)
            self.assertEqual(checked_cache(entry,component,4,2,lambda n:(out/n).read_bytes()),raw)
            for mutation in ('hash','truncated','component'):
                e=copy.deepcopy(entry);comp=copy.deepcopy(component);read=lambda n:(out/n).read_bytes()
                if mutation=='hash':e['sha256']='f'*64
                elif mutation=='component':comp['row_sha256'][2]='f'*64
                else:read=lambda n:raw[:-2]
                with self.assertRaises(AssertionError):checked_cache(e,comp,4,2,read)
    def test_all_heads_slots_and_last_dimension_including_prompt_and_inactive(self):
        a=bytes(CACHE_BYTES);b=bytearray(a)
        for head,slot,dim,code in ((0,5,0,0x8000),(3,3981,255,0x3f80),(2,4095,255,0x3f81)):
            index=(head*4096+slot)*256+dim;b[index*2:index*2+2]=struct.pack('<H',code)
        q=compare_cache(a,bytes(b));self.assertEqual(q['different_BF16_values'],3);self.assertEqual(q['different_by_head'],[1,0,1,1]);self.assertEqual(q['different_prompt_values'],1);self.assertEqual(q['different_decode_values'],1);self.assertEqual(q['different_inactive_values'],1);self.assertEqual(q['earliest_scheduled_decode_writer_step'],14);self.assertEqual(q['different_by_slot'][4095],1)
    def test_nonfinite_cache_bad_geometry_or_wrong_selected_row_rejected(self):
        raw=bytearray(CACHE_BYTES);raw[-2:]=struct.pack('<H',0x7fc0)
        with self.assertRaises(AssertionError):compare_cache(bytes(CACHE_BYTES),bytes(raw))
        for mutation in ('offset','width'):
            rows=layout();rows[35]['offset' if mutation=='offset' else 'state1_width']+=1
            with self.assertRaises(AssertionError):key_layout(rows)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError):capture_key(None,None,layout(),4,1,Path(folder),'bad',{})
    def test_primary_change_in_any_candidate_row_rejected_before_cache_interpretation(self):
        s,key,r,c,f,*_=fixtures.Selection52().fixture();c['mode']='candidate';c['library_sha256']='a'*64;old=copy.deepcopy(c)
        self.assertEqual(primary_parity(c,old,lambda n:f[n],lambda n:f[n])['complete_captures'],5)
        c['rows'][0]['post_step_states'][0]['capture']['logical']['components'][127]['row_sha256'][0]='f'*64
        with self.assertRaises(AssertionError):primary_parity(c,old,lambda n:f[n],lambda n:f[n])
    def test_integrated_raw_cache_links_to_both_primary_component70_witnesses(self):
        s,key,r,c,f,*_=fixtures.Selection52().fixture();a=bytes(CACHE_BYTES);b=bytearray(a);index=(3*4096+3981)*256+255;b[index*2:index*2+2]=struct.pack('<H',0x3f80);b=bytes(b)
        for mode,report,raw,position in (('reference',r,a,0),('candidate',c,b,2)):
            report['mode']=mode;report['library_sha256']='a'*64
            digest=sha(expand_bf16_carrier(raw))
            for capture in (report['rows'][0]['final_state'],report['rows'][0]['post_step_states'][1]['capture']):capture['logical']['components'][70]['row_sha256'][position]=digest
        oldr,oldc=copy.deepcopy(r),copy.deepcopy(c)
        for mode,report,raw,position in (('reference',r,a,0),('candidate',c,b,2)):
            name=mode+'-key35.bf16carrier';f[name]=raw;report['rows'][0]['key35_cache52']=dict(file=name,bytes=CACHE_BYTES,sha256=sha(raw),repeat_sha256=[sha(raw)]*2,logical_sha256=sha(expand_bf16_carrier(raw)),physical_row=position,geometry=key_layout(layout()),storage='little_endian_two_BF16_per_F32_carrier',step=52)
        q=analyze(s,key,r,c,oldr,oldc,lambda k,n:f[n],lambda n:f[n],lambda k,n:f[n],lambda n:f[n]);self.assertEqual(q['cache_comparison']['different_BF16_values'],1);self.assertEqual(q['cache_comparison']['earliest_scheduled_decode_writer_step'],14);self.assertFalse(q['MTP_quality_HTTP_or_performance_qualified'])
        name=c['rows'][0]['key35_cache52']['file'];f[name]=a
        with self.assertRaises(AssertionError):analyze(s,key,r,c,oldr,oldc,lambda k,n:f[n],lambda n:f[n],lambda k,n:f[n],lambda n:f[n])
if __name__=='__main__':unittest.main()
