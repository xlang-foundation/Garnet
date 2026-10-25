"""CPU token-coded I/O fixture; no native/GPU/memory/performance admission."""
import unittest
from .paged_prompt_rows import PagedPromptRows

H=5120
class Tensor:
    def __init__(self,shape,values):self.shape=tuple(shape);self.values=tuple(values)

class IO:
    def __init__(self):self.calls=[];self.fail=False
    def embedding(self,ids):self.calls.append(('embedding',len(ids)));return Tensor([len(ids),H],[('E',i) for i in ids])
    def storage_view(self,value,shape):
        self.calls.append(('view',tuple(shape)))
        count=1
        for n in shape:count*=n
        assert count==len(value.values)*H
        return Tensor(shape,value.values)
    def slice(self,value,start,shape):
        self.calls.append(('slice',start,tuple(shape)));assert value.shape==(1,64*H) and start%H==0 and shape[1]%H==0
        return Tensor(shape,value.values[start//H:(start+shape[1])//H])
    def pack_prompt(self,embedding,visuals,starts):
        self.calls.append(('pack',len(embedding),embedding[0].shape[0]))
        if self.fail:raise RuntimeError('explicit fake pack failure')
        count=embedding[0].shape[0];rows=[]
        for e,v,start in zip(embedding,visuals,starts):
            assert e.shape==(count,H);values=list(e.values)
            if v is not None:
                assert v.shape==(1,len(v.values),H) and 0<=start<=count-len(v.values)
                values[start:start+len(v.values)]=v.values
            else:assert start==-1
            rows.append(values)
        return [Tensor([len(rows),H],[row[t] for row in rows]) for t in range(count)]

class PagedPromptControls(unittest.TestCase):
    def test_b8_4k_target_slices_and_shifted_mtp_lookups(self):
        io=IO();ids=[[1000*b+t for t in range(3968)] for b in range(8)];source=PagedPromptRows(io,ids)
        self.assertEqual(io.calls,[])
        for start in range(0,3968,32):
            block=source[start:start+32]
            self.assertEqual([v.values for v in block],[tuple(('E',ids[b][t]) for b in range(8)) for t in range(start,start+32)])
            for t in range(start,min(start+32,3967)):self.assertEqual(source[t+1].values,tuple(('E',ids[b][t+1]) for b in range(8)))
        self.assertEqual(source.stats()['pack_calls'],31)
        self.assertLessEqual(source.stats()['peak_cached_packed_logical_bytes'],2*128*8*H*4)
        self.assertTrue(all(c[1]<=128 for c in io.calls if c[0]=='embedding'))
    def test_images_cross_page_boundaries_and_different_row_lengths(self):
        io=IO();ids=[list(range(180)),list(range(80)),list(range(64)),list(range(159))]
        visuals=[Tensor([1,64,H],[('V',b,t) for t in range(64)]) if b!=1 else None for b in range(4)];starts=[111,-1,0,95]
        source=PagedPromptRows(io,ids,visuals,starts,page_size=32)
        for t in range(180):
            expected=[]
            for b in range(4):expected.append(('V',b,t-starts[b]) if visuals[b] is not None and starts[b]<=t<starts[b]+64 else ('E',ids[b][t] if t<len(ids[b]) else 0))
            self.assertEqual(source[t].values,tuple(expected))
        self.assertLessEqual(len(source.pages),2)
    def test_negative_index_small_cross_page_slice_and_empty_slice(self):
        source=PagedPromptRows(IO(),[list(range(65))],page_size=32)
        self.assertEqual(source[-1].values,(('E',64),));self.assertEqual([x.values for x in source[30:34]],[(('E',t),) for t in range(30,34)])
        self.assertEqual(source[4:4],[])
        for key in (65,-66):
            with self.assertRaises(IndexError):source[key]
        for key in (True,1.0,'1'):
            with self.assertRaises(TypeError):source[key]
        for key in (slice(None,None,2),slice(None,None,-1),slice(0,64),slice(None,None,True)):
            with self.assertRaises(ValueError):source[key]
    def test_failure_does_not_cache_partial_page_or_return_stale_rows(self):
        io=IO();source=PagedPromptRows(io,[list(range(100))],page_size=32)
        source[0];source[32];io.fail=True
        with self.assertRaises(RuntimeError):source[64]
        self.assertNotIn(2,source.pages);io.fail=False;self.assertEqual(source[64].values,(('E',64),))
    def test_metadata_rejected_before_io_and_ids_snapshotted(self):
        for args,kw in [(([[True]],),{}),(([[248320]],),{}),(([[1]]*3,),{}),(([[1]],),{'page_size':True}),(([[1]],),{'visual_starts':[0]}),(([[1]],),{'visual_rows':[object()],'visual_starts':[0]})]:
            io=IO()
            with self.subTest(args=args,kw=kw),self.assertRaises(ValueError):PagedPromptRows(io,*args,**kw)
            self.assertEqual(io.calls,[])
        io=IO();ids=[[1,2]];source=PagedPromptRows(io,ids);ids[0][0]=99;self.assertEqual(source[0].values,(('E',1),))
    def test_close_releases_cache_and_visual_references(self):
        source=PagedPromptRows(IO(),[list(range(64))],[Tensor([1,64,H],[('V',0,t) for t in range(64)])],[0]);source[0];source.close();source.close()
        self.assertEqual(source.pages,{});self.assertEqual(source.visuals,());self.assertIsNone(source.io)
        for read in (lambda:len(source),lambda:source[0],source.stats):
            with self.assertRaises(ValueError):read()

if __name__=='__main__':unittest.main()
