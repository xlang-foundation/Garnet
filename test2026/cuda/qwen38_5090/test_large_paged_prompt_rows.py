"""CPU source/visual/lifetime/bounds controls; no native or speed qualification."""
import unittest
from .large_paged_prompt_rows import LargePagedPromptRows
from .paged_prompt_rows import PagedPromptRows
from .test_paged_prompt_rows import IO,Tensor,H


class LargePromptPageControls(unittest.TestCase):
    def test_full4k_target_slices_and_shifted_draft_inputs(self):
        io=IO();ids=[[b*4000+t for t in range(3968)] for b in range(4)]
        source=LargePagedPromptRows(io,ids);self.assertEqual(io.calls,[])
        for start in range(0,3968,32):
            self.assertEqual([v.values for v in source[start:start+32]],
                             [tuple(('E',ids[b][t]) for b in range(4)) for t in range(start,start+32)])
            for t in range(start,min(start+32,3967)):
                self.assertEqual(source[t+1].values,tuple(('E',ids[b][t+1]) for b in range(4)))
        self.assertEqual(source.stats()['pack_calls'],2)
        self.assertLessEqual(source.stats()['peak_cached_packed_logical_bytes'],2*2048*4*H*4)
        self.assertFalse(source.stats()['native_byte_or_model_performance_qualified'])

    def test_cross_page_visual_and_short_rows_and_held_views(self):
        io=IO();ids=[list(range(n)) for n in (2200,2100,64,2180)]
        visuals=[Tensor([1,64,H],[('V',b,t) for t in range(64)]) if b!=1 else None for b in range(4)]
        starts=[2020,-1,0,2040];source=LargePagedPromptRows(io,ids,visuals,starts)
        held=source[2025]
        for t in list(range(2016,2112))+[0,63,64,2199]:
            expected=tuple(('V',b,t-starts[b]) if visuals[b] is not None and starts[b]<=t<starts[b]+64
                           else ('E',ids[b][t] if t<len(ids[b]) else 0) for b in range(4))
            self.assertEqual(source[t].values,expected)
        before=held.values;source.close();self.assertEqual(held.values,before)
        with self.assertRaises(ValueError):source[0]

    def test_native_capacity_bound_and_old_policy_stays_unchanged(self):
        for batch,size in ((8,2048),(4,True),(4,128),(4,2048.0),(3,1024)):
            io=IO()
            with self.subTest(batch=batch,size=size),self.assertRaises(ValueError):
                LargePagedPromptRows(io,[[1]*64 for _ in range(batch)],page_size=size)
            self.assertEqual(io.calls,[])
        source=LargePagedPromptRows(IO(),[[1]*64]*8,page_size=1024)
        self.assertEqual(source.page_size,1024)
        self.assertEqual(PagedPromptRows(IO(),[[1]*64]).page_size,128)
        with self.assertRaises(ValueError):PagedPromptRows(IO(),[[1]*64],page_size=2048)

    def test_lru_failure_preserves_live_rows_and_does_not_cache_partial_page(self):
        io=IO();source=LargePagedPromptRows(io,[list(range(4096))],page_size=1024)
        held=source[0];source[1024];io.fail=True
        with self.assertRaises(RuntimeError):source[2048]
        self.assertNotIn(2,source.pages);self.assertEqual(held.values,(('E',0),))
        io.fail=False;self.assertEqual(source[2048].values,(('E',2048),));self.assertLessEqual(len(source.pages),2)
        source.close();source.close()


if __name__=='__main__':unittest.main()
