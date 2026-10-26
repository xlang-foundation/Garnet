import copy,types,unittest
from pathlib import Path
from .test_b4_cache_mode_characterization import fixture
from .test_kv_delta_graph_contract import functions
from .b4_cache_mode_characterization import compare_capture
from .bf16_batch8_external_reference import capture_exact

class Controls(unittest.TestCase):
    def test_owner_archives_partial_parent_and_rejects_unbound_parent(self):
        from .run_b4_packed_cache_characterization import owned_common_inputs,ROOT
        p={'partial_parent':{'job':'closed-parent'},'parents':{'closed-parent':{}},'new_files':{'Garnet/owned-method':'x'}}
        paths=owned_common_inputs(p,ROOT/'prep',ROOT/'receipt')
        self.assertIn(ROOT/'results/closed-parent-partial-verified.json',paths)
        self.assertIn(ROOT/'results/closed-parent-verified.json',paths)
        self.assertIn(ROOT/'Garnet/owned-method',paths)
        p['parents']={}
        with self.assertRaises(AssertionError):owned_common_inputs(p,ROOT/'prep',ROOT/'receipt')
    def data(self):
        a,fa=fixture(4);b,fb=fixture(8)
        for case,batch in ((a,4),(b,8)):
            for phase in ('reset_state','prefill_state','final_state'):
                logical=case[phase]['logical'];logical.update(physical_words_per_row=106823680,reference_bf16_roundtrip_checked=False,physical_row_sha256=['a'*64]*batch,slots=[0]*batch)
        ns={'compare_capture':compare_capture};functions(Path(__file__).with_name('probe_b4_packed_cache_characterization.py').read_bytes(),('empty_packed_exact',),ns)
        return a['reset_state'],b['reset_state'],fa,fb,ns['empty_packed_exact']
    def test_actual_function_routes_packed_empty_without_relaxing_full_only_gate(self):
        a,b,fa,fb,check=self.data()
        with self.assertRaisesRegex(ValueError,'Unqualified same-precision state representation'):capture_exact(a,b,[0,1,2,3],fa.__getitem__,fb.__getitem__)
        self.assertTrue(check(a,b,fa.__getitem__,fb.__getitem__)['all_states_and_vectors_exact'])
    def test_nonempty_wrong_dialect_or_actual_difference_rejected(self):
        a,b,fa,fb,check=self.data()
        for key,value in (('slots',[1]*4),('physical_words_per_row',173932544),('reference_bf16_roundtrip_checked',True),('logical_row_sha256',['b'*64]*4)):
            bad=copy.deepcopy(a);bad['logical'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):check(bad,b,fa.__getitem__,fb.__getitem__)

if __name__=='__main__':unittest.main()
