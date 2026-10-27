"""CPU binding, workspace, ownership and fail-closed geometry controls; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import ast,json,unittest
from . import test_mtp_prefill16_study as old
TRUE_FLAGS=('external_compressed_weights','joined_mlp_projections','host_embedding_lookup','retain_linear_inputs','register_delta_128','flat_concat','deferred_state_commit','shared_output_storage','packed_state_program')
FALSE_FLAGS=('bf16_kv_low_batch_state','bf16_kv_prefill128_study','embedded_projection_weights','join_decode_input_projections','kv_delta_state')
def decoder(**changes):
    values=dict(batch=4,capacity=4096,prefill_chunk_size=32,chunk_size=4,bf16_kv_state=True,config=dict(hidden_size=5120,num_attention_heads=24,num_key_value_heads=4,head_dim=256),projection_workspace_bytes=0,io=old.FakeIO(),**{k:True for k in TRUE_FLAGS},**{k:False for k in FALSE_FLAGS});values.update(changes);return SimpleNamespace(**values)
def load(program=old.FakeProgram):
    path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_b4_study.py';tree=ast.parse(path.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));scope=dict(json=json,CUDASpeculativeSession=old.FakeBase,CUDAMTPProgram=program);exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope);return scope[cls.name]
class B4Prefill16Controls(unittest.TestCase):
    def test_wrong_geometry_or_pin_geometry_fails_before_resources(self):
        cls=load();before=old.FakeBase.constructions
        for change in (dict(batch=1),dict(batch=True),dict(capacity=8192),dict(prefill_chunk_size=128),dict(prefill_chunk_size=16),dict(chunk_size=8),dict(bf16_kv_state=False),dict(bf16_kv_state=1),dict(config=dict(hidden_size=1024,num_attention_heads=24,num_key_value_heads=4,head_dim=256))):
            with self.subTest(change=change),self.assertRaises(ValueError):cls(decoder(**change),'fixture',[])
        self.assertEqual(old.FakeBase.constructions,before)
    def test_all_explicit_target_flags_required_without_truthy_substitutes(self):
        cls=load();before=old.FakeBase.constructions
        for flag in TRUE_FLAGS+FALSE_FLAGS:
            for value in (None,1,0,False if flag in TRUE_FLAGS else True):
                with self.subTest(flag=flag,value=value),self.assertRaises(ValueError):cls(decoder(**{flag:value}),'fixture',[])
        self.assertEqual(old.FakeBase.constructions,before)
    def test_pinned_decode_policy_and_distinct_allowed_prefill_workspace(self):
        value=load()(decoder(),'fixture',[])
        self.assertEqual(value.base_arguments,((3,),dict(head_candidates=0,prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=256<<20)))
        self.assertEqual(value.prefill16_program.args[2:5],(4,4096,16));self.assertEqual(value.prefill16_program.kwargs,dict(builder_workspace_bytes=128<<20))
        for name,buffer in value.prefill16_buffers.items():self.assertEqual(buffer['shape'][0:2],[4,16])
        value.close()
    def test_policy_variants_fail_before_any_base_allocation(self):
        cls=load();before=old.FakeBase.constructions
        for options in (dict(max_draft_tokens=7),dict(head_candidates=128),dict(head_candidates=False),dict(prefix_replay=False),dict(draft_chunk_size=16),dict(draft_builder_workspace_bytes=128<<20)):
            with self.subTest(options=options),self.assertRaises(ValueError):cls(decoder(),'fixture',[],**options)
        self.assertEqual(old.FakeBase.constructions,before)
    def test_temporary_B4_T16_bindings_share_existing_cache_and_restore_T4(self):
        value=load()(decoder(),'fixture',[]);before={n:getattr(value,n) for n in value._swapped_buffers};narrow=value.program;shared=(value.key,value.value,value.last_hidden)
        self.assertEqual(value._prefill_hook([],[],[]),'target prediction');self.assertEqual(value.during[:2],(16,value.prefill16_program));self.assertEqual(value.during[3:],shared);self.assertEqual({n:getattr(value,n) for n in value._swapped_buffers},before);self.assertIs(value.program,narrow);self.assertEqual(value.size,4);value.close()
    def test_failure_and_reentrancy_restore_all_bindings(self):
        for kind in ('fail','reenter'):
            value=load()(decoder(),'fixture',[]);before={n:getattr(value,n) for n in value._swapped_buffers};program=value.program;setattr(value,kind,True)
            with self.assertRaises((ValueError,RuntimeError)):value._prefill_hook([],[],[])
            self.assertEqual(value.size,4);self.assertIs(value.program,program);self.assertFalse(value._prefill16_active);self.assertEqual({n:getattr(value,n) for n in value._swapped_buffers},before);value.close()
    def test_extra_engine_and_buffers_counted_reset_and_closed(self):
        value=load()(decoder(),'fixture',[]);extra=value.prefill16_program;info=value.memory_info();self.assertEqual(info['owned_speculative_bytes'],150+info['prefill16_buffers_bytes']);self.assertEqual(info['physical_batch'],4);self.assertFalse(info['study_gpu_qualified']);self.assertFalse(info['study_performance_admitted']);self.assertTrue(info['extra_draft_engine_weights_duplicated']);value.reset();self.assertEqual(len(value.io.zeroed),6);value.close();self.assertTrue(extra.closed);self.assertTrue(value.base_closed);self.assertEqual(value.prefill16_buffers,{})
    def test_extra_build_failure_closes_base_and_old_B1_guard_unchanged(self):
        def fail(*args,**kwargs):raise RuntimeError('synthetic extra engine failure')
        with self.assertRaisesRegex(RuntimeError,'extra engine failure'):load(fail)(decoder(),'fixture',[])
        self.assertTrue(old.FakeBase.last.base_closed);self.assertEqual(old.FakeBase.last.prefill16_buffers,{})
        with self.assertRaises(ValueError):old.load()(decoder(),'fixture',[])
if __name__=='__main__':unittest.main()
