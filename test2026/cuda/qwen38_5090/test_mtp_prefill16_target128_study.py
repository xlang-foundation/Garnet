"""CPU ownership/binding/exception controls, not draft/model/GPU measurements."""
from pathlib import Path
from types import SimpleNamespace
import ast,json,unittest

class FakeGraph:
    def enable_cuda_graph(self,value):self.cuda=value
    def stats_json(self):return json.dumps({'allocator_tracking_enabled':True,'device_output_bytes':20,'trt_allocator_live_bytes':30})
class FakeProgram:
    def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;self.graph=FakeGraph();self.closed=False
    def close(self):self.closed=True
class FakeIO:
    def __init__(self):self.zeroed=[]
    def zeros(self,shape):return {'shape':shape}
    def zero(self,value):self.zeroed.append(value)
    def tensor_bytes(self,value):
        n=4
        for v in value['shape']:n*=v
        return n
class FakeBase:
    constructions=0
    def __init__(self,decoder,checkpoint,stops,*args,**kwargs):
        FakeBase.constructions+=1;FakeBase.last=self;self.decoder=decoder;self.io=decoder.io;self.hidden_size=5120;self.size=4;self.program=FakeProgram('narrow');self.base_arguments=(args,kwargs);self.key=object();self.value=object();self.last_hidden=object();self.reset_count=0;self.base_closed=False
        for name in self._swapped_buffers:setattr(self,name,object())
    def _prefill_hook(self,*args):
        self.during=(self.size,self.program,self.target,self.key,self.value,self.last_hidden)
        if getattr(self,'reenter',False):return self._prefill_hook(*args)
        if getattr(self,'fail',False):raise RuntimeError('synthetic prefill failure')
        return 'target prediction'
    def reset(self):self.reset_count+=1
    def memory_info(self):return {'owned_speculative_bytes':100}
    def close(self):self.base_closed=True

def load(program=FakeProgram):
    path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_target128_study.py';tree=ast.parse(path.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));scope={'json':json,'CUDASpeculativeSession':FakeBase,'CUDAMTPProgram':program};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope);return scope[cls.name]
def decoder(**changes):
    values={'batch':1,'capacity':4096,'prefill_chunk_size':128,'chunk_size':4,'bf16_kv_low_batch_state':True,'bf16_kv_prefill128_study':True,'config':{'hidden_size':5120,'num_attention_heads':24,'num_key_value_heads':4,'head_dim':256},'projection_workspace_bytes':0,'io':FakeIO()};values.update(changes);return SimpleNamespace(**values)

class Prefill16Target128Controls(unittest.TestCase):
    def test_wrong_target_profiles_fail_before_any_base_resource_construction(self):
        cls=load();before=FakeBase.constructions
        for values in ({'batch':4},{'batch':True},{'capacity':8192},{'prefill_chunk_size':16},{'chunk_size':8},{'bf16_kv_low_batch_state':False}):
            with self.subTest(values=values),self.assertRaises(ValueError):cls(decoder(**values),'fixture',[])
        self.assertEqual(FakeBase.constructions,before)
    def test_temporary_wide_bindings_share_cache_then_restore_narrow_decode(self):
        cls=load();value=cls(decoder(),'fixture',[]);before={n:getattr(value,n) for n in value._swapped_buffers};narrow=value.program;shared=(value.key,value.value,value.last_hidden)
        self.assertEqual(value._prefill_hook([],[],[]),'target prediction');self.assertEqual(value.during[:2],(16,value.prefill16_program));self.assertEqual(value.during[3:],shared)
        self.assertEqual({n:getattr(value,n) for n in value._swapped_buffers},before);self.assertIs(value.program,narrow);self.assertEqual(value.size,4);self.assertFalse(value._prefill16_active)
        self.assertEqual(value.base_arguments,((3,),{'head_candidates':0,'prefix_replay':True,'draft_chunk_size':4,'draft_builder_workspace_bytes':128<<20}))
        value.close()
    def test_failure_and_reentrance_restore_every_binding(self):
        for kind in ('fail','reenter'):
            value=load()(decoder(),'fixture',[]);before={n:getattr(value,n) for n in value._swapped_buffers};program=value.program;setattr(value,kind,True)
            with self.subTest(kind=kind),self.assertRaises((RuntimeError,ValueError)):value._prefill_hook([],[],[])
            self.assertEqual(value.size,4);self.assertIs(value.program,program);self.assertFalse(value._prefill16_active);self.assertEqual({n:getattr(value,n) for n in value._swapped_buffers},before);value.close()
    def test_extra_engine_and_buffers_are_counted_reset_and_closed(self):
        value=load()(decoder(),'fixture',[]);graph=value.prefill16_program;info=value.memory_info();self.assertGreater(info['prefill16_buffers_bytes'],0);self.assertEqual(info['owned_speculative_bytes'],150+info['prefill16_buffers_bytes']);self.assertTrue(info['extra_draft_engine_weights_duplicated']);self.assertFalse(info['study_gpu_qualified'])
        value.reset();self.assertEqual(len(value.io.zeroed),6);value.close();self.assertTrue(graph.closed);self.assertTrue(value.base_closed);self.assertEqual(value.prefill16_buffers,{})
    def test_decode_policy_changes_rejected_before_resource_construction(self):
        cls=load();before=FakeBase.constructions
        for options in ({'max_draft_tokens':7},{'head_candidates':128},{'prefix_replay':False},{'draft_chunk_size':16},{'draft_builder_workspace_bytes':256<<20},{'head_candidates':False}):
            with self.subTest(options=options),self.assertRaises(ValueError):cls(decoder(),'fixture',[],**options)
        self.assertEqual(FakeBase.constructions,before)
    def test_extra_engine_construction_failure_closes_base_resources(self):
        def fail(*args,**kwargs):raise RuntimeError('synthetic extra engine failure')
        cls=load(fail)
        with self.assertRaisesRegex(RuntimeError,'extra engine failure'):cls(decoder(),'fixture',[])
        self.assertTrue(FakeBase.last.base_closed);self.assertEqual(FakeBase.last.prefill16_buffers,{})
    def test_old_profile_and_missing_explicit_new_optin_are_rejected(self):
        cls=load();before=FakeBase.constructions
        for changes in ({'prefill_chunk_size':32},{'bf16_kv_prefill128_study':False},{'bf16_kv_prefill128_study':1},{'bf16_kv_prefill128_study':None}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):cls(decoder(**changes),'fixture',[])
        self.assertEqual(FakeBase.constructions,before)

    def test_existing_t32_class_keeps_its_original_t128_rejection(self):
        from .test_mtp_prefill16_study import load as old_load
        with self.assertRaises(ValueError):old_load()(decoder(),'fixture',[])

if __name__=='__main__':unittest.main()
