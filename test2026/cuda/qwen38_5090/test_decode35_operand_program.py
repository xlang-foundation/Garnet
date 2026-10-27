"""CPU symbolic graph/row contracts; no native library is imported or executed."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types,unittest
from unittest.mock import patch
from .test_all_layer_decode_boundary_program import isolated as layer_isolated
from .test_decode_boundary_program import IO
M=Path(__file__).resolve().parent

class Symbol:
    def __init__(self,kind=None,metadata=None):self.kind=kind;self.metadata=metadata or {}
    def __mul__(self,other):return other if isinstance(other,Symbol) else self

def isolated():
    layers,_,base=layer_isolated()
    tensor=types.SimpleNamespace(fusion=lambda **kw:lambda f:f,binary_op=lambda name:Symbol(name),unary_op=lambda name,**kw:Symbol(name,kw))
    packed=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_packed_program');packed.CUDAPackedDecoderProgram=base;packed.T=tensor
    for name in ('Qwen35CUDAChunk','flat_concat','concat','packed_view'):setattr(packed,name,None)
    parent='test2026.cuda.qwen38_5090.'
    spec=importlib.util.spec_from_file_location(parent+'bounded_step45_boundary_program',M/'bounded_step45_boundary_program.py');bounded=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{parent+'all_layer_decode_boundary_program':layers,packed.__name__:packed}):spec.loader.exec_module(bounded)
    spec=importlib.util.spec_from_file_location(parent+'isolated_decode35_operand_program',M/'decode35_operand_program.py');module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{parent+'bounded_step45_boundary_program':bounded,packed.__name__:packed}):spec.loader.exec_module(module)
    return module,base

class Contract(unittest.TestCase):
    def test_all_primary_boundary_and_operand_rows_remain_separate_and_ordered(self):
        module,_=isolated()
        for batch in (1,4):
            p=module.Decode35OperandProgram('checkpoint',{},IO(),batch,4096,4)
            start=17+64*20480;self.assertEqual(p.output_width,start+106496);self.assertEqual(len(p.boundary_buffers),64)
            primary=p.run([object()]*5);self.assertEqual(p.boundary_runs,1)
            for row in range(batch):
                base=row*10000000;self.assertEqual(list(primary.values[row]),list(range(base,base+17)))
                for layer,buffer in enumerate(p.boundary_buffers):self.assertEqual(buffer.values[row][0],base+17+layer*20480)
                cursor=start
                for (name,width),(recorded_name,offset,recorded_width),buffer in zip(module.OPERANDS,p.operand_offsets,p.operand_buffers):
                    self.assertEqual((recorded_name,offset,recorded_width),(name,cursor,width))
                    self.assertEqual((len(buffer.values[row]),buffer.values[row][0],buffer.values[row][-1]),(width,base+cursor,base+cursor+width-1));cursor+=width
            p.close();self.assertEqual(p.operand_buffers,[]);self.assertEqual(p.operand_offsets,[]);self.assertEqual(p.boundary_buffers,[])
            with self.assertRaises(ValueError):p.run([object()]*5)

    def test_old_profile_guard_rejects_other_shapes(self):
        module,_=isolated()
        for b,c,t in ((2,4096,4),(8,4096,4),(4,512,4),(4,4096,32),(True,4096,4)):
            with self.assertRaises(ValueError):module.Decode35OperandProgram('checkpoint',{},IO(),b,c,t)

    def test_constructor_binding_restored_and_prefill_unchanged(self):
        module,base=isolated();parent=types.ModuleType('xModel.qwen35.vl_0_8b');decoder=types.ModuleType(parent.__name__+'.cuda_packed_decoder');parent.cuda_packed_decoder=decoder;decoder.CUDAPackedDecoderProgram=base
        def factory():
            self.assertIsInstance(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,4),module.Decode35OperandProgram)
            self.assertIsInstance(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,32),base)
            raise RuntimeError('intentional')
        with patch.dict(sys.modules,{parent.__name__:parent,decoder.__name__:decoder}):
            with self.assertRaisesRegex(RuntimeError,'intentional'):module.construct_with_decode35_operand_program(factory)
        self.assertIs(decoder.CUDAPackedDecoderProgram,base)

    def test_private_chunk_forwards_exact_native_marker_and_both_cache_modes(self):
        module,_=isolated();config={'num_attention_heads':24,'num_key_value_heads':4,'head_dim':256,'rms_norm_eps':1e-6,'rope_parameters':{'partial_rotary_factor':.25,'rope_theta':1e7,'mrope_section':[11,11,10]}}
        for low in (True,False):
            value=module.Qwen35Decode35CUDAChunk(Symbol(),None,None,Symbol(),Symbol(),Symbol(),config,35,'full_attention',fused_attention=True,external_compressed_weights={'packed':[]},joined_mlp_projections=True,canonical_kv_state=Symbol(),canonical_kv_layout={'cache_batch_stride':106823680,'key_offset':55894016,'value_offset':57991168},bf16_low_batch_canonical_kv=low,bf16_canonical_kv=not low)
            self.assertEqual(value.kind,'qwen35_full_decoder_chunk');self.assertEqual(value.metadata['witness_full_decode35_stages'],1)
            self.assertEqual(value.metadata['layer_prefix'],'model.language_model.layers.35');self.assertEqual(value.metadata['canonical_state_width'],106823680)
            self.assertEqual(value.metadata['canonical_bf16_low_batch_kv_overlay' if low else 'canonical_bf16_kv_overlay'],1)
        for layer,kind in ((3,'full_attention'),(35,'linear_attention'),(34,'full_attention')):
            with self.assertRaises(ValueError):module.Qwen35Decode35CUDAChunk(Symbol(),None,None,Symbol(),Symbol(),Symbol(),config,layer,kind)

    def test_exact_frozen_chunk_math_and_parent_forward_derivations(self):
        from .decode35_stage_source import sources
        _,candidate,_=sources();p=json.loads((M/'evidence/decode35-operand-program-derivation.json').read_bytes());raw=(M/p['target']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),p['target_sha256'])
        text=raw.decode();chunk=text[text.index('@T.fusion('):text.index('class Decode35OperandProgram')].rstrip()+'\n'
        for change in reversed(p['chunk_changes']):self.assertEqual(chunk.count(change['new']),change['count']);chunk=chunk.replace(change['new'],change['old'])
        original=candidate[p['chunk_source']].decode();self.assertEqual(chunk,original[original.index('@T.fusion('):].rstrip()+'\n')
        current=text[text.index('    def _forward(self):'):text.index('    def run(self,inputs):')]
        for change in reversed(p['forward_changes']):self.assertEqual(current.count(change['new']),change['count']);current=current.replace(change['new'],change['old'])
        original=(M/p['forward_source']).read_text(encoding='utf8');self.assertEqual(current,original[original.index('    def _forward(self):'):original.index('def construct_with_bounded_step45_boundary_program')])

if __name__=='__main__':unittest.main()
