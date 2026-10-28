"""Execute private DSL/binding/output contracts with CPU symbols only."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,types,unittest
from unittest.mock import patch
from .test_decode35_operand_program import isolated as original_isolated,Symbol
from .test_decode_boundary_program import IO
M=Path(__file__).resolve().parent

def isolated():
    old,base=original_isolated();parent='test2026.cuda.qwen38_5090.'
    bounded=types.ModuleType(parent+'bounded_step45_boundary_program');bounded.BoundedStep45BoundaryProgram=old.BoundedStep45BoundaryProgram;bounded.CUDAPackedDecoderProgram=base
    packed=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_packed_program')
    for n in ('T','Qwen35CUDAChunk','flat_concat','concat','packed_view'):setattr(packed,n,getattr(old,n))
    spec=importlib.util.spec_from_file_location(parent+'isolated_rowwise_k_norm_program',M/'rowwise_k_norm_program.py');module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{bounded.__name__:bounded,packed.__name__:packed}):spec.loader.exec_module(module)
    return module,base

def method(raw,classname):
    text=raw.decode().replace('\r\n','\n');tree=ast.parse(text);c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname);f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_forward');return '\n'.join(text.splitlines()[f.lineno-1:f.end_lineno])+'\n\n'

class Program(unittest.TestCase):
    def test_ordinary_outputs_remain_primary_only(self):
        module,_=isolated();p=module.RowwiseKDecodeNormProgram('checkpoint',{},IO(),4,4096,4)
        self.assertEqual(p.output_width,17);self.assertFalse(hasattr(p,'boundary_buffers'));self.assertFalse(hasattr(p,'operand_buffers'))
        out=p.run([object()]*5)
        for row in range(4):self.assertEqual(list(out.values[row]),list(range(row*10000000,row*10000000+17)))
        p.close()

    def test_existing_boundary_prefix_and_all64_rows_preserved(self):
        module,_=isolated();p=module.RowwiseKDecodeNormBoundaryProgram('checkpoint',{},IO(),4,4096,4)
        self.assertEqual(p.output_width,17+64*20480);self.assertFalse(hasattr(p,'operand_buffers'))
        out=p.run([object()]*5);self.assertEqual(p.boundary_runs,1)
        for row in range(4):
            base=row*10000000;self.assertEqual(list(out.values[row]),list(range(base,base+17)))
            for layer,value in enumerate(p.boundary_buffers):self.assertEqual(value.values[row][0],base+17+layer*20480)
        p.close();self.assertEqual(p.boundary_buffers,[])

    def test_both_classes_reject_other_profiles_before_allocation(self):
        module,_=isolated()
        for cls in (module.RowwiseKDecodeNormProgram,module.RowwiseKDecodeNormBoundaryProgram):
            for batch,capacity,size in ((1,4096,4),(2,4096,4),(8,4096,4),(True,4096,4),(4.0,4096,4),(4,512,4),(4,4096,32)):
                with self.subTest(cls=cls.__name__,shape=(batch,capacity,size)),self.assertRaises(ValueError):cls('checkpoint',{},IO(),batch,capacity,size)

    def test_only_B4T4_binding_changes_and_restores_on_error(self):
        module,base=isolated();parent=types.ModuleType('xModel.qwen35.vl_0_8b');decoder=types.ModuleType(parent.__name__+'.cuda_packed_decoder');parent.cuda_packed_decoder=decoder;decoder.CUDAPackedDecoderProgram=base
        for observe in (False,True):
            def factory():
                selected=module.RowwiseKDecodeNormBoundaryProgram if observe else module.RowwiseKDecodeNormProgram
                self.assertIsInstance(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,4),selected)
                self.assertIs(type(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,32)),base)
                self.assertIs(type(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,4)),base)
                raise RuntimeError('intentional')
            with patch.dict(sys.modules,{parent.__name__:parent,decoder.__name__:decoder}):
                with self.assertRaisesRegex(RuntimeError,'intentional'):module.construct_with_rowwise_k_decode_norm_program(factory,observe_boundaries=observe)
            self.assertIs(decoder.CUDAPackedDecoderProgram,base)
        with self.assertRaises(TypeError):module.construct_with_rowwise_k_decode_norm_program(lambda:None,observe_boundaries=1)

    def test_actual_DSL_marker_all16_layers_and_exclusive_cache_modes(self):
        module,_=isolated();config={'num_attention_heads':24,'num_key_value_heads':4,'head_dim':256,'rms_norm_eps':1e-6,'rope_parameters':{'partial_rotary_factor':.25,'rope_theta':1e7,'mrope_section':[11,11,10]}}
        def call(layer=35,kind='full_attention',**changes):
            index=layer if type(layer) is int and layer in range(3,64,4) else 35;key=(index-3)//4*6676480+2482176
            options=dict(fused_attention=True,external_compressed_weights={'packed':[]},joined_mlp_projections=True,canonical_kv_state=Symbol(),canonical_kv_layout={'cache_batch_stride':106823680,'key_offset':key,'value_offset':key+2097152},bf16_canonical_kv=True);options.update(changes)
            return module.Qwen35RowwiseKDecodeNormCUDAChunk(Symbol(),None,None,Symbol(),Symbol(),Symbol(),config,layer,kind,**options)
        for layer in range(3,64,4):
            result=call(layer);self.assertEqual(result.kind,'qwen35_full_decoder_chunk');self.assertEqual(result.metadata['rowwise_full_decode_k_norm'],1)
            self.assertEqual(result.metadata['layer_prefix'],'model.language_model.layers.'+str(layer));self.assertNotIn('witness_full_decode35_stages',result.metadata)
            self.assertEqual(result.metadata['canonical_key_offset'],(layer-3)//4*6676480+2482176)
        for layer in (0,4,35.0,True,64):
            with self.assertRaises(ValueError):call(layer)
        for changes in ({'bf16_canonical_kv':False},{'bf16_low_batch_canonical_kv':True},{'bf16_prefill128_canonical_kv':True},{'bf16_kv_float_reference':True},{'canonical_kv_state':None},{'external_compressed_weights':None},{'external_weights':{'weight':[]}},{'fused_attention':False},{'joined_mlp_projections':False}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):call(**changes)
        with self.assertRaises(ValueError):call(kind='linear_attention')

    def test_exact_frozen_chunk_and_both_forward_derivations(self):
        d=json.loads((M/'evidence/rowwise-k-norm-program-derivation.json').read_bytes());raw=(M/d['target']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),d['target_sha256'])
        text=raw.decode();chunk=text[text.index('@T.fusion('):text.index('class _RowwiseKShape:')].rstrip()+'\n'
        for c in reversed(d['chunk_changes']):self.assertEqual(chunk.count(c['new']),c['count']);chunk=chunk.replace(c['new'],c['old'])
        original=(M/d['chunk_source']).read_bytes();self.assertEqual(hashlib.sha256(original).hexdigest(),d['chunk_source_sha256']);original=original.decode().replace('\r\n','\n');self.assertEqual(chunk,original[original.index('@T.fusion('):].rstrip()+'\n')
        for f in d['forward_derivations']:
            current=method(raw,f['target_class'])
            for c in reversed(f['changes']):self.assertEqual(current.count(c['new']),c['count']);current=current.replace(c['new'],c['old'])
            original=(M/f['source']).read_bytes();self.assertEqual(hashlib.sha256(original).hexdigest(),f['source_sha256']);self.assertEqual(current,method(original,f['source_class']))

if __name__=='__main__':unittest.main()
