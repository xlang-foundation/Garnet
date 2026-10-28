"""CPU-only original-prefix, binding restoration and source derivation controls."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types,unittest
from unittest.mock import patch
from .decode_boundary_layout import LAYERS,validate_profile,offsets,value_cache_word_offset
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class Buffer:
    def __init__(self,shape,values=None):self.shape=shape;self.values=values
class IO:
    def zeros(self,shape):return Buffer(shape)
    def slice_into(self,source,start,target):target.values=[row[start:start+target.shape[1]] for row in source.values]
class Base:
    def __init__(self,*args,**kwargs):self.io=args[2];self.batch=args[3];self.graph=True;self.output_width=17;self.hidden_width=4*5120
    def run(self,inputs):
        if self.graph is None or len(inputs)!=5:raise ValueError('closed or wrong arity')
        self.seen=inputs;return Buffer([self.batch,self.output_width],[list(range(r*1000000,r*1000000+self.output_width)) for r in range(self.batch)])
    def close(self):self.graph=None
def isolated():
    module=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_packed_program');module.CUDAPackedDecoderProgram=Base
    for name in ('T','Qwen35CUDAChunk','flat_concat','concat','packed_view'):setattr(module,name,None)
    spec=importlib.util.spec_from_file_location('isolated_decode_boundary_program',M/'decode_boundary_program.py');result=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{module.__name__:module}):spec.loader.exec_module(result)
    return result
class Contract(unittest.TestCase):
    def test_exact_original_math_and_original_guard_retained(self):
        p=json.loads((M/'evidence/decode-boundary-program-derivation.json').read_bytes());raw=(M/p['source']).read_bytes();self.assertEqual(sha(raw),p['source_sha256']);original=raw
        for c in p['changes']:self.assertEqual(raw.count(c['old'].encode()),c['count']);raw=raw.replace(c['old'].encode(),c['new'].encode())
        self.assertEqual(raw,(M/p['target']).read_bytes());self.assertEqual(sha(raw),p['target_sha256'])
        text=raw.decode();forward=text[text.index('    def _forward(self):'):text.index('    def run(self,inputs):')];self.assertEqual(sha(forward.encode()),p['original_forward_sha256']);self.assertTrue(p['math_forward_body_byte_identical']);self.assertIn('args[3]!=1',original.decode());self.assertIn('args[5] not in (32,128)',original.decode())
    def test_profile_and_offsets_reject_other_physical_shapes(self):
        for b in (1,4):validate_profile(b,4096,4)
        for b,c,t in ((0,4096,4),(2,4096,4),(8,4096,4),(1,512,4),(4,4096,32),(True,4096,4),(1,4096,4.)):
            with self.assertRaises(ValueError):validate_profile(b,c,t)
        rows=offsets(17,20480);self.assertEqual([r[0] for r in rows],list(LAYERS));self.assertEqual(rows[-1][1]+rows[-1][2],17+8*20480)
        for width in (0,3,5120,81920):
            with self.assertRaises(ValueError):offsets(17,width)
        layout=[{} for _ in range(64)]
        for layer in (3,11):
            base=sum(4194304 if i%4==3 else 827392 for i in range(layer));layout[layer]=dict(index=layer,kind='full_attention',offset=base,state1_width=2097152,state2_width=2097152)
            self.assertEqual(value_cache_word_offset(layout,layer,3,4095),base+4194304-128)
            with self.assertRaises(ValueError):value_cache_word_offset(layout,layer,0,4096)
        with self.assertRaises(ValueError):value_cache_word_offset(layout,7,0,0)
    def test_both_batches_return_all_original_primary_rows_and_separate_suffix(self):
        cls=isolated().DecodeBoundaryProgram
        for batch in (1,4):
            p=cls('checkpoint',{},IO(),batch,4096,4);self.assertEqual(p.output_width,17+8*20480);inputs=[object() for _ in range(5)];before=list(inputs);primary=p.run(inputs)
            self.assertEqual(primary.shape,[batch,17]);self.assertEqual(inputs,before)
            for row in range(batch):
                self.assertEqual(primary.values[row],list(range(row*1000000,row*1000000+17)))
                for i,buffer in enumerate(p.boundary_buffers):self.assertEqual(buffer.values[row][0],row*1000000+17+i*20480);self.assertEqual(len(buffer.values[row]),20480)
            self.assertEqual(p.boundary_runs,1);p.close();self.assertIsNone(p.primary_output);self.assertEqual(p.boundary_buffers,[])
            with self.assertRaises(ValueError):p.run(inputs)
    def test_private_binding_restored_and_prefill32_unchanged(self):
        study=isolated();parent=types.ModuleType('xModel.qwen35.vl_0_8b');module=types.ModuleType(parent.__name__+'.cuda_packed_decoder');module.CUDAPackedDecoderProgram=Base;parent.cuda_packed_decoder=module
        def factory():
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,4),study.DecodeBoundaryProgram);self.assertNotIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,32),study.DecodeBoundaryProgram)
            with self.assertRaises(ValueError):module.CUDAPackedDecoderProgram('checkpoint',{},IO(),8,4096,4)
            raise RuntimeError('intentional private-constructor failure')
        with patch.dict(sys.modules,{parent.__name__:parent,module.__name__:module}):
            with self.assertRaises(RuntimeError):study.construct_with_decode_boundary_program(factory)
        self.assertIs(module.CUDAPackedDecoderProgram,Base)
if __name__=='__main__':unittest.main()
