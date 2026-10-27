from pathlib import Path
import importlib.util,sys,types,unittest
from unittest.mock import patch
from . import test_decode_boundary_program as existing
M=Path(__file__).resolve().parent

def isolated():
    base=existing.isolated()
    class LazyBase(existing.Base):
        def run(self,inputs):
            if self.graph is None:raise ValueError('closed')
            return existing.Buffer([self.batch,self.output_width],[range(r*10000000,r*10000000+self.output_width) for r in range(self.batch)])
    # Rebind the existing observer against the lazy CPU stub, never import native code.
    program=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_packed_program');program.CUDAPackedDecoderProgram=LazyBase
    for n in ('T','Qwen35CUDAChunk','flat_concat','concat','packed_view'):setattr(program,n,None)
    name='test2026.cuda.qwen38_5090.decode_boundary_program'
    spec=importlib.util.spec_from_file_location(name,M/'decode_boundary_program.py');base=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{program.__name__:program}):spec.loader.exec_module(base)
    spec=importlib.util.spec_from_file_location('test2026.cuda.qwen38_5090.isolated_all_layer_decode_boundary_program',M/'all_layer_decode_boundary_program.py');result=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{name:base}):spec.loader.exec_module(result)
    return result,base,LazyBase

class Contract(unittest.TestCase):
    def test_same_original_forward_and_run_no_kernel_or_math_override(self):
        study,base,_=isolated();cls=study.AllLayerDecodeBoundaryProgram
        self.assertIs(cls._forward,base.DecodeBoundaryProgram._forward);self.assertIs(cls.run,base.DecodeBoundaryProgram.run)
        self.assertEqual(cls.boundary_layers,tuple(range(64)))
    def test_all_64_suffixes_and_primary_rows_stay_separate(self):
        study,_,_=isolated()
        for batch in (1,4):
            p=study.AllLayerDecodeBoundaryProgram('checkpoint',{},existing.IO(),batch,4096,4)
            self.assertEqual(len(p.boundary_buffers),64);self.assertEqual(p.output_width,17+64*20480)
            self.assertEqual(p.boundary_offsets[-1],(63,17+63*20480,20480));primary=p.run([object()]*5)
            for row in range(batch):
                self.assertEqual(list(primary.values[row]),list(range(row*10000000,row*10000000+17)))
                for layer,buffer in enumerate(p.boundary_buffers):
                    self.assertEqual(buffer.values[row][0],row*10000000+17+layer*20480);self.assertEqual(len(buffer.values[row]),20480)
            p.close();self.assertEqual(p.boundary_buffers,[])
    def test_unsupported_shapes_rejected_by_original_guard(self):
        study,_,_=isolated()
        for b,c,t in ((2,4096,4),(8,4096,4),(4,512,4),(4,4096,32),(True,4096,4)):
            with self.assertRaises(ValueError):study.AllLayerDecodeBoundaryProgram('checkpoint',{},existing.IO(),b,c,t)
    def test_constructor_binding_restored_after_failure_prefill_unchanged(self):
        study,_,base=isolated();parent=types.ModuleType('xModel.qwen35.vl_0_8b');module=types.ModuleType(parent.__name__+'.cuda_packed_decoder');parent.cuda_packed_decoder=module;module.CUDAPackedDecoderProgram=base
        def factory():
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},existing.IO(),4,4096,4),study.AllLayerDecodeBoundaryProgram)
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},existing.IO(),4,4096,32),base)
            raise RuntimeError('intentional')
        with patch.dict(sys.modules,{parent.__name__:parent,module.__name__:module}):
            with self.assertRaises(RuntimeError):study.construct_with_all_layer_decode_boundary_program(factory)
        self.assertIs(module.CUDAPackedDecoderProgram,base)
if __name__=='__main__':unittest.main()
