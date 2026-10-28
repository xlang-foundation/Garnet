"""Private all-layer observer; requires same-profile ordinary-state parity on GPU."""
from .decode_boundary_program import DecodeBoundaryProgram,CUDAPackedDecoderProgram

class AllLayerDecodeBoundaryProgram(DecodeBoundaryProgram):
    boundary_layers=tuple(range(64))
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.boundary_offsets=[(layer,self.ordinary_output_width+i*self.hidden_width,self.hidden_width)
                               for i,layer in enumerate(self.boundary_layers)]

def construct_with_all_layer_decode_boundary_program(factory,*args,**kwargs):
    from xModel.qwen35.vl_0_8b import cuda_packed_decoder as module
    original=module.CUDAPackedDecoderProgram
    assert original is CUDAPackedDecoderProgram
    def choose(*values,**options):
        return (AllLayerDecodeBoundaryProgram if len(values)>=6 and values[5]==4 else original)(*values,**options)
    module.CUDAPackedDecoderProgram=choose
    try:return factory(*args,**kwargs)
    finally:module.CUDAPackedDecoderProgram=original
