"""Bounded three-concat observer; existing native 128-extra-parts guard unchanged."""
from .all_layer_decode_boundary_program import AllLayerDecodeBoundaryProgram,CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.cuda_packed_program import T,Qwen35CUDAChunk,flat_concat,concat,packed_view

class BoundedStep45BoundaryProgram(AllLayerDecodeBoundaryProgram):
    def _forward(self):
        config=self.config;b=self.batch;t=self.size;h=config['hidden_size'];layout=self.layout;join=flat_concat if self.flat_concat else concat
        @T.fusion(name='qwen35_packed_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,slots,*weights):
            state_outputs=[];payload_outputs=[];boundary_outputs=[]
            for layer,indices in zip(layout,self.weight_layers):
                w1=layer['state1_width'];w2=layer['state2_width'];offset=layer['offset']
                compact=self.delta_layout['layers'][layer['index']] if self.delta_layout is not None else None
                canonical=compact if compact is not None and layer['kind']=='full_attention' else None
                if canonical is not None and getattr(self,'bf16_kv_state',False):
                    canonical={'cache_batch_stride':self.state_width,'key_offset':compact['offset'],'value_offset':compact['offset']+compact['state1_width']}
                first=second=None
                if canonical is None:
                    first=packed_view(state,offset,w1,layer['state1_shape']);second=packed_view(state,offset+w1,w2,layer['state2_shape'])
                external={} if self.external_compressed_weights else {self.weight_names[i]:weights[self.weight_slots[i][0]] for i in indices}
                compressed={self.weight_names[i]:[weights[j] for j in self.weight_slots[i]] for i in indices} if self.external_compressed_weights else None
                packed=Qwen35CUDAChunk(hidden,first,second,active,positions,slots,config,layer['index'],layer['kind'],True,True,
                    self.workspace,self.retain,external,self.register_delta_128,self.join_input_projections and layer['kind']=='linear_attention',external_compressed_weights=compressed,joined_mlp_projections=self.joined_mlp_projections,
                    canonical_kv_state=state if canonical is not None else None,canonical_kv_layout=canonical,kv_overlay_strategy=self.kv_overlay_strategy,
                    bf16_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_state',False) and not getattr(self,'bf16_kv_low_batch_state',False) and not getattr(self,'bf16_kv_prefill128_state',False)),
                    bf16_low_batch_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_low_batch_state',False)),
                    bf16_prefill128_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_prefill128_state',False)),
                    bf16_kv_float_reference=bool(layer['kind']=='full_attention' and getattr(self,'bf16_kv_float_reference',False)))
                hidden=packed_view(packed,0,t*h,[b,t,h])
                if layer['index'] in self.boundary_layers:boundary_outputs.append(hidden*T.unary_op('reshape',shape=[b,t*h]))
                if compact is not None:w1=compact['output_state1_width'];w2=compact['output_state2_width']
                state_outputs.append(packed_view(packed,t*h,w1+w2,[b,w1+w2]))
                if layer['payload_width']:
                    payload_outputs.append(packed_view(packed,t*h+w1+w2,layer['payload_width'],[b,layer['payload_width']]))
                if layer['index']+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            flat=hidden*T.unary_op('reshape',shape=[b,t*h])
            primary=join([flat]+state_outputs+payload_outputs)
            observations=join(boundary_outputs)
            return join([primary,observations])
        return forward


def construct_with_bounded_step45_boundary_program(factory,*args,**kwargs):
    from xModel.qwen35.vl_0_8b import cuda_packed_decoder as module
    original=module.CUDAPackedDecoderProgram
    assert original is CUDAPackedDecoderProgram
    def choose(*values,**options):
        return (BoundedStep45BoundaryProgram if len(values)>=6 and values[5]==4 else original)(*values,**options)
    module.CUDAPackedDecoderProgram=choose
    try:return factory(*args,**kwargs)
    finally:module.CUDAPackedDecoderProgram=original
