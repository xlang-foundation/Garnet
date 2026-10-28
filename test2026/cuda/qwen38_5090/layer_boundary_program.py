"""Private read-only full-program layer-boundary outputs; original commit prefix retained.

No native diagnostic guard, kernel, original math or precision changes. This study
must prove primary same-profile parity before interpreting appended observations.
"""
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram,T,Qwen35CUDAChunk,flat_concat,concat,packed_view

class LayerBoundaryProgram(CUDAPackedDecoderProgram):
    boundary_layers=(0,1,2,3,4)
    def __init__(self,*args,**kwargs):
        self.primary_output=None;self.boundary_buffers=[];self.boundary_runs=0
        if len(args)<6 or args[3]!=1 or args[4]!=4096 or args[5] not in (32,128):raise ValueError('Boundary study requires explicit B1/cap4096/T32 or T128')
        super().__init__(*args,**kwargs)
        try:
            self.ordinary_output_width=self.output_width
            self.primary_output=self.io.zeros([1,self.ordinary_output_width])
            self.boundary_buffers=[self.io.zeros([1,self.hidden_width]) for _ in self.boundary_layers]
            # Decoder shared output storage must cover the actual wider graph.
            # The native commit layouts and returned primary tensor stay original.
            self.output_width=self.ordinary_output_width+len(self.boundary_layers)*self.hidden_width
        except:self.close();raise

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
            return join([flat]+state_outputs+payload_outputs+boundary_outputs)
        return forward

    def run(self,inputs):
        wide=super().run(inputs)
        self.io.slice_into(wide,0,self.primary_output)
        for i,value in enumerate(self.boundary_buffers):self.io.slice_into(wide,self.ordinary_output_width+i*self.hidden_width,value)
        self.boundary_runs+=1
        return self.primary_output

    def close(self):
        self.primary_output=None;self.boundary_buffers=[]
        super().close()

def construct_with_boundary_program(factory,*args,**kwargs):
    # Private single-process study binding only; constructor restored even on error.
    from xModel.qwen35.vl_0_8b import cuda_packed_decoder as module
    original=module.CUDAPackedDecoderProgram
    assert original is CUDAPackedDecoderProgram
    def choose(*values,**options):
        return (LayerBoundaryProgram if len(values)>=6 and values[5] in (32,128) else original)(*values,**options)
    module.CUDAPackedDecoderProgram=choose
    try:return factory(*args,**kwargs)
    finally:module.CUDAPackedDecoderProgram=original
