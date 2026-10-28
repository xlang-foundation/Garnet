"""Private B4T4 rowwise K normalization, with ordinary and boundary-control variants.

Uses existing primary outputs and arithmetic. Requires the exact source87 DLL,
explicit native opt-in, actual 16-layer trace and numerical qualification.
"""
from .bounded_step45_boundary_program import BoundedStep45BoundaryProgram,CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.cuda_packed_program import T,Qwen35CUDAChunk,flat_concat,concat,packed_view

@T.fusion(name='qwen35_rowwise_k_decode_norm_chunk',role='decoder_layer',boundary='required')
def Qwen35RowwiseKDecodeNormCUDAChunk(hidden,state1,state2,active,positions,slots,config,layer_idx,kind,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False,external_weights=None,register_delta_128=False,join_input_projections=False,external_compressed_weights=None,joined_mlp_projections=False,canonical_kv_state=None,canonical_kv_layout=None,kv_overlay_strategy=1,bf16_canonical_kv=False,bf16_kv_float_reference=False,bf16_low_batch_canonical_kv=False,bf16_prefill128_canonical_kv=False):
    if type(layer_idx) is not int or layer_idx not in range(3,64,4) or kind!='full_attention':raise ValueError('Rowwise K norm requires one of the sixteen full-attention layers')
    if not bf16_canonical_kv or bf16_low_batch_canonical_kv or bf16_prefill128_canonical_kv or bf16_kv_float_reference or canonical_kv_state is None or not external_compressed_weights or external_weights or not fused_attention or not joined_mlp_projections:raise ValueError('Rowwise K norm requires the exclusive shared mixed canonical B4 decode profile')
    p='model.language_model.layers.'+str(layer_idx)
    extra={'external_weights':external_weights} if external_weights else {}
    if external_compressed_weights:extra['external_compressed_weights']=external_compressed_weights
    extra['rowwise_full_decode_k_norm']=1
    if type(bf16_prefill128_canonical_kv) is not bool or (bf16_prefill128_canonical_kv and (bf16_low_batch_canonical_kv or bf16_canonical_kv or bf16_kv_float_reference or canonical_kv_state is None or kind!="full_attention")):
        raise ValueError("T128 BF16 canonical KV requires its separate exclusive full-attention binding")
    if type(bf16_low_batch_canonical_kv) is not bool or (bf16_low_batch_canonical_kv and (bf16_canonical_kv or bf16_kv_float_reference or canonical_kv_state is None)):
        raise ValueError("B1 BF16 canonical KV requires its own exclusive explicit binding")
    if type(bf16_kv_float_reference) is not bool or (bf16_kv_float_reference and (kind!='full_attention' or canonical_kv_state is not None or bf16_canonical_kv or state1 is None or state2 is None)):
        raise ValueError('BF16 float reference requires ordinary full-attention cache bindings')
    if bf16_kv_float_reference:extra['bf16_kv_float_reference']=1
    if type(bf16_canonical_kv) is not bool or (bf16_canonical_kv and canonical_kv_state is None):raise ValueError('BF16 canonical KV requires an explicit full-attention binding')
    if canonical_kv_state is not None and (kind!='full_attention' or canonical_kv_layout is None or state1 is not None or state2 is not None):
        raise ValueError('Canonical KV requires only the full-attention packed-state binding')
    if kind=='linear_attention':
        bound=hidden*T.binary_op('qwen35_conv_bind_state')*state1
        bound=bound*T.binary_op('qwen35_delta_bind_state')*state2
        bound=bound*T.binary_op('qwen35_conv_bind_active')*active
        return bound*T.unary_op('qwen35_linear_decoder_chunk',layer_prefix=p,
            key_heads=config['linear_num_key_heads'],value_heads=config['linear_num_value_heads'],
            key_dim=config['linear_key_head_dim'],value_dim=config['linear_value_head_dim'],eps=config['rms_norm_eps'],fused_delta=int(fused_delta),projection_workspace_bytes=projection_workspace_bytes,retain_linear_inputs=int(retain_linear_inputs),register_delta_128=int(register_delta_128),joined_input_projections=int(join_input_projections),joined_mlp_projections=int(joined_mlp_projections),**extra)
    if canonical_kv_state is not None:
        bound=hidden*T.binary_op('qwen35_attention_bind_canonical')*canonical_kv_state
        extra.update({('canonical_bf16_prefill128_kv_overlay' if bf16_prefill128_canonical_kv else ('canonical_bf16_low_batch_kv_overlay' if bf16_low_batch_canonical_kv else ('canonical_bf16_kv_overlay' if bf16_canonical_kv else 'canonical_kv_overlay'))):1})
        extra.update(kv_overlay_strategy=kv_overlay_strategy,
            canonical_state_width=canonical_kv_layout['cache_batch_stride'],canonical_key_offset=canonical_kv_layout['key_offset'],canonical_value_offset=canonical_kv_layout['value_offset'])
    else:
        bound=hidden*T.binary_op('qwen35_attention_bind_key')*state1
        bound=bound*T.binary_op('qwen35_attention_bind_value')*state2
    bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
    bound=bound*T.binary_op('qwen35_attention_bind_slots')*slots
    bound=bound*T.binary_op('qwen35_attention_bind_active')*active
    rope=config['rope_parameters']
    return bound*T.unary_op('qwen35_full_decoder_chunk',layer_prefix=p,
        num_heads=config['num_attention_heads'],num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
        rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
        mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],eps=config['rms_norm_eps'],fused_attention=int(fused_attention),projection_workspace_bytes=projection_workspace_bytes,joined_mlp_projections=int(joined_mlp_projections),**extra)

class _RowwiseKShape:
    def __init__(self,*args,**kwargs):
        shape=tuple(args[i] if len(args)>i else kwargs.get(name) for i,name in ((3,'batch'),(4,'capacity'),(5,'chunk_size')))
        if any(type(value) is not int for value in shape) or shape!=(4,4096,4):raise ValueError('Rowwise K norm supports only B4 cap4096 T4')
        super().__init__(*args,**kwargs)

class RowwiseKDecodeNormProgram(_RowwiseKShape,CUDAPackedDecoderProgram):
    def _forward(self):
        config=self.config;b=self.batch;t=self.size;h=config['hidden_size'];layout=self.layout;join=flat_concat if self.flat_concat else concat
        @T.fusion(name='qwen35_packed_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,slots,*weights):
            state_outputs=[];payload_outputs=[]
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
                packed=(Qwen35RowwiseKDecodeNormCUDAChunk if layer['kind']=='full_attention' else Qwen35CUDAChunk)(hidden,first,second,active,positions,slots,config,layer['index'],layer['kind'],True,True,
                    self.workspace,self.retain,external,self.register_delta_128,self.join_input_projections and layer['kind']=='linear_attention',external_compressed_weights=compressed,joined_mlp_projections=self.joined_mlp_projections,
                    canonical_kv_state=state if canonical is not None else None,canonical_kv_layout=canonical,kv_overlay_strategy=self.kv_overlay_strategy,
                    bf16_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_state',False) and not getattr(self,'bf16_kv_low_batch_state',False) and not getattr(self,'bf16_kv_prefill128_state',False)),
                    bf16_low_batch_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_low_batch_state',False)),
                    bf16_prefill128_canonical_kv=bool(canonical is not None and getattr(self,'bf16_kv_prefill128_state',False)),
                    bf16_kv_float_reference=bool(layer['kind']=='full_attention' and getattr(self,'bf16_kv_float_reference',False)))
                hidden=packed_view(packed,0,t*h,[b,t,h])
                if compact is not None:w1=compact['output_state1_width'];w2=compact['output_state2_width']
                state_outputs.append(packed_view(packed,t*h,w1+w2,[b,w1+w2]))
                if layer['payload_width']:
                    payload_outputs.append(packed_view(packed,t*h+w1+w2,layer['payload_width'],[b,layer['payload_width']]))
                if layer['index']+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            flat=hidden*T.unary_op('reshape',shape=[b,t*h])
            return join([flat]+state_outputs+payload_outputs)
        return forward

class RowwiseKDecodeNormBoundaryProgram(_RowwiseKShape,BoundedStep45BoundaryProgram):
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
                packed=(Qwen35RowwiseKDecodeNormCUDAChunk if layer['kind']=='full_attention' else Qwen35CUDAChunk)(hidden,first,second,active,positions,slots,config,layer['index'],layer['kind'],True,True,
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

def construct_with_rowwise_k_decode_norm_program(factory,*args,observe_boundaries=False,**kwargs):
    if type(observe_boundaries) is not bool:raise TypeError('observe_boundaries must be a boolean')
    from xModel.qwen35.vl_0_8b import cuda_packed_decoder as module
    original=module.CUDAPackedDecoderProgram
    assert original is CUDAPackedDecoderProgram
    selected=RowwiseKDecodeNormBoundaryProgram if observe_boundaries else RowwiseKDecodeNormProgram
    def choose(*values,**options):
        return (selected if len(values)>=6 and values[3]==4 and values[5]==4 else original)(*values,**options)
    module.CUDAPackedDecoderProgram=choose
    try:return factory(*args,**kwargs)
    finally:module.CUDAPackedDecoderProgram=original
