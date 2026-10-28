# SPDX-License-Identifier: Apache-2.0
"""Unwired whole-decoder program with one canonical packed state input.

By default large matrices are readonly inputs from the caller's BF16 pool.
The opt-in embedded mode uses checkpoint constants and native mixed lowering.
An explicit shared compressed mode borrows one mixed NVFP4/FP8 pool.
Layer boundaries explicitly materialize FP32 hidden values. This component
does not yet replace the serving decoder or its prefix-commit helper.
"""
import json
import garnet
import tensor
from .decoder_layer import T
from .cuda_chunk import Qwen35CUDAChunk
from .joined_mlp_profile import validate_joined_mlp_profile
from .kv_delta_layout import build_kv_delta_layout
from .kv_bf16_study_layout import build_bf16_kv_study_layout
from .bf16_kv_profile import validate_bf16_kv_program
from .bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_program
from .bf16_prefill128_kv_profile import validate_bf16_prefill128_kv_program
from .kv_bf16_prefill128_study_layout import build_bf16_prefill128_layout
from .kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
from .bf16_float_reference_profile import validate_bf16_float_reference_program

def packed_view(value,start,width,shape):
    return value*T.unary_op('qwen35_packed_view',start=start,width=width,shape=shape)
def concat(values):
    result=values[0]
    for value in values[1:]:result=result*T.binary_op('qwen35_concat_packed')*value
    return result

def flat_concat(values):
    return values[0]*T.unary_op('qwen35_concat_parts',parts=values[1:]) if len(values)>1 else values[0]

def decoder_state_layout(config,batch,capacity):
    result=[];offset=0
    for index,kind in enumerate(config['layer_types']):
        if kind=='linear_attention':
            channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
            first=[batch,channels,config['linear_conv_kernel_dim']]
            second=[batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']]
        elif kind=='full_attention':first=second=[batch,config['num_key_value_heads'],capacity,config['head_dim']]
        else:raise ValueError('Unsupported packed decoder layer kind')
        widths=[]
        for shape in (first,second):
            width=1
            for d in shape[1:]:width*=d
            widths.append(width)
        result.append({'index':index,'kind':kind,'offset':offset,'state1_shape':first,'state2_shape':second,
            'state1_width':widths[0],'state2_width':widths[1]})
        offset+=sum(widths)
    return result,offset

class CUDAPackedDecoderProgram:
    def __init__(self,checkpoint,config,io,batch=1,capacity=512,chunk_size=4,retain_linear_inputs=False,projection_workspace_bytes=0,register_delta_128=False,flat_concat=False,join_input_projections=False,shared_activation_storage=False,internal_packed_stores=False,compact_kv_outputs=False,phase_specification=None,embedded_projection_weights=False,external_compressed_weights=False,joined_mlp_projections=False,kv_delta_state=False,kv_overlay_strategy=1,bf16_kv_state=False,bf16_kv_float_reference=False,bf16_kv_low_batch_state=False,bf16_kv_prefill128_state=False):
        if batch not in (1,2,4,8) or not 32<=capacity<=8192 or chunk_size not in (4,8,16,32,64,128):raise ValueError('Unsupported packed decoder profile')
        if projection_workspace_bytes not in (0,16384):raise ValueError('Invalid projection workspace')
        if type(bf16_kv_low_batch_state) is not bool:raise ValueError('B1 BF16 KV opt-in must be boolean')
        if type(bf16_kv_prefill128_state) is not bool:raise ValueError("T128 BF16 program opt-in must be boolean")
        if bf16_kv_prefill128_state:
            if bf16_kv_low_batch_state is not False:raise ValueError("T128 and T4/T32 BF16 program dialects are exclusive")
            validate_bf16_prefill128_kv_program(True,batch,capacity,chunk_size,retain_linear_inputs,config,
                external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
                embedded_projection_weights=embedded_projection_weights,join_input_projections=join_input_projections,
                kv_delta_state=kv_delta_state,kv_overlay_strategy=kv_overlay_strategy,legacy_bf16_kv_state=bf16_kv_state,bf16_kv_float_reference=bf16_kv_float_reference)
        if bf16_kv_low_batch_state:
            validate_bf16_low_batch_kv_program(True,batch,capacity,chunk_size,retain_linear_inputs,config,
                external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
                embedded_projection_weights=embedded_projection_weights,join_input_projections=join_input_projections,
                kv_delta_state=kv_delta_state,kv_overlay_strategy=kv_overlay_strategy,legacy_bf16_kv_state=bf16_kv_state,bf16_kv_float_reference=bf16_kv_float_reference)

        validate_bf16_float_reference_program(bf16_kv_float_reference,batch,capacity,chunk_size,retain_linear_inputs,config,
            bf16_kv_state=bf16_kv_state,external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
            embedded_projection_weights=embedded_projection_weights,join_input_projections=join_input_projections,kv_delta_state=kv_delta_state,kv_overlay_strategy=kv_overlay_strategy)
        self.bf16_kv_float_reference=bf16_kv_float_reference
        if type(bf16_kv_state) is not bool:raise ValueError('BF16 KV program opt-in must be boolean')
        if bf16_kv_state:
            validate_bf16_kv_program(True,batch,capacity,chunk_size,retain_linear_inputs,config,
                external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
                embedded_projection_weights=embedded_projection_weights,join_input_projections=join_input_projections,
                kv_delta_state=kv_delta_state,kv_overlay_strategy=kv_overlay_strategy)
            validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights,join_input_projections,bf16_cache_profile=True)
        elif bf16_kv_low_batch_state:validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights,join_input_projections,bf16_low_batch_cache_profile=True)
        elif not bf16_kv_prefill128_state:validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights,join_input_projections,bf16_float_reference=bf16_kv_float_reference)
        self.bf16_kv_state=bf16_kv_state or bf16_kv_low_batch_state or bf16_kv_prefill128_state;self.bf16_kv_low_batch_state=bf16_kv_low_batch_state;self.bf16_kv_prefill128_state=bf16_kv_prefill128_state
        if type(kv_delta_state) is not bool or type(kv_overlay_strategy) is not int or kv_overlay_strategy not in (0,1):raise ValueError('Invalid explicit canonical KV profile')
        if kv_delta_state and ((batch,capacity,chunk_size)!=(4,4096,4) or retain_linear_inputs is not True or external_compressed_weights is not True or joined_mlp_projections is not True or embedded_projection_weights or join_input_projections):
            raise ValueError('Canonical KV program requires retained shared mixed joined B4/T4/cap4096')
        self.kv_delta_state=kv_delta_state;self.kv_overlay_strategy=kv_overlay_strategy;self.delta_layout=None
        self.joined_mlp_projections=joined_mlp_projections
        self.config=config;self.io=io;self.batch=batch;self.capacity=capacity;self.size=chunk_size;self.retain=retain_linear_inputs
        self.register_delta_128=register_delta_128;self.flat_concat=flat_concat
        self.embedded_projection_weights=bool(embedded_projection_weights)
        self.external_compressed_weights=bool(external_compressed_weights)
        if self.external_compressed_weights and (self.embedded_projection_weights or join_input_projections):raise ValueError("Shared compressed weights exclude embedded/joined BF16 modes")
        self.join_input_projections=bool(join_input_projections)
        self.internal_packed_stores=bool(internal_packed_stores)
        self.compact_kv_outputs=bool(compact_kv_outputs);self.state_commit=None
        self.phase_specification=None;self.replay_only_linear_state=False
        if phase_specification is not None:
            from .cuda_phase_output_plan import commit_spec
            requested=phase_specification
            replay=requested.get('replay_only_linear_state')
            start,count=(0,len(config['layer_types'])) if replay is True else requested.get('copy_layer_range',[-1,-1])
            expected=commit_spec(config,batch,capacity,chunk_size,start,count,replay)
            if requested!=expected:raise ValueError('Phase graph specification differs from validated layout')
            if not self.compact_kv_outputs or retain_linear_inputs!=replay or replay and not register_delta_128:
                raise ValueError('Phase graph requires compact KV and matching retained native128 replay')
            self.phase_specification=expected;self.replay_only_linear_state=replay
        if self.join_input_projections and chunk_size!=4:raise ValueError('Joined input projections require decode width4')
        self.workspace=projection_workspace_bytes
        if bf16_kv_state or bf16_kv_low_batch_state or bf16_kv_prefill128_state:
            builder=build_bf16_prefill128_layout if bf16_kv_prefill128_state else (build_bf16_low_batch_layout if bf16_kv_low_batch_state else build_bf16_kv_study_layout)
            self.delta_layout=builder(batch,capacity,chunk_size,retain_linear_inputs=retain_linear_inputs)
            self.layout=[dict(l) for l in self.delta_layout['layers']];self.state_width=self.delta_layout['canonical_state_width']
        else:self.layout,self.state_width=decoder_state_layout(config,batch,capacity)
        if kv_delta_state:self.delta_layout=build_kv_delta_layout(self.layout,batch,capacity,chunk_size,config['hidden_size'])
        if self.phase_specification is not None:self.layout=[dict(layer) for layer in self.phase_specification['layout']]
        self.output_state_width=0;full_layers=0
        for layer in self.layout:
            if self.compact_kv_outputs:
                width=chunk_size*config['num_key_value_heads']*config['head_dim'] if layer['kind']=='full_attention' else (0 if self.replay_only_linear_state else None)
                layer['output_offset']=self.output_state_width
                layer['output_state1_width']=width if width is not None else layer['state1_width']
                layer['output_state2_width']=width if width is not None else layer['state2_width']
                if layer['kind']=='full_attention':layer['control_index']=full_layers;full_layers+=1
                self.output_state_width+=layer['output_state1_width']+layer['output_state2_width']
            else:self.output_state_width+=layer['state1_width']+layer['state2_width']
        self.hidden_width=chunk_size*config['hidden_size'];self.payload_width=0;self.weight_names=[];self.weight_layers=[];self.weight_slots=[];self.weights=[];self.graph=None
        symbols=[tensor.input('packed_hidden',shape=[batch,chunk_size,config['hidden_size']],dtype=tensor.float32),
            tensor.input('packed_state',shape=[batch,self.state_width],dtype=tensor.float32),
            tensor.input('packed_active',shape=[batch,chunk_size],dtype=tensor.float32),
            tensor.input('packed_positions',shape=[batch,chunk_size,3],dtype=tensor.float32),
            tensor.input('packed_slots',shape=[batch,chunk_size],dtype=tensor.float32)]
        if self.joined_mlp_projections:
            for layer in self.layout:
                prefix='model.language_model.layers.'+str(layer['index'])+'.mlp.'
                info=json.loads(io.compressed_weight_info(prefix+'gate_up_proj.weight'))
                if not info['joined_mlp'] or info['source_weights']!=[prefix+'gate_proj.weight',prefix+'up_proj.weight']:
                    raise ValueError('Joined MLP provenance mismatch')
        for layer in self.layout:
            index=layer['index'];kind=layer['kind'];p='model.language_model.layers.'+str(index)+'.'
            suffixes=['mlp.gate_up_proj.weight','mlp.down_proj.weight'] if self.joined_mlp_projections else ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
            joined=None;joined_name=p+'linear_attn.in_proj_joined.weight'
            if kind=='linear_attention' and self.join_input_projections:
                joined=io.group_weights([p+'linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')])
                suffixes+=['linear_attn.in_proj_joined.weight','linear_attn.out_proj.weight']
            else:suffixes+=['linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if kind=='linear_attention' else ['self_attn.'+name+'.weight' for name in ('q_proj','k_proj','v_proj','o_proj')]
            indices=[]
            # Empty external bindings deliberately select the existing native
            # checkpoint lookup/lowering, including mixed compressed constants.
            for suffix in ([] if self.embedded_projection_weights else suffixes):
                name=p+suffix
                if self.external_compressed_weights:
                    original={'dtype':'U8'} if self.joined_mlp_projections and suffix=='mlp.gate_up_proj.weight' else json.loads(io.weight_info(name))
                    if original['dtype']=='BF16':
                        if kind!='linear_attention' or suffix not in ('linear_attn.in_proj_a.weight','linear_attn.in_proj_b.weight'):
                            raise ValueError('Unexpected BF16 fallback in shared compressed projection: '+name)
                        continue
                    info=json.loads(io.compressed_weight_info(name));slots=[]
                    buffers=[('packed',info['packed_shape'],io.compressed_weight(name))]
                    if info['scale_bytes']:buffers.append(('scales',info['scale_shape'],io.compressed_scales(name)))
                    for role,shape,value in buffers:
                        slots.append(len(self.weights));self.weights.append(value)
                        symbols.append(tensor.input('compressed_'+role+'_'+str(len(self.weights)),shape=shape,dtype=tensor.int32))
                    indices.append(len(self.weight_names));self.weight_names.append(name);self.weight_slots.append(slots)
                    continue
                if joined is not None and name==joined_name:
                    rows=2*config['linear_num_key_heads']*config['linear_key_head_dim']+2*config['linear_num_value_heads']*config['linear_value_head_dim']+2*config['linear_num_value_heads']
                    info={'dtype':'BF16','shape':[rows,config['hidden_size']]};value=joined
                else:info=json.loads(io.weight_info(name));value=io.weight(name)
                if info['dtype']!='BF16' or len(info['shape'])!=2:raise ValueError('Packed decoder requires BF16 matrix weights')
                indices.append(len(self.weight_names));self.weight_names.append(name);self.weight_slots.append([len(self.weights)]);self.weights.append(value)
                symbols.append(tensor.input('packed_weight_'+str(len(self.weight_names)),shape=info['shape'],dtype=tensor.bfloat16))
            self.weight_layers.append(indices)
            layer['payload_width']=0
            if retain_linear_inputs and kind=='linear_attention':
                channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                layer['payload_width']=chunk_size*(channels+config['linear_num_value_heads']*config['linear_value_head_dim']+2*config['linear_num_value_heads'])
                layer['payload_offset']=self.hidden_width+self.output_state_width+self.payload_width;self.payload_width+=layer['payload_width']
        self.status_offset=self.hidden_width+self.output_state_width+self.payload_width
        self.output_width=self.status_offset+full_layers
        if self.phase_specification is not None:
            spec=self.phase_specification
            if self.layout!=spec['layout'] or any(actual!=spec[key] for actual,key in ((self.state_width,'state_width'),(self.hidden_width,'hidden_width'),(self.output_state_width,'output_state_width'),(self.status_offset,'status_offset'),(self.output_width,'packed_width'))):
                raise ValueError('Compiled phase metadata differs from validated helper contract')
        if self.delta_layout is not None:
            self.output_width=self.delta_layout['output_width']
            for layer,compact in zip(self.layout,self.delta_layout['layers']):
                if layer['payload_width']:layer['payload_offset']=compact['payload_offset']
        self.graph=garnet.compiled_graph();self.graph.set_builder_workspace_bytes(256<<20)
        if shared_activation_storage:self.graph.enable_user_managed_workspace(True)
        try:
            self.graph.build(self._forward()(*symbols),symbols,checkpoint);self.graph.enable_cuda_graph(True)
            if self.compact_kv_outputs:
                self.state_commit=garnet.cuda_packed_prefix()
                self.state_commit.build(checkpoint,json.dumps(self.commit_spec(not self.replay_only_linear_state)))
        except:self.close();raise

    def _forward(self):
        config=self.config;b=self.batch;t=self.size;h=config['hidden_size'];layout=self.layout;join=flat_concat if self.flat_concat else concat
        @T.fusion(name='qwen35_packed_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,slots,*weights):
            state_outputs=[];payload_outputs=[];control_outputs=[]
            root=hidden*T.unary_op('qwen35_internal_root',width=self.output_width) if self.internal_packed_stores else None
            for ordinal,(layer,indices) in enumerate(zip(layout,self.weight_layers)):
                w1=layer['state1_width'];w2=layer['state2_width'];offset=layer['offset']
                if self.delta_layout is not None:
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
                    if compact is not None:w1=compact['output_state1_width'];w2=compact['output_state2_width']
                    state_outputs.append(packed_view(packed,t*h,w1+w2,[b,w1+w2]))
                    if layer['payload_width']:
                        payload_outputs.append(packed_view(packed,t*h+w1+w2,layer['payload_width'],[b,layer['payload_width']]))
                    if layer['index']+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
                else:
                    first=packed_view(state,offset,w1,layer['state1_shape']);second=packed_view(state,offset+w1,w2,layer['state2_shape'])
                    external={self.weight_names[i]:weights[i] for i in indices}
                    out1=layer.get('output_state1_width',w1);out2=layer.get('output_state2_width',w2);out_offset=layer.get('output_offset',offset)
                    offsets=[self.hidden_width+out_offset,self.hidden_width+out_offset+out1]
                    omit=self.replay_only_linear_state and layer['kind']=='linear_attention'
                    if omit:offsets=[]
                    if self.compact_kv_outputs and layer['kind']=='full_attention':offsets.append(self.status_offset+layer['control_index'])
                    if self.internal_packed_stores and layer['payload_width']:
                        channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                        widths=[t*channels,t*config['linear_num_value_heads']*config['linear_value_head_dim'],t*config['linear_num_value_heads'],t*config['linear_num_value_heads']]
                        cursor=layer['payload_offset']
                        for width in widths:offsets.append(cursor);cursor+=width
                    packed=Qwen35CUDAChunk(hidden,first,second,active,positions,slots,config,layer['index'],layer['kind'],True,True,
                        self.workspace,self.retain,external,self.register_delta_128,self.join_input_projections and layer['kind']=='linear_attention',root,offsets,self.internal_packed_stores and ordinal+1==len(layout),self.compact_kv_outputs,omit)
                    if self.internal_packed_stores:
                        root=packed;hidden=root*T.unary_op('qwen35_stored_hidden',shape=[b,t,h])
                    else:
                        hidden=packed_view(packed,0,t*h,[b,t,h])
                        if out1+out2:state_outputs.append(packed_view(packed,t*h,out1+out2,[b,out1+out2]))
                        if self.compact_kv_outputs and layer['kind']=='full_attention':control_outputs.append(packed_view(packed,t*h+out1+out2,1,[b,1]))
                        if layer['payload_width']:
                            payload_outputs.append(packed_view(packed,t*h+out1+out2,layer['payload_width'],[b,layer['payload_width']]))
                    if ordinal+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            if self.internal_packed_stores:return root
            flat=hidden*T.unary_op('reshape',shape=[b,t*h])
            return join([flat]+state_outputs+payload_outputs+control_outputs)
        return forward

    def commit_spec(self,copy_only=False):
        if self.phase_specification is not None:
            if copy_only and self.replay_only_linear_state:raise ValueError('Replay-only graph has no full linear state to copy')
            return dict(self.phase_specification)
        result={'config':self.config,'layout':self.layout,'batch':self.batch,'tokens':self.size,'capacity':self.capacity,
                'state_width':self.state_width,'packed_width':self.output_width,'hidden_width':self.hidden_width}
        if self.compact_kv_outputs:result.update({'compact_kv_outputs':True,'copy_only':copy_only,'output_state_width':self.output_state_width,'status_offset':self.status_offset})
        return result

    def commit_all(self,packed,state,slots,active):
        if self.state_commit is None:raise ValueError('Compact state commit is unavailable')
        self.state_commit.commit_all(packed,state,slots,active)

    def commit_prefix(self,packed,state,spans):
        if not self.replay_only_linear_state or self.state_commit is None:raise ValueError('Replay-only phase commit is unavailable')
        self.state_commit.commit(packed,state,spans)

    def run(self,inputs):
        if self.graph is None:raise ValueError('Packed decoder program is closed')
        if len(inputs)!=5:raise ValueError('Packed decoder requires hidden/state/active/positions/slots')
        return self.graph.run_device(list(inputs)+self.weights)

    def memory_info(self):
        if self.graph is None:raise ValueError('Packed decoder program is closed')
        return {'graph':json.loads(self.graph.stats_json()),'embedded_projection_weights':self.embedded_projection_weights,'external_compressed_weights':self.external_compressed_weights,'joined_mlp_projections':self.joined_mlp_projections,'kv_delta_state':self.kv_delta_state,'kv_overlay_strategy':self.kv_overlay_strategy,'output_width':self.output_width,'external_projection_matrix_count':len(self.weight_names),'register_delta_128':self.register_delta_128,'flat_concat':self.flat_concat,'joined_input_projections':self.join_input_projections,'internal_packed_stores':self.internal_packed_stores,'state_width':self.state_width,'hidden_width':self.hidden_width,
            'payload_width':self.payload_width,'external_weight_count':len(self.weights),'fp32_layer_barriers':max(0,len(self.layout)-1),
            'compact_kv_outputs':self.compact_kv_outputs,'output_state_width':self.output_state_width,'status_offset':self.status_offset,
            'phase_output_layout':self.phase_specification['phase_output_layout'] if self.phase_specification is not None else None,'replay_only_linear_state':self.replay_only_linear_state,'layer_indices':[layer['index'] for layer in self.layout],
            'state_commit':json.loads(self.state_commit.stats_json()) if self.state_commit is not None else None,
            'scope':'Program graph only; canonical state, request buffers and the one projection pool belong to caller. Not process peak.'}

    def close(self):
        if self.graph is not None:self.graph.close()
        if self.state_commit is not None:self.state_commit.close()
        self.state_commit=None
        self.graph=None;self.weights=[];self.io=None
