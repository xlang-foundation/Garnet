# SPDX-License-Identifier: Apache-2.0
"""Experimental whole-target tree forward over a borrowed canonical decoder.

Requires the native tree lowering and plugins. This class is not a serving
route or a speculative generator. The caller holds the parent's existing
state transaction through validation and any eventual accepted-path commit.
"""
import json,threading,struct
import garnet
from .cuda_tree_storage import bind_parent_storage
import tensor
from .decoder_layer import T
from .cuda_packed_program import decoder_state_layout,packed_view,flat_concat


def tree_chunk(hidden,state1,state2,active,positions,metadata,config,layer,kind,external,capacity,workspace):
    extra={'external_weights':external,'tree_metadata':metadata,'capacity':capacity,'projection_workspace_bytes':workspace}
    if kind=='linear_attention':
        bound=hidden*T.binary_op('qwen35_conv_bind_state')*state1
        bound=bound*T.binary_op('qwen35_delta_bind_state')*state2
        bound=bound*T.binary_op('qwen35_conv_bind_active')*active
        return bound*T.unary_op('qwen35_linear_tree_chunk',layer_prefix='model.language_model.layers.'+str(layer),
            key_heads=config['linear_num_key_heads'],value_heads=config['linear_num_value_heads'],
            key_dim=config['linear_key_head_dim'],value_dim=config['linear_value_head_dim'],eps=config['rms_norm_eps'],
            fused_delta=1,retain_linear_inputs=1,register_delta_128=1,joined_input_projections=0,**extra)
    bound=hidden*T.binary_op('qwen35_attention_bind_key')*state1
    bound=bound*T.binary_op('qwen35_attention_bind_value')*state2
    bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
    bound=bound*T.binary_op('qwen35_attention_bind_active')*active
    rope=config['rope_parameters']
    return bound*T.unary_op('qwen35_full_tree_chunk',layer_prefix='model.language_model.layers.'+str(layer),
        num_heads=config['num_attention_heads'],num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
        rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
        mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],
        eps=config['rms_norm_eps'],fused_attention=1,compact_kv_outputs=0,**extra)


class CUDATreeDecoderProgram:
    def __init__(self,parent,nodes=32,share_parent_storage=False):
        if nodes not in (8,16,32,64):raise ValueError('Native tree profiles require8/16/32/64 nodes')
        if parent.head is None or parent.poisoned or not parent.deferred_state_commit or not parent.register_delta_128:
            raise ValueError('Native tree forward requires an open deferred/register128 canonical parent')
        self.share_parent_storage=bool(share_parent_storage)
        self.parent=parent;self.io=parent.io;self.config=parent.config;self.batch=parent.batch;self.capacity=parent.capacity;self.nodes=nodes
        self.layout,self.state_width=decoder_state_layout(self.config,self.batch,self.capacity)
        self.hidden_width=nodes*self.config['hidden_size'];self.payload_width=0;self.weight_names=[];self.weight_layers=[];self.weights=[]
        self.graph=None;self.norm_graph=None;self.hidden=None;self.status=None;self.metadata=None;self.head_input=None;self.head_chunk=None
        self.last_output=None;self.last_inputs=None;self.owner=None;self.verified=False;self.verified_epoch=None;self.borrow_registered=False;self.last_status=[]
        b=self.batch;n=nodes;c=self.config
        self.symbol_shapes=[[b,n,c['hidden_size']],[b,self.state_width],[b,n],[b,n,3],[b,n],[b,1]]
        symbols=[tensor.input('tree_input_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(self.symbol_shapes)]
        parent.tree_program_borrows+=1;self.borrow_registered=True
        try:
            for layer in self.layout:
                index=layer['index'];kind=layer['kind'];prefix='model.language_model.layers.'+str(index)+'.'
                suffixes=['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
                suffixes+=['linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if kind=='linear_attention' else ['self_attn.'+name+'.weight' for name in ('q_proj','k_proj','v_proj','o_proj')]
                indices=[]
                for suffix in suffixes:
                    name=prefix+suffix;info=json.loads(self.io.weight_info(name))
                    if info['dtype']!='BF16' or len(info['shape'])!=2:raise ValueError('Native tree matrices require original BF16 storage')
                    indices.append(len(self.weight_names));self.weight_names.append(name);self.weights.append(self.io.weight(name))
                    symbols.append(tensor.input('tree_weight_'+str(len(self.weight_names)),shape=info['shape'],dtype=tensor.bfloat16))
                self.weight_layers.append(indices)
                if kind=='linear_attention':
                    channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
                    width=n*(channels+2*c['linear_num_value_heads'])
                else:width=2*n*c['num_key_value_heads']*c['head_dim']
                layer['tree_payload_offset']=self.hidden_width+self.payload_width;layer['tree_payload_width']=width;self.payload_width+=width
            self.status_offset=self.hidden_width+self.payload_width;self.metadata_offset=self.status_offset+len(self.layout);self.output_width=self.metadata_offset+836
            self.graph=garnet.compiled_graph();self.graph.set_builder_workspace_bytes(256<<20)
            if self.share_parent_storage:self.graph.enable_user_managed_workspace(True)
            self.graph.build(self._forward()(*symbols),symbols,parent.checkpoint);self.graph.enable_cuda_graph(True)
            self.graph.enable_serial_graph_reuse(bool(getattr(parent,'serial_target_graph_reuse',False)))
            if self.share_parent_storage:bind_parent_storage(self.graph,parent,b,self.output_width)
            self.hidden=self.io.zeros([b,n,c['hidden_size']]);self.status=self.io.zeros([b,len(self.layout)]);self.metadata=self.io.zeros([b,836])
            self.head_input=self.io.zeros([b,n,c['hidden_size']])
            if b*n>128:self.head_chunk=self.io.zeros([1,128,c['hidden_size']])
            symbol=tensor.input('tree_final_hidden',shape=[b,n,c['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_tree_final_norm',boundary='required')
            def normalize(value):return value*T.unary_op('qwen35_rms_norm_packed',weight_name='model.language_model.norm.weight',eps=c['rms_norm_eps'])
            self.norm_graph=garnet.compiled_graph();self.norm_graph.build(normalize(symbol),[symbol],parent.checkpoint);self.norm_graph.enable_cuda_graph(True)
        except:self.close();raise

    def _forward(self):
        b=self.batch;n=self.nodes;c=self.config;h=c['hidden_size']
        @T.fusion(name='qwen35_tree_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,parents,prefix,*weights):
            metadata=parents*T.unary_op('qwen35_tree_metadata',active=active,prefix=prefix,capacity=self.capacity)
            payloads=[];statuses=[]
            for layer,indices in zip(self.layout,self.weight_layers):
                start=layer['offset'];w1=layer['state1_width'];w2=layer['state2_width']
                first=packed_view(state,start,w1,layer['state1_shape']);second=packed_view(state,start+w1,w2,layer['state2_shape'])
                external={self.weight_names[i]:weights[i] for i in indices}
                packed=tree_chunk(hidden,first,second,active,positions,metadata,c,layer['index'],layer['kind'],external,self.capacity,self.parent.projection_workspace_bytes)
                hidden=packed_view(packed,0,n*h,[b,n,h]);width=layer['tree_payload_width']
                payloads.append(packed_view(packed,n*h,width,[b,width]));statuses.append(packed_view(packed,n*h+width,1,[b,1]))
                if layer['index']+1<len(self.layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            exported=metadata*T.unary_op('qwen35_tree_metadata_f32',batch=b,nodes=n,capacity=self.capacity)
            return flat_concat([hidden*T.unary_op('reshape',shape=[b,n*h])]+payloads+statuses+[exported])
        return forward

    def _ready(self,owner):
        d=self.parent
        if self.graph is None or d.head is None or d.poisoned:raise ValueError('Tree program or parent is closed/poisoned')
        if d.state_transaction is not owner or d.transaction_thread!=threading.get_ident():raise ValueError('Tree execution requires the current parent state transaction')

    def run(self,hidden,active,positions,parents,prefix,owner):
        self._ready(owner)
        if self.share_parent_storage and owner.phase!='verifying':raise ValueError('Shared target output requires the target verification phase')
        self.owner=owner;self.verified=False;self.verified_epoch=None;self.last_status=[]
        self.parent.verification_chunk_allowed=False
        self.last_inputs=[hidden,self.parent.state,active,positions,parents,prefix]
        self.last_output=self.graph.run_device(self.last_inputs+self.weights)
        self.io.slice_into(self.last_output,0,self.hidden);self.io.slice_into(self.last_output,self.status_offset,self.status)
        self.io.slice_into(self.last_output,self.metadata_offset,self.metadata)
        # This synchronization/status read belongs to the verification boundary;
        # device-resident generation may replace it only after separate gates.
        raw=self.io.finite_f32_bytes(self.status)
        flags=list(struct.unpack('<'+str(self.batch*len(self.layout))+'f',raw));self.last_status=flags
        if any(v!=0 for v in flags):raise ValueError('Native tree metadata/consumer status rejected the verification')
        self.verified=True;self.verified_epoch=self.parent.state_transaction_epoch
        return self.last_output

    def predictions(self,owner):
        self._ready(owner)
        if self.share_parent_storage and owner.phase!='verified':raise ValueError('Shared target predictions require the verified phase')
        if self.last_output is None or not self.verified or self.owner is not owner or self.verified_epoch!=self.parent.state_transaction_epoch:raise ValueError('Predictions require a successful tree under this transaction')
        normalized=self.norm_graph.run_device([self.hidden]);self.io.slice_into(normalized,0,self.head_input)
        # Native packed RMS deliberately returns [B,N*H]; slice_into requires
        # packed rank2 source and preserves the existing [B,N,H] destination.
        rows=self.batch*self.nodes;h=self.config['hidden_size']
        if rows<=128:return list(self.io.predict_tokens(self.head_input))
        flat=self.io.storage_view(self.head_input,[1,rows*h]);result=[]
        for start in range(0,rows,128):
            count=min(128,rows-start)
            chunk=self.io.storage_view(self.head_chunk,[1,count,h])
            self.io.slice_into(flat,start*h,chunk)
            result.extend(list(self.io.predict_tokens(chunk)))
        return result

    def memory_info(self):
        if self.graph is None:raise ValueError('Tree program is closed')
        buffers=sum(self.io.tensor_bytes(v) for v in (self.hidden,self.status,self.metadata,self.head_input))
        if self.head_chunk is not None:buffers+=self.io.tensor_bytes(self.head_chunk)
        return {'graph':json.loads(self.graph.stats_json()),'norm_graph':json.loads(self.norm_graph.stats_json()),'owned_request_observation_bytes':buffers,
            'nodes':self.nodes,'payload_width':self.payload_width,'status_offset':self.status_offset,'metadata_offset':self.metadata_offset,
            'output_width':self.output_width,'borrowed_projection_weight_entries':len(self.weights),'canonical_state_output_bytes':0,
            'scope':'One tree program/norm/buffers only; weights borrowed from parent readonly pool, canonical state remains parent-owned. Not total process memory/peak or model admission.'}

    def close(self):
        if self.owner is not None and self.parent.state_transaction is self.owner:raise ValueError('Finish the parent tree transaction before closing its program')
        if self.norm_graph is not None:self.norm_graph.close()
        if self.graph is not None:self.graph.close()
        if self.borrow_registered:self.parent.tree_program_borrows-=1;self.borrow_registered=False
        self.norm_graph=None;self.graph=None;self.weights=[];self.last_output=None;self.last_inputs=None
        self.hidden=None;self.status=None;self.metadata=None;self.head_input=None;self.head_chunk=None;self.owner=None;self.verified=False;self.verified_epoch=None;self.last_status=[]
