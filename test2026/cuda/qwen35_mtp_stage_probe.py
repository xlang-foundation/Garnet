"""Bounded diagnostic graph. Extra outputs may change TensorRT fusion."""
import json
import garnet
import tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T

class MTPStageProbe:
    def __init__(self,decoder,tree):
        self.graph=None;self.weights=[];self.tree=bool(tree)
        b=decoder.batch;n=32;c=decoder.config;h=c['hidden_size'];kv=c['num_key_value_heads'];dim=c['head_dim']
        if b!=1:raise ValueError('Stage probe is B1 N32 only')
        cache=[b,kv,decoder.capacity,dim]
        shapes=[[b,n,h],[b,n,h],cache,cache,[b,n,3],[b,n],[b,n]]
        if tree:shapes.append([b,1])
        self.input_count=len(shapes)
        symbols=[tensor.input('mtp_stage_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        names=['mtp.fc.weight']+['mtp.layers.0.'+name+'.weight' for name in (
            'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')]
        for i,name in enumerate(names):
            info=json.loads(decoder.io.weight_info(name));assert info['dtype']=='BF16'
            self.weights.append(decoder.io.weight(name));symbols.append(tensor.input('mtp_stage_weight_'+str(i),shape=info['shape'],dtype=tensor.bfloat16))
        rope=c['rope_parameters']
        @T.fusion(name='qwen35_mtp_stage_probe_'+str(int(tree)),role='decoder_layer',boundary='required')
        def forward(features,embedding,key,value,positions,index,active,*tail):
            weights=tail[1:] if tree else tail
            bound=features*T.binary_op('qwen35_mtp_bind_embedding')*embedding
            bound=bound*T.binary_op('qwen35_attention_bind_key')*key
            bound=bound*T.binary_op('qwen35_attention_bind_value')*value
            bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
            if not tree:bound=bound*T.binary_op('qwen35_attention_bind_slots')*index
            bound=bound*T.binary_op('qwen35_attention_bind_active')*active
            extra={'parents':index,'prefix':tail[0]} if tree else {}
            return bound*T.unary_op('qwen35_mtp_stage_probe',tree=int(tree),
                num_heads=c['num_attention_heads'],num_kv_heads=kv,head_dim=dim,
                rotary_dim=int(dim*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
                mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],
                eps=c['rms_norm_eps'],projection_workspace_bytes=decoder.projection_workspace_bytes,
                external_weights={names[i]:weights[i] for i in range(len(names))},**extra)
        self.layout=[];offset=0
        parts=[('previous',h),('norm_embedding',h),('norm_previous',h),('projected_fc',h),('layer_input_norm',h),
            ('query_gate',2*c['num_attention_heads']*dim),('projected_key',kv*dim),('projected_value',kv*dim),
            ('attention_context',c['num_attention_heads']*dim),('attention_output',h),('residual',h),('raw_hidden',h),
            ('normalized',h),('retained',h),('mean_square',1),('denominator',1)]
        for name,width in parts:
            self.layout.append({'name':name,'offset':offset,'width_per_node':width,'width':n*width});offset+=n*width
        self.layout.append({'name':'norm_scale','offset':offset,'width':h});offset+=h
        self.layout.append({'name':'status','offset':offset,'width':1});offset+=1;self.output_width=offset
        self.graph=garnet.compiled_graph();self.graph.set_builder_workspace_bytes(256<<20)
        try:self.graph.build(forward(*symbols),symbols,decoder.checkpoint);self.graph.enable_cuda_graph(True)
        except:self.close();raise
    def run(self,inputs):
        if self.graph is None or len(inputs)!=self.input_count:raise ValueError('Wrong stage probe input count or closed graph')
        return self.graph.run_device(inputs+self.weights)
    def close(self):
        if self.graph is not None:self.graph.close()
        self.graph=None;self.weights=[]
