"""Private original-prefix program for readonly canonical layer3 stage observations."""
import hashlib
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram,T,flat_concat,concat,packed_view
from .full_chunk_stage_chunk import FullChunkStageChunk as Qwen35CUDAChunk
from .full_chunk_stage_layout import STAGES,stage_offsets
from .bf16_model_state_witness import logical_state_witness

class FullChunkStageProgram(CUDAPackedDecoderProgram):
    def __init__(self,*args,**kwargs):
        self.primary_output=None;self.stage_buffers={};self.stage_runs=0;self.readonly_observations=[]
        if len(args)<6 or args[3]!=1 or args[4]!=4096 or args[5] not in (32,128):raise ValueError('Stage program requires explicit B1/cap4096/T32 orT128')
        super().__init__(*args,**kwargs)
        try:
            self.ordinary_output_width=self.output_width;self.primary_output=self.io.zeros([1,self.output_width])
            self.stage_buffers={name:self.io.zeros([1,self.size*width]) for name,width in STAGES}
            self.output_width+=self.size*sum(width for name,width in STAGES)
        except:self.close();raise

    def _forward(self):
        config=self.config;b=self.batch;t=self.size;h=config['hidden_size'];layout=self.layout;join=flat_concat if self.flat_concat else concat
        @T.fusion(name='qwen35_packed_decoder_program',boundary='required')
        def forward(hidden,state,active,positions,slots,*weights):
            state_outputs=[];payload_outputs=[];stage_outputs=[]
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
                if compact is not None:w1=compact['output_state1_width'];w2=compact['output_state2_width']
                state_outputs.append(packed_view(packed,t*h,w1+w2,[b,w1+w2]))
                if layer['index']==3:
                    for name,offset,words in stage_offsets(t,t*h+w1+w2):stage_outputs.append(packed_view(packed,offset,words,[b,words]))
                if layer['payload_width']:
                    payload_outputs.append(packed_view(packed,t*h+w1+w2,layer['payload_width'],[b,layer['payload_width']]))
                if layer['index']+1<len(layout):hidden=hidden*T.unary_op('qwen35_f32_barrier')
            flat=hidden*T.unary_op('reshape',shape=[b,t*h])
            return join([flat]+state_outputs+payload_outputs+stage_outputs)
        return forward

    def input_witness(self,inputs,start):
        state=logical_state_witness(self.io,inputs[1],self.layout,1,self.state_width,[start],packed_bf16=True)
        controls={}
        for i,words in ((0,self.size*5120),(2,self.size),(3,self.size*3),(4,self.size)):
            raw=bytes(self.io.finite_f32_bytes(inputs[i]));assert len(raw)==words*4
            controls[str(i)]={'words':words,'sha256':hashlib.sha256(raw).hexdigest()}
        return {'state':state,'controls':controls}

    def run(self,inputs):
        if self.graph is None:raise ValueError('Stage program is closed')
        if len(inputs)!=5:raise ValueError('Stage program requires all five original inputs')
        start=(self.stage_runs%(3968//self.size))*self.size;capture=start<128
        before=self.input_witness(inputs,start) if capture else None
        wide=super().run(inputs);self.io.slice_into(wide,0,self.primary_output);offset=self.ordinary_output_width
        for name,width in STAGES:self.io.slice_into(wide,offset,self.stage_buffers[name]);offset+=self.size*width
        assert offset==self.output_width
        if capture:
            after=self.input_witness(inputs,start);assert before==after
            self.readonly_observations.append({'start':start,'run':self.stage_runs,'before':before,'after':after})
        self.stage_runs+=1
        return self.primary_output

    def close(self):
        self.primary_output=None;self.stage_buffers={}
        super().close()

def construct_with_full_chunk_stages(factory,*args,**kwargs):
    from xModel.qwen35.vl_0_8b import cuda_packed_decoder as module
    original=module.CUDAPackedDecoderProgram;assert original is CUDAPackedDecoderProgram
    def choose(*values,**options):return (FullChunkStageProgram if len(values)>=6 and values[5] in (32,128) else original)(*values,**options)
    module.CUDAPackedDecoderProgram=choose
    try:return factory(*args,**kwargs)
    finally:module.CUDAPackedDecoderProgram=original
