"""Mistral7B text adapter over shared dense native Metal operators.

Research bring-up only; generation/coding/performance admission is separate.
"""
# SPDX-License-Identifier: Apache-2.0
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder


class MistralMetalDecoder(DenseMetalDecoder):
    def _profile_batches(self):
        return (1,2,4) if self.dense_prefill_limit else (1,2)

    def reset(self):
        if getattr(self,'_building_native_resident',False) and self.session is None:
            # Native session construction allocates and initializes its private
            # caches from graph metadata. It never consumes these host copies.
            self.poisoned=False;self.slots=[0 for row in range(self.batch)]
            for layer in self.layers:layer['state1']=None;layer['state2']=None
        else:DenseMetalDecoder.reset(self)

    def __init__(self,checkpoint,config,batch=1,capacity=256,resident=True):
        self.compact_prefill_graphs={}
        self.verification_graphs=[];self.verification_head=None;self.verification_tokens=0
        expected=dict(model_type='mistral',hidden_size=4096,intermediate_size=14336,
                      num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,
                      vocab_size=32768)
        for key,value in expected.items():
            if config.get(key)!=value:raise ValueError('Unsupported Mistral7B profile: '+key)
        if config.get('tie_word_embeddings',False):raise ValueError('Mistral7B profile requires its untied head')
        if capacity>config.get('max_position_embeddings',0):raise ValueError('Context exceeds model RoPE capacity')
        quant=config.get('quantization',{})
        if quant.get('bits')!=4 or quant.get('group_size')!=64 or quant.get('mode','affine')!='affine':
            raise ValueError('Mistral7B profile requires affine INT4/group64')
        if 'affine_coefficient_dtype=fp16' not in garnet.backend_info():
            raise ValueError('Mistral7B requires the native FP16 coefficient engine profile')
        # The inherited legacy "bf16_head" flag selects the native head route;
        # packed U32 weights use the engine's validated FP16 coefficient type.
        self._building_native_resident=bool(resident and 'prepared_dense_prefill' in garnet.backend_info())
        try:
            DenseMetalDecoder.__init__(self,checkpoint,config,batch,capacity,
                bf16_head=True,resident=resident,bf16_projections=True,
                required_checkpoint_format='mlx_mistral_affine_int4_g64_fp16')
        finally:self._building_native_resident=False

    def prepare_prefill(self,chunk=16):
        if self.session is None:raise ValueError('Prepare Mistral profiles on a resident engine')
        self.enable_prefill(chunk)
        self.prefill_compile_ms=0
        # Compile the bounded chunk family during engine setup. Actual prompt
        # length and tail masks remain request data; no context-sized padding.
        if self.dense_prefill_limit:
            for size in reversed(self._prefill_profiles()):
                if size<=chunk:self._resident_prefill_graphs(size)
        else:self._resident_prefill_graphs(chunk)
        self.profile_prepare_compile_ms=self.prefill_compile_ms
        return True

    def prepare_compact_prefill(self,chunk=128):
        if self.batch!=4 or self.session is None or 'compact_owner_prefill' not in garnet.backend_info():
            raise ValueError('Compact owner prefill requires the native B4 dense profile extension')
        self.prepare_prefill(chunk)
        import tensor
        from xModel.qwen.metal_dense.layers import T,DensePrefillLayer
        width=self.config['hidden_size']
        for size in reversed(self._prefill_profiles()):
            if size>chunk:continue
            fallback=self._resident_prefill_graphs(size)
            for rows in (1,2):
                graphs=[]
                for idx,layer in enumerate(self.layers):
                    key=str(rows)+'-'+str(size)+'-'+str(idx)
                    graph=self.compact_prefill_graphs.get(key)
                    if graph is None:
                        states=[[rows]+list(shape[1:]) for shape in layer['shapes'][1:3]]
                        shapes=[[rows,size,width]]+states+[[rows,size,3],[rows,size],[rows,size]]
                        symbols=[tensor.input('mistral_compact_'+key+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
                        @T.fusion(name='mistral_compact_prefill',boundary='required')
                        def forward(hidden,key,value,positions,slots,active):
                            return DensePrefillLayer(hidden,key,value,positions,slots,active,self.config,idx,rows,size,self.capacity,'full_attention')
                        graph=garnet.compiled_graph()
                        try:
                            self._build_prefill_graph(graph,forward(*symbols),symbols,layer['graph'])
                            self.session.prepare_prefill(graph,idx)
                        except:graph.close();raise
                        self.compact_prefill_graphs[key]=graph
                    graphs.append(graph)
                self.session.prepare_compact_prefill(graphs,fallback)
        return True

    def prepare_verification(self,tokens=4):
        # The native verifier retains ordinary FP32 decode math, owns K/V span
        # backups and commits only the target-accepted prefix. This prepares
        # target verification; it does not create a trained proposal model.
        if self.session is None or 'native_greedy_dense_verification' not in garnet.backend_info():
            raise ValueError('Prepared native dense verification extension required')
        if type(tokens) is not int or tokens<1 or self.batch*tokens>4:
            raise ValueError('Mistral verification supports up to four physical token rows')
        if self.verification_tokens:
            if self.verification_tokens!=tokens:raise ValueError('Verification profile already prepared')
            return True
        import tensor
        from xModel.qwen.metal_dense.layers import T,DensePrefillLayer,norm
        staged=[];head=None
        try:
            width=self.config['hidden_size']
            for idx,layer in enumerate(self.layers):
                shapes=[[self.batch,tokens,width]]+layer['shapes'][1:3]+[[self.batch,tokens,3],[self.batch,tokens],[self.batch,tokens]]
                symbols=[tensor.input('mistral_verify_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
                @T.fusion(name='mistral_verify_'+str(idx),boundary='required')
                def forward(hidden,key,value,positions,slots,active):
                    return DensePrefillLayer(hidden,key,value,positions,slots,active,self.config,idx,self.batch,tokens,self.capacity,'full_attention')
                graph=garnet.compiled_graph();staged.append(graph)
                graph.build_verification_shared(forward(*symbols),symbols,self.checkpoint,layer['graph'])
            symbol=tensor.input('mistral_verify_head',shape=[self.batch*tokens,width],dtype=tensor.float32)
            @T.fusion(name='mistral_verify_head',boundary='required')
            def project(hidden):
                return norm(hidden,'model.norm',self.config)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
            head=garnet.compiled_graph();head.build_shared(project(symbol),[symbol],self.checkpoint,self.head)
            self.session.prepare_verification(staged,head)
            self.verification_graphs=staged;self.verification_head=head;self.verification_tokens=tokens
        except:
            for graph in staged:graph.close()
            if head is not None:head.close()
            raise
        return True

    def close(self):
        try:DenseMetalDecoder.close(self)
        finally:
            for graph in getattr(self,'compact_prefill_graphs',{}).values():graph.close()
            self.compact_prefill_graphs={}
            for graph in getattr(self,'verification_graphs',[]):graph.close()
            head=getattr(self,'verification_head',None)
            if head is not None:head.close()
            self.verification_graphs=[];self.verification_head=None;self.verification_tokens=0
