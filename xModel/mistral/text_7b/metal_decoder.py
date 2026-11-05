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
