"""Mistral7B text adapter over shared dense native Metal operators.

Research bring-up only; generation/coding/performance admission is separate.
"""
# SPDX-License-Identifier: Apache-2.0
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder


class MistralMetalDecoder(DenseMetalDecoder):
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
        DenseMetalDecoder.__init__(self,checkpoint,config,batch,capacity,
            bf16_head=True,resident=resident,bf16_projections=True,
            required_checkpoint_format='mlx_mistral_affine_int4_g64_fp16')

    def prepare_prefill(self,chunk=16):
        if self.session is None:raise ValueError('Prepare Mistral profiles on a resident engine')
        self.enable_prefill(chunk)
        self.prefill_compile_ms=0
        # Compile the bounded chunk family during engine setup. Actual prompt
        # length and tail masks remain request data; no context-sized padding.
        self._resident_prefill_graphs(chunk)
        self.profile_prepare_compile_ms=self.prefill_compile_ms
        return True
