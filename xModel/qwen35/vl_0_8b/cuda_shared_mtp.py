# SPDX-License-Identifier: Apache-2.0
"""Shared BF16 MTP graph; caller binds one serial workspace/output pool."""
import json
import garnet
import tensor
from .decoder_layer import T

class CUDASharedMTPProgram:
    def __init__(self, checkpoint, config, batch, capacity, chunk_size, projection_workspace_bytes, io):
        if chunk_size not in (1, 4, 8, 16) or batch not in (1, 2, 4, 8):
            raise ValueError('Unsupported experiment graph profile')
        self.graph = None
        self.size = chunk_size
        self.weights = []
        self.weight_names = ['mtp.fc.weight'] + ['mtp.layers.0.' + name + '.weight' for name in (
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
        h = config['hidden_size']
        cache = [batch, config['num_key_value_heads'], capacity, config['head_dim']]
        shapes = [[batch, chunk_size, h], [batch, chunk_size, h], cache, cache,
                  [batch, chunk_size, 3], [batch, chunk_size], [batch, chunk_size]]
        symbols = [tensor.input('shared_mtp_' + str(i), shape=s, dtype=tensor.float32) for i, s in enumerate(shapes)]
        self.weight_bytes = 0
        for i, name in enumerate(self.weight_names):
            info = json.loads(io.weight_info(name))
            assert info['dtype'] == 'BF16' and len(info['shape']) == 2
            self.weight_bytes += info['bytes']
            self.weights.append(io.weight(name))
            symbols.append(tensor.input('shared_mtp_weight_' + str(i), shape=info['shape'], dtype=tensor.bfloat16))
        names = self.weight_names
        rope = config['rope_parameters']

        @T.fusion(name='qwen35_shared_mtp_program', role='decoder_layer', boundary='required')
        def forward(target, embedding, key, value, positions, slots, active, *weights):
            bound = target * T.binary_op('qwen35_mtp_bind_embedding') * embedding
            bound = bound * T.binary_op('qwen35_attention_bind_key') * key
            bound = bound * T.binary_op('qwen35_attention_bind_value') * value
            bound = bound * T.binary_op('qwen35_attention_bind_positions') * positions
            bound = bound * T.binary_op('qwen35_attention_bind_slots') * slots
            bound = bound * T.binary_op('qwen35_attention_bind_active') * active
            return bound * T.unary_op('qwen35_mtp_decoder_chunk', num_heads=config['num_attention_heads'],
                num_kv_heads=config['num_key_value_heads'], head_dim=config['head_dim'],
                rotary_dim=int(config['head_dim'] * rope['partial_rotary_factor']), rope_theta=rope['rope_theta'],
                mrope_t=rope['mrope_section'][0], mrope_h=rope['mrope_section'][1], mrope_w=rope['mrope_section'][2],
                eps=config['rms_norm_eps'], projection_workspace_bytes=projection_workspace_bytes,
                external_weights={names[i]: weights[i] for i in range(len(names))})

        self.graph = garnet.compiled_graph()
        self.graph.enable_user_managed_workspace(True)
        try:
            self.graph.build(forward(*symbols), symbols, checkpoint)
        except:
            self.close()
            raise

    def run(self, inputs):
        if self.graph is None or len(inputs) != 7:
            raise ValueError('Closed experiment graph or wrong inputs')
        return self.graph.run_device(inputs + self.weights)

    def close(self):
        if self.graph is not None:
            self.graph.close()
        self.graph = None
        self.weights = []


