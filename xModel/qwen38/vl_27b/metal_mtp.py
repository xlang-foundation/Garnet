# SPDX-License-Identifier: Apache-2.0
"""Explicit auxiliary MTP graph; no speculative serving admission.

Inputs pair normalized target state at t with the embedding/position of t+1.
The raw auxiliary checkpoint retains +1 norm offsets. Output hidden is already
MTP-normalized and must use only the shared target vocabulary projection.
The caller owns independent K/V and must correct draft history after rejection.
"""
import garnet
import tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T, norm, mlp, add
from xModel.qwen35.vl_0_8b.prefill_layer import reshape, concat
from xModel.qwen35.vl_0_8b.full_attention import unpack


@T.fusion(name='qwen38_metal_mtp', role='decoder_layer', boundary='required')
def MetalMTPChunk(target, embedding, key, value, positions, slots, active,
                  config, batch, chunk, capacity):
    width = config['hidden_size']
    target = target * T.binary_op('qwen35_mask_chunk') * active
    embedding = embedding * T.binary_op('qwen35_mask_chunk') * active
    target = norm(reshape(target, [batch*chunk, width]), 'mtp.pre_fc_norm_hidden', config)
    embedding = norm(reshape(embedding, [batch*chunk, width]), 'mtp.pre_fc_norm_embedding', config)
    hidden = concat(embedding, target) * T.unary_op('qwen35_mtp_fc')
    normalized = reshape(norm(hidden, 'mtp.layers.0.input_layernorm', config), [batch, chunk, width])
    bound = normalized * T.binary_op('qwen35_attention_bind_key') * key
    bound = bound * T.binary_op('qwen35_attention_bind_value') * value
    bound = bound * T.binary_op('qwen35_attention_bind_positions') * positions
    bound = bound * T.binary_op('qwen35_attention_bind_slots') * slots
    bound = bound * T.binary_op('qwen35_attention_bind_active') * active
    rope = config['rope_parameters']
    packed = bound * T.unary_op('qwen35_attention_chunk', weight_prefix='mtp.layers.0.self_attn',
        num_heads=config['num_attention_heads'], num_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'], rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),
        rope_theta=rope['rope_theta'], mrope_t=rope['mrope_section'][0],
        mrope_h=rope['mrope_section'][1], mrope_w=rope['mrope_section'][2], eps=config['rms_norm_eps'])
    packed_config = dict(config)
    packed_config['hidden_size'] = chunk*width
    attention, key, value = unpack(packed, packed_config, capacity)
    residual = add(hidden, reshape(attention, [batch*chunk, width]))
    hidden = add(residual, mlp(norm(residual, 'mtp.layers.0.post_attention_layernorm', config), 'mtp.layers.0.mlp'))
    hidden = norm(hidden, 'mtp.norm', config)
    return concat(concat(reshape(hidden, [batch, chunk*width]), reshape(key, [batch, -1])), reshape(value, [batch, -1]))


class MetalMTPProgram:
    def __init__(self, checkpoint, config, batch=1, capacity=4096, chunk=4, source=None, prefill=False):
        if 'raw_bf16_mtp_graph' not in garnet.backend_info():
            raise ValueError('Explicit native MTP extension is required')
        if type(prefill) is not bool:
            raise ValueError('MTP prefill mode must be explicit bool')
        if prefill and 'raw_bf16_mtp_prefill_profiles' not in garnet.backend_info():
            raise ValueError('Explicit MTP prefill profile extension is required')
        row_limit = 128 if prefill else 4
        if any(type(x) is not int for x in (batch, capacity, chunk)) or batch not in (1, 2, 4) or not 1 <= chunk <= row_limit//batch or not 1 <= capacity <= 4096:
            raise ValueError('MTP exceeds explicit decode/prefill row or capacity bounds')
        if config.get('mtp_num_hidden_layers', 1) != 1:
            raise ValueError('Exactly one auxiliary MTP layer is required')
        width = config['hidden_size']
        cache = [batch, config['num_key_value_heads'], capacity, config['head_dim']]
        self.shapes = [[batch, chunk, width], [batch, chunk, width], cache, cache,
                       [batch, chunk, 3], [batch, chunk], [batch, chunk]]
        symbols = [tensor.input('metal_mtp_'+str(i), shape=s, dtype=tensor.float32) for i, s in enumerate(self.shapes)]
        self.graph = garnet.compiled_graph()
        try:
            captured = MetalMTPChunk(*symbols, config, batch, chunk, capacity)
            if prefill:
                self.graph.build_mtp_prefill(captured, symbols, checkpoint, None if source is None else source.graph)
            elif source is None:
                self.graph.build_mtp(captured, symbols, checkpoint)
            else:
                self.graph.build_shared(captured, symbols, checkpoint, source.graph)
        except:
            self.close()
            raise

    def run(self, inputs):
        if self.graph is None:
            raise ValueError('MTP program is closed')
        return self.graph.run(inputs)

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None
