"""Separate B1/T128 target-prefill layout; no runtime or performance admission.

The qualified T4 decode and T32 prefill contracts remain unchanged. This dialect
keeps their canonical packed BF16 KV/FP32 GDN storage and emits T128 KV deltas.
"""
from .kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout

CONTRACT = 'garnet_qwen38_bf16_prefill128_delta_study_v1'
TARGET_WORDS = 44564480


def build_bf16_prefill128_layout(batch=1, capacity=4096, chunk=128, *, retain_linear_inputs=False):
    if any(type(value) is not int for value in (batch, capacity, chunk)):
        raise ValueError('Explicit integer target-prefill study dimensions required')
    if (batch, capacity, chunk) != (1, 4096, 128) or retain_linear_inputs is not False:
        raise ValueError('Separate B1/cap4096/T128 prefill without replay payload required')
    result = build_bf16_low_batch_layout(1, 4096, 32, retain_linear_inputs=False)
    result.update(contract=CONTRACT, chunk=128, hidden_width=128 * 5120)
    target = result['hidden_width']
    for layer in result['layers']:
        layer['target_offset'] = target
        if layer['kind'] == 'full_attention':
            layer['output_state1_width'] = layer['output_state2_width'] = 128 * 4 * 256
        target += layer['output_state1_width'] + layer['output_state2_width']
    assert target == TARGET_WORDS
    result.update(target_state_end=target, output_width=target)
    return result


def target_kv_index(layout, layer_index, which, token, head, dimension):
    """FP32 target-delta word offset; unrelated to packed canonical byte offsets."""
    if layout != build_bf16_prefill128_layout():
        raise ValueError('Exact unqualified T128 metadata required')
    if any(type(value) is not int for value in (layer_index, token, head, dimension)):
        raise ValueError('Integer target indices required')
    if not (0 <= layer_index < 64 and layer_index % 4 == 3 and 0 <= token < 128 and
            0 <= head < 4 and 0 <= dimension < 256) or which not in ('key', 'value'):
        raise ValueError('Target KV index outside T128 study')
    layer = layout['layers'][layer_index]
    return layer['target_offset'] + (layer['output_state1_width'] if which == 'value' else 0) + (token * 4 + head) * 256 + dimension
