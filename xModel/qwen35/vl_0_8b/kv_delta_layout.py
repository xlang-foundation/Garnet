# SPDX-License-Identifier: Apache-2.0
"""Unwired B4/T4 KV-delta layout/dispatch contract; no GPU imports.

Canonical state offsets stay unchanged. A future opt-in target may return full
linear-attention states plus only current-chunk full-attention K/V. Committing
those deltas is a separate transaction-final operation, never plugin aliasing.
This module alone does not enable that contract in a decoder or native runtime.
"""

def validate_kv_delta_profile(enabled, batch, capacity, prefill_chunk, decode_chunk,
                              *, external_compressed_weights, retain_linear_inputs,
                              deferred_state_commit, shared_output_storage,
                              joined_mlp_projections, host_embedding_lookup,
                              embedded_projection_weights=False,
                              join_input_projections=False):
    if type(enabled) is not bool:
        raise ValueError('KV delta opt-in must be a boolean')
    if not enabled:
        return
    if any(type(v) is not int for v in (batch, capacity, prefill_chunk, decode_chunk)) or (batch, capacity, prefill_chunk, decode_chunk) != (4, 4096, 4, 4):
        raise ValueError('KV delta study requires exactly B4/cap4096/prefillT4/decodeT4')
    flags = (external_compressed_weights, retain_linear_inputs, deferred_state_commit,
             shared_output_storage, joined_mlp_projections, host_embedding_lookup)
    if any(v is not True for v in flags) or embedded_projection_weights is not False or join_input_projections is not False:
        raise ValueError('KV delta requires shared compressed/retained/deferred/shared-output/joined-MLP/host-embedding mode only')


def build_kv_delta_layout(canonical, batch=4, capacity=4096, chunk=4, hidden=5120):
    """Return a new compact layout from the existing canonical layout metadata.

    Widths/offsets are FP32 words per batch row. Full-attention native readers
    must use canonical_state_width as their batch stride, rather than request
    materialized contiguous K/V slices. No canonical metadata is mutated.
    """
    if any(type(v) is not int for v in (batch, capacity, chunk, hidden)) or (batch, capacity, chunk, hidden) != (4, 4096, 4, 5120):
        raise ValueError('Unqualified KV delta layout geometry')
    if not isinstance(canonical, (list, tuple)) or len(canonical) != 64:
        raise ValueError('KV delta layout requires the pinned 64-layer structure')
    layers = []
    canonical_cursor = 0
    compact_cursor = chunk * hidden
    replay_cursor = 0
    payload_width = 0
    for index, old in enumerate(canonical):
        if not isinstance(old, dict):
            raise ValueError('Invalid canonical layer record')
        kind = 'full_attention' if index % 4 == 3 else 'linear_attention'
        if type(old.get('index')) is not int or old['index'] != index or old.get('kind') != kind:
            raise ValueError('Unqualified canonical layer order')
        if type(old.get('offset')) is not int or old['offset'] != canonical_cursor:
            raise ValueError('Canonical state offset is not contiguous')
        first = old.get('state1_shape')
        second = old.get('state2_shape')
        wanted = ([batch, 4, capacity, 256], [batch, 4, capacity, 256]) if kind == 'full_attention' else ([batch, 10240, 4], [batch, 48, 128, 128])
        for shape, want in zip((first, second), wanted):
            if not isinstance(shape, (list, tuple)) or list(shape) != want or any(type(v) is not int for v in shape):
                raise ValueError('Unqualified canonical state shape')
        widths = []
        for shape in (first, second):
            width = 1
            for value in shape[1:]:
                width *= value
            widths.append(width)
        if any(type(old.get('state'+str(i)+'_width')) is not int or old['state'+str(i)+'_width'] != widths[i-1] for i in (1, 2)):
            raise ValueError('Canonical state width disagrees with its shape')
        canonical_cursor += sum(widths)
        output_shapes = [[batch, chunk, 4, 256]] * 2 if kind == 'full_attention' else [list(first), list(second)]
        output_widths = [chunk * 4 * 256] * 2 if kind == 'full_attention' else widths
        layer = {'index': index, 'kind': kind, 'canonical_offset': old['offset'],
                 'canonical_state1_width': widths[0], 'canonical_state2_width': widths[1],
                 'output_offset': compact_cursor, 'output_state1_width': output_widths[0],
                 'output_state2_width': output_widths[1],
                 'output_state1_shape': list(output_shapes[0]), 'output_state2_shape': list(output_shapes[1])}
        compact_cursor += sum(output_widths)
        if kind == 'linear_attention':
            layer['replay_offset'] = replay_cursor
            replay_cursor += sum(widths)
            layer['payload_width'] = chunk * (first[1] + second[1] * second[3] + 2 * second[1])
            payload_width += layer['payload_width']
        else:
            layer.update(key_offset=old['offset'], value_offset=old['offset'] + widths[0], cache_row_width=widths[0])
        layers.append(layer)
    payload_cursor = compact_cursor
    for layer in layers:
        if layer['kind'] == 'linear_attention':
            layer['payload_offset'] = payload_cursor
            payload_cursor += layer['payload_width']
        else:
            layer['cache_batch_stride'] = canonical_cursor
    assert payload_cursor == compact_cursor + payload_width
    return {'batch': batch, 'capacity': capacity, 'chunk': chunk, 'hidden_width': chunk * hidden,
            'canonical_state_width': canonical_cursor,
            'compact_state_width': compact_cursor - chunk * hidden,
            'linear_replay_width': replay_cursor, 'payload_width': payload_width,
            'output_width': payload_cursor, 'layers': layers}


def choose_kv_overlay_strategy(slots, active):
    """Choose compact0 for all-short controls, otherwise legacy-geometry1.

    A future decoder can select between two captured engines sharing output
    storage, using controls it already owns on the CPU. No device readback or
    changing a captured graph's native environment flags is needed.512 is the
    declared study cutoff, not an automatically tuned or universal threshold.
    """
    if not isinstance(slots, (list, tuple)) or not isinstance(active, (list, tuple)) or len(slots) != 16 or len(active) != 16:
        raise ValueError('KV delta controls require B4/T4')
    long_prefix = False
    for row in range(4):
        used = set()
        for t in range(4):
            slot, flag = slots[row*4+t], active[row*4+t]
            if type(flag) is not int or flag not in (0, 1) or type(slot) is not int or not 0 <= slot <= 4096:
                raise ValueError('KV delta controls require validated integral slots/flags')
            if flag:
                if slot == 4096 or slot in used:
                    raise ValueError('Active KV slots must be unique and within capacity')
                used.add(slot)
                long_prefix = long_prefix or slot + 1 > 512
    return int(long_prefix)
