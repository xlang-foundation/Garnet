#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline actual-weight Mistral layer0 MLP export, not whole-model inference.

Exact group64 palettes preserve the native FP16 dequantized weights. This
changes neither the checkpoint nor its quantization coefficients. Core ML
intermediate/accumulator arithmetic is a separate validation requirement.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import resource
import struct
import time
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--representation', choices=('lut', 'fp16'), default='lut')
    p.add_argument('--rows', type=int, choices=(1, 128, 512), default=128)
    p.add_argument('--input-mode', choices=('embedding', 'normalized_embedding'), default='embedding')
    p.add_argument('--arithmetic', choices=('fp16', 'fp32_prefill_control'), default='fp16')
    p.add_argument('--inner-tile', type=int, choices=(0, 64, 256), default=0)
    p.add_argument('--palette-row-group', type=int, choices=(1, 16), default=1)
    p.add_argument('--intermediate-tile', type=int, choices=(0,1024,2048), default=0)
    p.add_argument('--activation-scale', type=int, choices=(1,16,64,256), default=1)
    a = p.parse_args()
    assert ct.__version__ == '9.0'
    assert not a.intermediate_tile or (a.representation == 'fp16' and a.inner_tile == 0 and a.arithmetic == 'fp16')
    assert a.activation_scale == 1 or a.intermediate_tile, 'Activation scaling requires the explicit channel-tile profile'
    assert a.arithmetic == 'fp16' or a.representation == 'fp16', 'FP32 control uses dense exact FP16 values expanded to FP32'
    assert a.palette_row_group == 1 or (a.inner_tile == 64 and a.representation == 'lut')
    assert sha(a.checkpoint) == 'd7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    with a.checkpoint.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    native = json.loads((a.assets/'native-rows/native.json').read_text())
    records, weights = [], {}

    def mapped(name, dtype):
        info = header[name]
        assert info['dtype'] == {'<u4': 'U32', '<f2': 'F16'}[dtype]
        assert info['data_offsets'][1]-info['data_offsets'][0] == np.prod(info['shape'])*np.dtype(dtype).itemsize
        return np.memmap(a.checkpoint, mode='r', dtype=dtype,
                         offset=8+length+info['data_offsets'][0], shape=tuple(info['shape']))

    for number, (name, n, k) in enumerate((('gate_proj', 14336, 4096), ('up_proj', 14336, 4096), ('down_proj', 4096, 14336))):
        base = 'model.layers.0.mlp.'+name
        packed, scales, biases = mapped(base+'.weight', '<u4'), mapped(base+'.scales', '<f2'), mapped(base+'.biases', '<f2')
        assert packed.shape == (n, k//8) and scales.shape == biases.shape == (n, k//64)
        record = native[number]
        assert record['tensor'] == base+'.weight' and record['rows'] == list(range(n)) and record['columns'] == k and record['dtype'] == 'F16'
        indices = np.empty((n, k), dtype=np.uint8)
        for nibble in range(8):
            indices[:, nibble::8] = (packed >> (4*nibble)) & 15
        # q * fp16(scale) is exactly representable in fp32 for q0..15.
        # The fp32 addition then matches native fma before the final fp16 cast.
        palette = (scales.astype(np.float32)[..., None]*np.arange(16, dtype=np.float32)
                   + biases.astype(np.float32)[..., None]).astype(np.float16)
        assert np.isfinite(palette).all()
        reference = np.memmap(record['rounded'], mode='r', dtype='<f4', shape=(n, k))
        matched = 0
        for begin in range(0, n, 32):
            end = min(n, begin+32)
            restored = np.take_along_axis(palette[begin:end], indices[begin:end].reshape(end-begin, k//64, 64), axis=2).reshape(end-begin, k)
            expected = np.asarray(reference[begin:end], dtype=np.float16)
            assert np.array_equal(restored.view(np.uint16), expected.view(np.uint16)), (name, begin)
            matched += restored.size
        if a.representation == 'lut':
            weights[name] = (indices.astype(types.np_uint4_dtype).reshape(n, k, 1, 1),
                             palette.reshape(n, k//64, 1, 1, 16, 1))
        else:
            weights[name] = np.asarray(reference, dtype=np.float16).reshape(n, k, 1, 1)
        records.append(dict(tensor=base+'.weight', shape=[n, k], native_fp16_words_matched=matched,
                            native_rounded_sha256=sha(Path(record['rounded'])),
                            packed_weight_sha256=hashlib.sha256(packed).hexdigest(),
                            palette_sha256=hashlib.sha256(palette).hexdigest(),
                            palette_entries=palette.size, representation=a.representation))
        print('native_weight_parity', name, matched, flush=True)
        del packed, scales, biases, indices, palette, reference

    source = native[3]
    assert source['tensor'] == 'model.embed_tokens.weight' and source['columns'] == 4096 and len(source['rows']) >= min(a.rows, 128)
    embedding = np.memmap(source['rounded'], mode='r', dtype='<f4', shape=(len(source['rows']), 4096))
    input_rows = np.asarray(embedding[np.arange(a.rows) % len(embedding)], dtype='<f4')
    if a.input_mode == 'normalized_embedding':
        norm = mapped('model.layers.0.post_attention_layernorm.weight', '<f2')
        assert norm.shape == (4096,)
        config = json.loads((a.checkpoint.parent/'config.json').read_text())
        variance = np.mean(input_rows*input_rows, axis=1, dtype=np.float32, keepdims=True)
        input_rows *= np.reciprocal(np.sqrt(variance+np.float32(config['rms_norm_eps'])))
        input_rows *= norm.astype(np.float32)
        del norm
    input_path = a.output/'input.f32'
    input_rows.tofile(input_path)
    # Channel-first 1x1 convolutions keep the sequence dimension separate from
    # the learned projection channels, following Apple's ANE transformer layout.
    tile_weight_words = 0
    scaled_up_roundtrip_words = 0
    def projection(x, name):
        nonlocal tile_weight_words
        if not a.inner_tile:
            if a.representation == 'lut':
                indices, palette = weights[name]
                w = mb.constexpr_lut_to_dense(indices=indices, lut=palette, name=name+'_palette')
            else:
                w = weights[name]
            if a.arithmetic == 'fp32_prefill_control':
                w = mb.cast(x=w, dtype='fp32')
            return mb.conv(x=x, weight=w, name=name)
        n, k = (weights[name][0] if a.representation == 'lut' else weights[name]).shape[:2]
        parts = []
        for begin in range(0, k, a.inner_tile):
            end = begin+a.inner_tile
            label = name+'_k'+str(begin)
            part = mb.slice_by_index(x=x, begin=[0, begin, 0, 0], end=[1, end, 1, a.rows])
            if a.representation == 'lut':
                indices, palette = weights[name]
                q = np.asarray(indices[:, begin:end, 0, 0], dtype=np.uint8)
                local = palette[:, begin//64:end//64].reshape(n, a.inner_tile//64, 16)
                expected = np.take_along_axis(local, q.reshape(n, a.inner_tile//64, 64), axis=2).reshape(n, a.inner_tile)
                if a.palette_row_group == 1:
                    table = np.ascontiguousarray(local.reshape(n, -1))
                    remapped = q.astype(np.uint16)+(np.arange(a.inner_tile, dtype=np.uint16)//64)[None, :]*16
                else:
                    table = np.ascontiguousarray(local.reshape(n//16, 256))
                    remapped = q.astype(np.uint16)+(np.arange(n, dtype=np.uint16)%16)[:, None]*16
                row_table = np.arange(n)//a.palette_row_group
                restored = table[row_table[:, None], remapped]
                assert np.array_equal(restored.view(np.uint16), expected.view(np.uint16)), label
                tile_weight_words += restored.size
                bits = int(np.log2(table.shape[1]))
                assert table.shape[1] == 2**bits
                dtype = {4: types.np_uint4_dtype, 6: types.np_uint6_dtype, 8: np.uint8}[bits]
                w = mb.constexpr_lut_to_dense(indices=remapped.astype(dtype).reshape(n, a.inner_tile, 1, 1),
                    lut=table.reshape(n//a.palette_row_group, 1, 1, 1, table.shape[1], 1), name=label+'_palette')
            else:
                w = np.ascontiguousarray(weights[name][:, begin:end])
                tile_weight_words += n*a.inner_tile
            if a.arithmetic == 'fp32_prefill_control':
                w = mb.cast(x=w, dtype='fp32')
            parts.append(mb.conv(x=part, weight=w, name=label))
        # A balanced reduction limits the number of rounded additions on each
        # path. This changes accumulation order and needs separate validation.
        level = 0
        while len(parts) > 1:
            parts = [mb.add(x=parts[i], y=parts[i+1], name=name+'_sum'+str(level)+'_'+str(i))
                     if i+1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
            level += 1
        return parts[0]

    def channel_branches(x):
        nonlocal tile_weight_words, scaled_up_roundtrip_words
        parts = []
        for begin in range(0, 14336, a.intermediate_tile):
            end = begin + a.intermediate_tile
            gate_weight = np.ascontiguousarray(weights['gate_proj'][begin:end])
            up_weight = np.ascontiguousarray(weights['up_proj'][begin:end])
            scaled = (up_weight * np.float16(a.activation_scale)).astype(np.float16)
            restored = (scaled / np.float16(a.activation_scale)).astype(np.float16)
            assert np.isfinite(scaled).all() and np.array_equal(restored.view(np.uint16),up_weight.view(np.uint16))
            scaled_up_roundtrip_words += up_weight.size
            up_weight = scaled
            down_weight = np.ascontiguousarray(weights['down_proj'][:, begin:end])
            tile_weight_words += gate_weight.size + up_weight.size + down_weight.size
            label = 'channels_' + str(begin)
            gate = mb.conv(x=x, weight=gate_weight, name=label+'_gate')
            up = mb.conv(x=x, weight=up_weight, name=label+'_up')
            activated = mb.mul(x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up)
            parts.append(mb.conv(x=activated, weight=down_weight, name=label+'_down'))
        level = 0
        while len(parts) > 1:
            parts = [mb.add(x=parts[i], y=parts[i+1], name='channel_sum_'+str(level)+'_'+str(i))
                     if i+1 < len(parts) else parts[i] for i in range(0,len(parts),2)]
            level += 1
        return mb.mul(x=parts[0], y=np.float16(1/a.activation_scale), name='y')

    @mb.program(input_specs=[mb.TensorSpec((1, 4096, 1, a.rows), dtype=types.fp16)], opset_version=ct.target.iOS18)
    def program(x):
        if a.intermediate_tile:
            return channel_branches(x)
        if a.arithmetic == 'fp32_prefill_control':
            x = mb.cast(x=x, dtype='fp32')
        gate = projection(x, 'gate_proj')
        up = projection(x, 'up_proj')
        activated = mb.mul(x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up)
        if a.arithmetic == 'fp32_prefill_control':
            # Match the strict native prefill's FP16 down-projection operand.
            activated = mb.cast(x=mb.cast(x=activated, dtype='fp16'), dtype='fp32')
            return mb.cast(x=projection(activated, 'down_proj'), dtype='fp16', name='y')
        return mb.identity(x=projection(activated, 'down_proj'), name='y')

    print('convert', a.representation, a.rows, flush=True)
    model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16 if a.arithmetic == 'fp16' else ct.precision.FLOAT32,
                       compute_units=ct.ComputeUnit.CPU_ONLY,
                       skip_model_load=True)
    spec = model.get_spec()
    counts = collections.Counter()
    for block in spec.mlProgram.functions['main'].block_specializations.values():
        counts.update(op.type for op in block.operations)
    expected_convs = 3*(14336//a.intermediate_tile) if a.intermediate_tile else (3 if not a.inner_tile else 2*(4096//a.inner_tile)+14336//a.inner_tile)
    assert counts['conv'] == expected_convs and counts['constexpr_lut_to_dense'] == (expected_convs if a.representation == 'lut' else 0), counts
    assert not (a.inner_tile or a.intermediate_tile) or tile_weight_words == 176160768
    assert not a.intermediate_tile or scaled_up_roundtrip_words == 58720256
    package = a.output/'mistral_mlp.mlpackage'
    model.save(str(package))
    print('compile', flush=True)
    compiled = ct.models.utils.compile_model(str(package), destination_path=str(a.output/'mistral_mlp.mlmodelc'))
    io_bytes = a.rows*4096*2
    profile = dict(compute_units='cpu_ane', function='main', maximum_io_bytes=2*io_bytes,
                   maximum_state_bytes=0, require_ane_plan=False,
                   inputs=dict(x=dict(dtype='float16', maximum_shape=[1, 4096, 1, a.rows])),
                   outputs=dict(y=dict(dtype='float16', maximum_shape=[1, 4096, 1, a.rows])))
    (a.output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    report = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint),
                  coremltools=ct.__version__, numpy=np.__version__, rows=a.rows, hidden=4096,
                  intermediate=14336, layer=0, representation=a.representation, arithmetic=a.arithmetic,
                  inner_tile=a.inner_tile, intermediate_tile=a.intermediate_tile, activation_scale=a.activation_scale,
                  scaled_up_roundtrip_words_matched=scaled_up_roundtrip_words,
                  scale_cancellation="up weights multiplied by exact power of two; reciprocal final output scale",
                  palette_row_group=a.palette_row_group,
                  projection_convolutions=expected_convs, remapped_tile_weight_words_matched=tile_weight_words,
                  reduction='balanced_intermediate_channel_branches' if a.intermediate_tile else ('balanced' if a.inner_tile else 'single_conv'),
                  compiled=str(compiled), operation_counts=dict(counts), weights=records,
                  input=str(input_path), input_sha256=sha(input_path),
                  input_mode=a.input_mode,
                  input_scope='Frozen prompt embedding proxy (repeated for512 rows); normalized_embedding applies real layer0 post-attention norm weights and RMS epsilon. Not actual post-attention hidden activations.',
                  input_rms=float(np.sqrt(np.mean(input_rows.astype(np.float64)**2))),
                  export_seconds=time.monotonic()-started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='One actual-weight Mistral MLP block; exact FP16 weight values. Arithmetic mode is explicit: FP16 or FP32 prefill control with rounded down operand. Optional up-weight scaling is exactly reversible with reciprocal output scaling, but changes intermediate range and rounding. No full-model/quality/ANE execution/performance claim.', files={})
    for path in a.output.rglob('*'):
        if path.is_file():
            report['files'][str(path.relative_to(a.output))] = sha(path)
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('representation', 'rows', 'operation_counts', 'export_seconds', 'max_rss_bytes')}), flush=True)


if __name__ == '__main__':
    main()
