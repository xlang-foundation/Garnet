"""Synthetic converted-affine checkpoint for native temporal verification parity.

This is a correctness gate, never a model speed or coding-quality measurement.
Arguments: xlang3 runtime, native build directory, source root, new output folder.
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

runtime, build, source, destination = sys.argv[1:5]
output = Path(destination)
output.mkdir(exist_ok=False)
rng = np.random.default_rng(380027)
header, payload = {}, bytearray()

def put(name, dtype, shape, data):
    header[name] = dict(dtype=dtype, shape=list(shape),
                        data_offsets=[len(payload), len(payload)+len(data)])
    payload.extend(data)

def bf16(value):
    value = np.asarray(value, dtype='<f4')
    bits = value.view(np.uint32)
    return ((bits+0x7fff+((bits >> 16) & 1)) >> 16).astype('<u2')

def vector(name, shape, center=0):
    put(name, 'BF16', shape, bf16(rng.normal(center, .035, shape)).tobytes())

def affine(base, rows, columns):
    assert columns % 64 == 0
    values = rng.integers(0, 16, (rows, columns), dtype=np.uint32)
    packed = np.zeros((rows, columns//8), dtype='<u4')
    for nibble in range(8):
        packed |= values[:, nibble::8] << (nibble*4)
    scales = rng.uniform(.003, .008, (rows, columns//64)).astype(np.float32)
    put(base+'.weight', 'U32', packed.shape, packed.tobytes())
    put(base+'.scales', 'BF16', scales.shape, bf16(scales).tobytes())
    put(base+'.biases', 'BF16', scales.shape, bf16(-7.5*scales).tobytes())

width = 64
config = dict(hidden_size=width, intermediate_size=128, vocab_size=128,
              linear_num_key_heads=1, linear_num_value_heads=4,
              linear_key_head_dim=128, linear_value_head_dim=16,
              linear_conv_kernel_dim=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
              tie_word_embeddings=False,
              rope_parameters=dict(partial_rotary_factor=1, rope_theta=10000,
                                   mrope_section=[6, 5, 5], mrope_interleaved=True))
affine('language_model.model.embed_tokens', 128, width)
affine('language_model.lm_head', 128, width)
vector('language_model.model.norm.weight', (width,), 1)
for layer in range(4):
    p = f'language_model.model.layers.{layer}.'
    for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
        vector(p+name, (width,), 1)
    for name, rows, cols in (('mlp.gate_proj', 128, width), ('mlp.up_proj', 128, width),
                             ('mlp.down_proj', width, 128)):
        affine(p+name, rows, cols)
    if layer % 2 == 0:
        p += 'linear_attn.'
        for name, rows, cols in (('in_proj_qkv', 320, width), ('in_proj_z', 64, width),
                                 ('in_proj_a', 4, width), ('in_proj_b', 4, width),
                                 ('out_proj', width, 64)):
            affine(p+name, rows, cols)
        vector(p+'conv1d.weight', (320, 4, 1))
        vector(p+'A_log', (4,), -1)
        vector(p+'dt_bias', (4,), -.5)
        vector(p+'norm.weight', (16,), 1)
    else:
        p += 'self_attn.'
        for name, rows, cols in (('q_proj', 256, width), ('k_proj', 64, width),
                                 ('v_proj', 64, width), ('o_proj', width, 128)):
            affine(p+name, rows, cols)
        vector(p+'q_norm.weight', (32,), 1)
        vector(p+'k_norm.weight', (32,), 1)
encoded = json.dumps(header, separators=(',', ':')).encode()
encoded += b' '*((-len(encoded)) % 8)
checkpoint = output/'weights.safetensors'
checkpoint.write_bytes(struct.pack('<Q', len(encoded))+encoded+payload)
(output/'config.json').write_text(json.dumps(dict(model_type='qwen3_5',
    quantization=dict(bits=4, group_size=64, mode='affine'), text_config=config)))
fixture = output/'fixture.json'
fixture.write_text(json.dumps(dict(config=config, checkpoint=str(checkpoint))))
(output/'checkpoint-sha256.txt').write_text(hashlib.sha256(checkpoint.read_bytes()).hexdigest()+'\n')
subprocess.run([runtime, '-B', str(Path(__file__).with_name('run_compiled_verification.py')),
                str(fixture), build, source], check=True)
