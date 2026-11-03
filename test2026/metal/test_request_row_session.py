"""Independent mixed linear/KV reference for replacing A while B/C/D continue."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pi5b'))
from reference_decoder import linear_block, attention_block

runtime, build, source, output = sys.argv[1:5]
output = Path(output)
output.mkdir(exist_ok=False)
rng = np.random.default_rng(350327)
batch, width, capacity = 4, 11, 8
c = dict(hidden_size=width, linear_num_key_heads=1, linear_num_value_heads=3,
         linear_key_head_dim=7, linear_value_head_dim=5, linear_conv_kernel_dim=4,
         num_attention_heads=2, num_key_value_heads=1, head_dim=6, rms_norm_eps=1e-6,
         rope_parameters=dict(partial_rotary_factor=1, rope_theta=10000,
                              mrope_section=[1, 1, 1], mrope_interleaved=True))
common = dict(input_norm='input_layernorm.weight', post_norm='post_attention_layernorm.weight',
              mlp_gate='mlp.gate_proj.weight', mlp_up='mlp.up_proj.weight', mlp_down='mlp.down_proj.weight')
common_shapes = dict(input_norm=(width,), post_norm=(width,), mlp_gate=(17, width),
                     mlp_up=(17, width), mlp_down=(width, 17))
families = [
    (dict(qkv='linear_attn.in_proj_qkv.weight', z='linear_attn.in_proj_z.weight',
          b_proj='linear_attn.in_proj_b.weight', a_proj='linear_attn.in_proj_a.weight',
          conv='linear_attn.conv1d.weight', a_log='linear_attn.A_log',
          dt_bias='linear_attn.dt_bias', norm='linear_attn.norm.weight', out_proj='linear_attn.out_proj.weight'),
     dict(qkv=(29, width), z=(15, width), b_proj=(3, width), a_proj=(3, width),
          conv=(29, 1, 4), a_log=(3,), dt_bias=(3,), norm=(5,), out_proj=(width, 15))),
    (dict(q='self_attn.q_proj.weight', k='self_attn.k_proj.weight', v='self_attn.v_proj.weight',
          out='self_attn.o_proj.weight', q_norm='self_attn.q_norm.weight', k_norm='self_attn.k_norm.weight'),
     dict(q=(24, width), k=(6, width), v=(6, width), out=(width, 12), q_norm=(6,), k_norm=(6,))),
]
header, payload, weights = {}, bytearray(), []
for layer, (names, shapes) in enumerate(families):
    names = dict(common, **names)
    shapes = dict(common_shapes, **shapes)
    decoded = {}
    for name, shape in shapes.items():
        value = rng.normal(0, .07, shape).astype(np.float32)
        if name == 'norm':
            value += 1
        if name in ('a_log', 'dt_bias', 'norm'):
            dtype, raw = 'F32', value.astype('<f4').tobytes()
        else:
            bits = value.view(np.uint32)
            bf16 = ((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
            value = (bf16.astype(np.uint32)<<16).view(np.float32)
            dtype, raw = 'BF16', bf16.tobytes()
        decoded[name] = value
        header[f'model.language_model.layers.{layer}.{names[name]}'] = dict(
            dtype=dtype, shape=list(shape), data_offsets=[len(payload), len(payload)+len(raw)])
        payload.extend(raw)
    weights.append(decoded)
encoded = json.dumps(header, separators=(',', ':')).encode()
encoded += b' '*((-len(encoded))%8)
checkpoint = output/'weights.safetensors'
checkpoint.write_bytes(struct.pack('<Q', len(encoded))+encoded+payload)
states = [[np.zeros((batch, 29, 4), np.float32), np.zeros((batch, 3, 7, 5), np.float32)],
          [np.zeros((batch, 1, capacity, 6), np.float32), np.zeros((batch, 1, capacity, 6), np.float32)]]
slots = np.zeros(batch, np.float32)
steps = []
for index in range(7):
    reset = 0 if index == 2 else -1
    if reset >= 0:
        for parts in states:
            for part in parts:
                part[reset] = 0
        slots[reset] = 0
    hidden = rng.normal(0, .25, (batch, width)).astype(np.float32)
    active = np.ones(batch, np.float32)
    if index in (3, 4):
        active[2] = 0
    positions = np.repeat(slots[:, None], 3, axis=1)
    result = dict(reset_row=reset, hidden=hidden.ravel().tolist(), active=active.tolist(),
                  slots=slots.tolist(), positions=positions.ravel().tolist())
    x, a, b = linear_block(hidden, *states[0], active, weights[0], 1, 3, 7, 5)
    states[0] = [a.astype(np.float32), b.astype(np.float32)]
    x, a, b = attention_block(x, *states[1], positions, slots, active, weights[1], 2, 1, 6, 6, [1, 1, 1], 10000)
    states[1] = [a.astype(np.float32), b.astype(np.float32)]
    result.update(expected_hidden=x.ravel().tolist(), expected_predictions=np.argmax(x, axis=-1).tolist(), expected_states=[
        [part.ravel().tolist() for part in parts] for parts in states])
    steps.append(result)
    slots += active
for with_head in (False, True):
    fixture = output/('fixture-head-'+str(int(with_head))+'.json')
    fixture.write_text(json.dumps(dict(config=c, batch=batch, capacity=capacity, with_head=with_head,
                                       checkpoint=str(checkpoint), steps=steps)))
    subprocess.run([runtime, str(Path(__file__).with_name('run_request_row_session.py')),
                    str(fixture), build, source], check=True)
