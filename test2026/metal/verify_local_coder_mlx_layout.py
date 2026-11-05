"""Inspect safetensors headers against the 27B xModel without loading weights.

This proves representation/shape compatibility only, not native execution or
model quality. Payload contents are pinned separately by the asset hash manifest.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


def inspect(checkpoint, factory, include_vision=False):
    config = json.loads((checkpoint/'config.json').read_text())
    spec = importlib.util.spec_from_file_location('local_coder_contract', factory)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    contract.validate_config(config)
    quant = config['quantization']
    if (quant['bits'], quant['group_size'], quant['mode']) != (4, 64, 'affine'):
        raise ValueError('Expected affine INT4/group64')
    mapping = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    for name in sorted(set(mapping.values())):
        path = checkpoint/name
        if path.resolve().parent != checkpoint.resolve():
            raise ValueError('Shard is outside checkpoint directory')
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 64*1024*1024 or size+8 > path.stat().st_size:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(size))
        headers[name] = header
    def check(name, dtype, shape):
        value = headers[mapping[name]][name]
        if value['dtype'] != dtype or value['shape'] != list(shape):
            raise ValueError(f'Wrong layout for {name}: {value}')
        begin, end = value['data_offsets']
        elements = 1
        for dim in shape:
            elements *= dim
        if end-begin != elements*{'U32': 4, 'BF16': 2}[dtype]:
            raise ValueError('Wrong payload length for '+name)
        return end-begin
    shapes = contract.projection_shapes(config)
    shapes['model.language_model.embed_tokens'] = (config['text_config']['vocab_size'], config['text_config']['hidden_size'])
    packed_bytes = coefficient_bytes = 0
    for canonical, (rows, columns) in shapes.items():
        name = ('language_model.model.'+canonical[len('model.language_model.'):]
                if canonical.startswith('model.language_model.') else 'language_model.'+canonical)
        packed_bytes += check(name+'.weight', 'U32', (rows, columns//8))
        coefficient_bytes += check(name+'.scales', 'BF16', (rows, columns//64))
        coefficient_bytes += check(name+'.biases', 'BF16', (rows, columns//64))
    text = config['text_config']
    for layer, kind in enumerate(text['layer_types']):
        base = f'language_model.model.layers.{layer}.'
        for norm in ('input_layernorm', 'post_attention_layernorm'):
            check(base+norm+'.weight', 'BF16', (text['hidden_size'],))
        if kind == 'linear_attention':
            channels = 2*text['linear_num_key_heads']*text['linear_key_head_dim']+text['linear_num_value_heads']*text['linear_value_head_dim']
            check(base+'linear_attn.conv1d.weight', 'BF16', (channels, 4, 1))
            check(base+'linear_attn.norm.weight', 'BF16', (text['linear_value_head_dim'],))
            for key in ('A_log', 'dt_bias'):
                check(base+'linear_attn.'+key, 'BF16', (text['linear_num_value_heads'],))
        else:
            for key in ('q_norm', 'k_norm'):
                check(base+'self_attn.'+key+'.weight', 'BF16', (text['head_dim'],))
    check('language_model.model.norm.weight', 'BF16', (text['hidden_size'],))
    result = dict(status='header_contract_pass_not_execution_qualification', projections=len(shapes)-1,
                embedding_tables=1, packed_bytes=packed_bytes, scale_bias_bytes=coefficient_bytes,
                total_quantized_bytes=packed_bytes+coefficient_bytes,
                xmodel_sha256=hashlib.sha256(factory.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
                vision='Not qualified by this text-layout inspection')
    if include_vision:
        vision = config['vision_config']
        hidden, middle = vision['hidden_size'], vision['intermediate_size']
        unit = vision['spatial_merge_size']**2
        tensors = {
            'patch_embed.proj.weight': (hidden, vision['temporal_patch_size'], vision['patch_size'], vision['patch_size'], vision['in_channels']),
            'patch_embed.proj.bias': (hidden,),
            'pos_embed.weight': (vision['num_position_embeddings'], hidden),
            'merger.norm.weight': (hidden,), 'merger.norm.bias': (hidden,),
            'merger.linear_fc1.weight': (unit*hidden, unit*hidden),
            'merger.linear_fc1.bias': (unit*hidden,),
            'merger.linear_fc2.weight': (vision['out_hidden_size'], unit*hidden),
            'merger.linear_fc2.bias': (vision['out_hidden_size'],),
        }
        for layer in range(vision['depth']):
            for name, shape in {
                'attn.qkv.weight': (3*hidden, hidden), 'attn.qkv.bias': (3*hidden,),
                'attn.proj.weight': (hidden, hidden), 'attn.proj.bias': (hidden,),
                'mlp.linear_fc1.weight': (middle, hidden), 'mlp.linear_fc1.bias': (middle,),
                'mlp.linear_fc2.weight': (hidden, middle), 'mlp.linear_fc2.bias': (hidden,),
                'norm1.weight': (hidden,), 'norm1.bias': (hidden,),
                'norm2.weight': (hidden,), 'norm2.bias': (hidden,),
            }.items():
                tensors[f'blocks.{layer}.{name}'] = shape
        result['vision_bf16_bytes'] = sum(check('vision_tower.'+name, 'BF16', shape) for name, shape in tensors.items())
        result['vision_tensors_checked'] = len(tensors)
        result['vision'] = 'Header layouts verified including converted Conv3d; native numerical and end-to-end gates pending'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-vision', action='store_true')
    parser.add_argument('--factory', type=Path, default=Path(__file__).resolve().parents[2]/'xModel/qwen38/vl_27b/model.py')
    args = parser.parse_args()
    result = inspect(args.checkpoint, args.factory, args.include_vision)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
