"""Derive an eight-channel BF16 attention fixture for vector cache kernels."""
import argparse
import json
from pathlib import Path
import random
import struct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    original=json.loads(args.fixture.read_text())
    raw=Path(original['checkpoint']).read_bytes();length=struct.unpack('<Q',raw[:8])[0]
    metadata=json.loads(raw[8:8+length]);payload=raw[8+length:]
    width=original['config']['hidden_size'];rng=random.Random(35038)
    replacements={'self_attn.q_proj.weight':[32,width], 'self_attn.k_proj.weight':[8,width],
                  'self_attn.v_proj.weight':[8,width], 'self_attn.o_proj.weight':[width,16],
                  'self_attn.q_norm.weight':[8], 'self_attn.k_norm.weight':[8]}
    header={};data=bytearray()
    for name,entry in metadata.items():
        if name=='__metadata__':continue
        suffix=name.removeprefix('model.language_model.layers.1.')
        if suffix in replacements:
            shape=replacements[suffix];count=1
            for dimension in shape:count*=dimension
            block=bytearray()
            for _ in range(count):
                bits=struct.unpack('<I',struct.pack('<f',rng.gauss(0,.07)))[0]
                block.extend(struct.pack('<H',((bits+0x7fff+((bits>>16)&1))>>16)&0xffff))
            dtype='BF16'
        else:
            shape=entry['shape'];dtype=entry['dtype'];start,end=entry['data_offsets'];block=payload[start:end]
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[len(data),len(data)+len(block)])
        data.extend(block)
    args.output.mkdir(parents=True,exist_ok=False)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    checkpoint=args.output/'weights.safetensors';checkpoint.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
    config=dict(original['config'],head_dim=8)
    config['rope_parameters']=dict(original['config']['rope_parameters'],mrope_section=[2,1,1])
    fixture=dict(config=config,batch=original['batch'],capacity=original['capacity'],checkpoint=str(checkpoint.resolve()),
                 scope='Eight-channel vector KV fixture; sequential decode supplies the chunk-state reference')
    (args.output/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')


if __name__=='__main__':main()
