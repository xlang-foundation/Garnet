"""Actual MLX embedding-row oracle; run only between measured campaigns.

Reads selected packed rows directly, never loads the full learned model. The
MLX call matches QuantizedEmbedding.__call__, including inferred BF16 output.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    index_path = args.checkpoint/'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    tokens = [0, 1, 198, 727, 12305, 71093, 248319]
    names = ['language_model.model.embed_tokens.'+suffix for suffix in ('weight', 'scales', 'biases')]
    arrays, evidence = [], []
    for name in names:
        path = args.checkpoint/index[name]
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            entry = header[name]
            count, width = entry['shape']
            expected_dtype, element_size = ('U32', 4) if name.endswith('.weight') else ('BF16', 2)
            if entry['dtype'] != expected_dtype or count != 248320:
                raise ValueError('Unexpected pinned embedding layout')
            if entry['data_offsets'][1]-entry['data_offsets'][0] != count*width*element_size:
                raise ValueError('Invalid embedding payload size')
            rows, digests = [], []
            for token in tokens:
                stream.seek(8+header_size+entry['data_offsets'][0]+token*width*element_size)
                raw = stream.read(width*element_size)
                if len(raw) != width*element_size:
                    raise ValueError('Truncated embedding row')
                digests.append(hashlib.sha256(raw).hexdigest())
                if expected_dtype == 'U32':
                    rows.append(list(struct.unpack('<'+'I'*width, raw)))
                else:
                    bits = struct.unpack('<'+'H'*width, raw)
                    rows.append([struct.unpack('<f', struct.pack('<I', value << 16))[0] for value in bits])
        arrays.append(mx.array(rows, dtype=mx.uint32 if expected_dtype == 'U32' else mx.bfloat16))
        evidence.append(dict(name=name, file=path.name, shape=entry['shape'], dtype=entry['dtype'],
                             selected_row_sha256=digests))
    if [value.shape[1] for value in arrays] != [640, 80, 80]:
        raise ValueError('Expected width5120/group64 embedding')
    result = mx.dequantize(arrays[0], scales=arrays[1], biases=arrays[2], group_size=64, bits=4, mode='affine')
    mx.eval(result)
    if result.dtype != mx.bfloat16:
        raise ValueError('Actual MLX embedding output is not BF16')
    report = dict(scope='Actual selected-row MLX oracle; not whole-model numerical parity',
                  checkpoint=str(index_path.resolve()), tokens=tokens, width=5120,
                  mlx_version=importlib.metadata.version('mlx'), output_dtype=str(result.dtype),
                  index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(), tensors=evidence,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  expected=result.astype(mx.float32).tolist())
    with args.output.open('x') as stream:
        json.dump(report, stream)
        stream.write('\n')


if __name__ == '__main__':
    main()
