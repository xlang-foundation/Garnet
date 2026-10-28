"""Create immutable learned-weight byte-codec fixtures; no model/GPU execution."""
import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from bf16_block_codec import decode, encode

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
output = a.root / 'lossless-bf16-projection-v1-fixtures.json'
assert not output.exists(), 'Preserve fixture manifest'
guard = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
    'if(Get-Process xlang3,llama-server,nvcc,cl,ptxas,ninja,cmake,msbuild,link,compute-sanitizer,nsys '
    '-ErrorAction SilentlyContinue){exit 1}'], capture_output=True)
assert guard.returncode == 0, 'Close native models/compilers/diagnostics first'
records, checkpoint_headers = [], []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(label, words, inner, columns, names):
    raw_path = a.root / ('lossless-bf16-projection-v1-' + label + '.u16')
    packed_path = a.root / ('lossless-bf16-projection-v1-' + label + '.packed')
    assert not raw_path.exists() and not packed_path.exists(), 'Preserve fixture bytes'
    original = np.asarray(words, dtype='<u2').reshape(-1)
    raw, encoded = original.tobytes(), encode(original)
    assert decode(encoded, int(original.size)).tobytes() == raw, 'Learned physical-byte round trip failed'
    raw_path.write_bytes(raw)
    packed_path.write_bytes(encoded)
    records.append({'name': label, 'inner': inner, 'columns': columns, 'words': original.size,
        'source_tensors': names, 'raw_file': str(raw_path), 'raw_sha256': digest(raw),
        'packed_file': str(packed_path), 'packed_sha256': digest(encoded),
        'source_bytes': len(raw), 'packed_bytes': len(encoded),
        'all_words_byte_roundtrip_exact': True})
    print('BF16_BLOCK_FIXTURE', label, len(raw), len(encoded), flush=True)


for size in ('2b', '4b'):
    checkpoint = a.root / 'models' / ('Qwen3.5-' + size.upper())
    index = {}
    for path in sorted(checkpoint.glob('*.safetensors')):
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            raw_header = stream.read(length)
            header = json.loads(raw_header)
        checkpoint_headers.append({'file': str(path), 'size_mtime_ns': [path.stat().st_size, path.stat().st_mtime_ns],
                                   'header_sha256': digest(raw_header)})
        for name, entry in header.items():
            if name != '__metadata__':
                index[name] = (path, 8 + length, entry)

    def read(name):
        path, base, entry = index[name]
        assert entry['dtype'] == 'BF16' and len(entry['shape']) == 2
        begin, end = entry['data_offsets']
        with path.open('rb') as stream:
            stream.seek(base + begin)
            raw = stream.read(end - begin)
        assert len(raw) == end - begin
        return np.frombuffer(raw, dtype='<u2').reshape(entry['shape'])

    prefix = 'model.language_model.layers.0.'
    joined_names = [prefix + 'linear_attn.in_proj_' + name + '.weight' for name in ('qkv', 'z', 'b', 'a')]
    joined = np.concatenate([read(name) for name in joined_names], axis=0)
    save(size + '-joined', joined, joined.shape[1], joined.shape[0], joined_names)
    down_name = prefix + 'mlp.down_proj.weight'
    down = read(down_name)
    save(size + '-down', down, down.shape[1], down.shape[0], [down_name])
special = np.concatenate((np.arange(65536, dtype=np.uint16), np.array([0x8001], dtype=np.uint16)))
save('all-bits-tail', special, 0, 0, [])
for record in checkpoint_headers:
    path = Path(record['file'])
    assert [path.stat().st_size, path.stat().st_mtime_ns] == record['size_mtime_ns'], 'Checkpoint changed'
result = {'fixtures': records, 'checkpoint_headers': checkpoint_headers,
    'scope': 'Exact CPU physical-byte round trips for learned layer0 joined qkv/z/b/a and MLP '
        'down weights of both models, plus every BF16 bit pattern and odd tail. No model '
        'precision change, GPU decompression, GPU component speed or inference claim.'}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(output, flush=True)
