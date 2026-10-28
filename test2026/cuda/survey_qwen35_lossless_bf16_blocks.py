"""Read-only projection-weight range survey; run only with model processes closed.

This estimates a lossless 12-bit BF16 format with per-block exponent bases,
indexed full-word exceptions and a raw fallback. It does not create compressed
weights, change model precision, benchmark kernels or qualify inference.
"""
import argparse
import hashlib
import json
import mmap
import struct
import subprocess
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('checkpoint', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists(), 'Preserve weight survey evidence'
guard = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
    'if(Get-Process xlang3,llama-server,nvcc,cl,ptxas,ninja,cmake,msbuild,link,compute-sanitizer,nsys '
    '-ErrorAction SilentlyContinue){exit 1}'], capture_output=True)
assert guard.returncode == 0, 'Close native models/compilers/diagnostics before checkpoint scans'
guest = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'ps', '-e', '-o', 'comm='], capture_output=True)
assert guest.returncode == 0 and b'VLLM' not in guest.stdout and b'llama-server' not in guest.stdout

records, files = [], []
block_words, header_bytes = 256, 8
for path in sorted(a.checkpoint.glob('*.safetensors')):
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        raw_header = stream.read(length)
        entries = json.loads(raw_header)
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            for name, entry in entries.items():
                if name == '__metadata__' or not name.startswith('model.language_model.layers.'):
                    continue
                if entry['dtype'] != 'BF16' or len(entry['shape']) != 2:
                    continue
                start, end = entry['data_offsets']
                words = (end - start) // 2
                assert words * 2 == end - start
                estimated, exceptions, raw_blocks, packed_blocks = 0, 0, 0, 0
                histogram = np.zeros(block_words + 1, dtype=np.int64)
                for offset in range(0, words, 1 << 20):
                    count = min(1 << 20, words - offset)
                    full = count // block_words * block_words
                    source = np.frombuffer(mapping, dtype='<u2', count=full,
                        offset=8 + length + start + offset * 2).reshape(-1, block_words)
                    exponents = ((source >> 7) & 255).astype(np.int16)
                    bases = exponents.max(axis=1)
                    delta = bases[:, None] - exponents
                    escaped = (delta > 14) | (exponents == 255)
                    counts = escaped.sum(axis=1)
                    histogram += np.bincount(counts, minlength=block_words + 1)
                    packed = block_words * 12 // 8 + header_bytes + counts * 2
                    raw = block_words * 2 + header_bytes
                    use_packed = packed < raw
                    estimated += int(np.minimum(packed, raw).sum())
                    exceptions += int(counts[use_packed].sum())
                    packed_blocks += int(use_packed.sum())
                    raw_blocks += int((~use_packed).sum())
                    # Algebraic reconstruction of every nonexception BF16 word.
                    normal = ~escaped
                    sign_fraction = ((source >> 8) & 128) | (source & 127)
                    codes = (sign_fraction << 4) | np.minimum(delta, 15).astype(np.uint16)
                    rebuilt = ((codes >> 11) << 15) | (((bases[:, None] - (codes & 15)) & 255) << 7) | ((codes >> 4) & 127)
                    assert np.array_equal(rebuilt[normal].astype(np.uint16), source[normal])
                    if count != full:
                        estimated += (count - full) * 2 + header_bytes
                        raw_blocks += 1
                    del source, exponents, bases, delta, escaped, counts, packed, use_packed, normal, sign_fraction, codes, rebuilt
                records.append({'tensor': name, 'shape': entry['shape'], 'source_bytes': end - start,
                    'estimated_format_bytes': estimated, 'packed_blocks': packed_blocks,
                    'raw_blocks': raw_blocks, 'stored_exception_words': exceptions,
                    'exception_count_histogram': histogram.tolist()})
    assert before == (path.stat().st_size, path.stat().st_mtime_ns), 'Checkpoint changed during survey'
    files.append({'file': str(path), 'size_mtime_ns': before,
                  'header_sha256': hashlib.sha256(raw_header).hexdigest()})
assert records, 'No projection matrices found'
original = sum(row['source_bytes'] for row in records)
estimated = sum(row['estimated_format_bytes'] for row in records)
result = {'checkpoint': str(a.checkpoint), 'files': files, 'records': records,
    'projection_source_bytes': original, 'estimated_format_bytes': estimated,
    'estimated_stored_bytes_saved': original - estimated,
    'estimated_format_over_source': estimated / original,
    'block_words': block_words, 'header_bytes_per_block': header_bytes,
    'matrix_directory_and_alignment_excluded': True,
    'all_nonexception_bf16_words_algebraically_reconstructed': True,
    'scope': 'Read-only stored BF16 projection matrices, each counted once. Twelve-bit sign/fraction '
        'plus exponent-delta records; delta15 uses an eight-bit index into exact full-word exceptions. '
        'Eight-byte block header with per-matrix relative offsets, raw fallback when smaller, raw tail. '
        'Whole-model matrix directory and allocation alignment are excluded from this estimate. Every nonexception raw BF16 '
        'word checked by algebraic reconstruction; exceptions retain original words by definition. '
        'No physical byte codec/GPU decoder, cache/VRAM/runtime ownership, CUDA kernel speed, model '
        'output/quality or2x qualification. Embedding/head, vision, MTP and nonmatrix weights excluded.'}
a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('projection_source_bytes', 'estimated_format_bytes',
    'estimated_stored_bytes_saved', 'estimated_format_over_source')}))
