"""Independent golden bytes, all BF16 bit patterns and hostile byte streams."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from bf16_block_codec import HEADER, decode, encode

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists(), 'Preserve codec test evidence'
checks = []


def roundtrip(label, words):
    original = np.array(words, dtype='<u2')
    before = original.copy()
    blob = encode(original)
    restored = decode(blob, int(original.size))
    assert np.array_equal(before, original), 'Encoder mutated its input'
    assert original.tobytes() == restored.tobytes(), label
    assert blob == encode(original), 'Encoding is not deterministic'
    checks.append({'name': label, 'words': original.size, 'stored_bytes': len(blob),
                   'source_sha256': hashlib.sha256(original.tobytes()).hexdigest()})
    return blob


def reject(label, blob, count):
    try:
        decode(blob, count)
    except ValueError:
        checks.append({'name': label, 'rejected': True})
    else:
        raise AssertionError('Malformed stream admitted: ' + label)


positive = roundtrip('positive-one golden', np.full(256, 0x3f80, dtype=np.uint16))
assert positive == struct.pack('<IBBH', 0, 127, 1, 256) + bytes(384)
negative = roundtrip('negative-one golden', np.full(256, 0xbf80, dtype=np.uint16))
assert negative == struct.pack('<IBBH', 0, 127, 1, 256) + bytes((0, 8, 128)) * 128
fraction = roundtrip('one-and-half golden', np.full(256, 0x3fc0, dtype=np.uint16))
assert fraction == struct.pack('<IBBH', 0, 127, 1, 256) + bytes((0, 4, 64)) * 128
all_words = np.arange(65536, dtype=np.uint16)
roundtrip('all65536 BF16 bit patterns sorted', all_words)
roundtrip('all65536 BF16 bit patterns permuted', all_words[(np.arange(65536) * 40503 + 17) % 65536])
for length in (0, 1, 2, 255, 256, 257, 511, 512, 513):
    roundtrip(f'boundary length {length}', np.resize(all_words, length))
special = np.full(256, 0x3f80, dtype=np.uint16)
special[:8] = [0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xff81]
roundtrip('special values within normal block', special)
exception = np.full(256, 0x3f80, dtype=np.uint16)
exception[5] = 1
encoded_exception = roundtrip('indexed exception packed block', exception)
assert HEADER.unpack_from(encoded_exception)[2] == 1 and len(encoded_exception) == 394
raw_block = (np.arange(256, dtype=np.uint16) << 7) | 17
raw_blob = roundtrip('wide exponent raw fallback', raw_block)
assert raw_blob == struct.pack('<IBBH', 0, 0, 0, 256) + raw_block.astype('<u2').tobytes()
for cut in (0, 1, 4, 7, 8, 9, 200, 391):
    reject(f'truncated packed {cut}', positive[:cut], 256)
reject('extra trailing bytes', positive + b'\0\0', 256)
reject('wrong logical count', positive, 255)
reject('negative logical count', b'', -1)
reject('nonempty zero count', b'\0', 0)
for label, location, value in (('unknown mode', 5, 9), ('wrong count', 6, 1), ('underflow exponent', 4, 0)):
    bad = bytearray(negative if label == 'underflow exponent' else positive)
    bad[location] = value
    if label == 'underflow exponent':
        bad[8] |= 1
    reject(label, bad, 256)
bad = bytearray(encoded_exception)
# Word5 is upper12bits of its24-bit pair. Replace its exception index0 with1.
pair_start = 8 + (5 // 2) * 3
word = int.from_bytes(bad[pair_start:pair_start + 3], 'little')
word = (word & 4095) | (31 << 12)
bad[pair_start:pair_start + 3] = word.to_bytes(3, 'little')
reject('exception index out of range', bad, 256)
double = bytearray(encode(np.full(512, 0x3f80, dtype=np.uint16)))
double[8:12] = (0).to_bytes(4, 'little')
reject('overlapping payloads', double, 512)
double[8:12] = (385).to_bytes(4, 'little')
reject('payload gap', double, 512)
result = {'all_passed': True, 'checks': checks,
          'scope': 'CPU byte codec only. Golden bytes independent of decoder, all65536 BF16 '
                   'patterns including signed zeros/subnormals/NaNs/infinities, exact bit '
                   'round trips, immutable inputs, deterministic encoding and malformed '
                   'streams. No learned-weight GPU decoder or inference/performance claim.'}
a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'all_passed': True, 'checks': len(checks)}))
