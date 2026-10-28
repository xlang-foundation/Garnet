"""Experimental lossless BF16-bit storage. No floating arithmetic or model changes.

One matrix blob contains ceil(words/256) little-endian <IBBH> headers, then
payloads. Header: payload-relative uint32 offset, exponent base, mode, word
count. Mode0 is exact raw uint16; mode1 has256 twelve-bit records (384 bytes)
followed by exact uint16 exceptions. Record low4bits are exponent delta0..14;
high8bits are sign/fraction. Delta15 uses high8bits as exception index.
"""
import struct

import numpy as np

BLOCK_WORDS = 256
HEADER = struct.Struct('<IBBH')
PACKED_BYTES = 384


def encode(words):
    source = np.asarray(words)
    if source.dtype.kind != 'u' or source.dtype.itemsize != 2:
        raise ValueError('BF16 source must contain raw unsigned16-bit words')
    source = source.astype('<u2', copy=False).reshape(-1)
    headers, payloads, offset = [], [], 0
    for start in range(0, source.size, BLOCK_WORDS):
        block = source[start:start + BLOCK_WORDS]
        exponent = ((block >> 7) & 255).astype(np.int16)
        base = int(exponent.max())
        delta = base - exponent
        escaped = (delta > 14) | (exponent == 255)
        exceptions = block[escaped]
        raw = block.tobytes()
        if block.size == BLOCK_WORDS and PACKED_BYTES + exceptions.size * 2 < len(raw):
            high = ((block >> 8) & 128) | (block & 127)
            codes = (high << 4) | np.minimum(delta, 15).astype(np.uint16)
            codes[escaped] = (np.arange(exceptions.size, dtype=np.uint16) << 4) | 15
            paired = codes.reshape(-1, 2).astype(np.uint32)
            combined = paired[:, 0] | (paired[:, 1] << 12)
            packed = np.empty((combined.size, 3), dtype=np.uint8)
            packed[:, 0] = combined & 255
            packed[:, 1] = (combined >> 8) & 255
            packed[:, 2] = (combined >> 16) & 255
            payload = packed.tobytes() + exceptions.tobytes()
            mode = 1
        else:
            payload, mode, base = raw, 0, 0
        if offset + len(payload) >= 1 << 32:
            raise ValueError('Single matrix payload exceeds uint32 relative-offset capacity')
        headers.append(HEADER.pack(offset, base, mode, block.size))
        payloads.append(payload)
        offset += len(payload)
    return b''.join(headers + payloads)


def decode(blob, word_count):
    if not isinstance(word_count, int) or word_count < 0:
        raise ValueError('Invalid logical BF16 word count')
    raw = memoryview(blob)
    blocks = (word_count + BLOCK_WORDS - 1) // BLOCK_WORDS
    header_size = blocks * HEADER.size
    if len(raw) < header_size or (not word_count and len(raw)):
        raise ValueError('Truncated header or nonempty zero-word blob')
    output = np.empty(word_count, dtype='<u2')
    previous_end = 0
    for index in range(blocks):
        offset, base, mode, count = HEADER.unpack_from(raw, index * HEADER.size)
        expected_count = min(BLOCK_WORDS, word_count - index * BLOCK_WORDS)
        next_offset = (HEADER.unpack_from(raw, (index + 1) * HEADER.size)[0]
                       if index + 1 < blocks else len(raw) - header_size)
        if count != expected_count or offset != previous_end or not offset <= next_offset <= len(raw) - header_size:
            raise ValueError('Invalid count, overlap, gap or payload bound')
        payload = raw[header_size + offset:header_size + next_offset]
        if mode == 0:
            if base != 0 or len(payload) != count * 2:
                raise ValueError('Invalid raw block metadata or payload length')
            values = np.frombuffer(payload, dtype='<u2')
        elif mode == 1:
            if count != BLOCK_WORDS or not PACKED_BYTES <= len(payload) < BLOCK_WORDS * 2 or (len(payload) - PACKED_BYTES) % 2:
                raise ValueError('Invalid packed block size')
            triples = np.frombuffer(payload[:PACKED_BYTES], dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
            combined = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
            codes = np.empty(BLOCK_WORDS, dtype=np.uint16)
            codes[0::2], codes[1::2] = combined & 4095, combined >> 12
            delta = codes & 15
            escaped = delta == 15
            normal = ~escaped
            exceptions = np.frombuffer(payload[PACKED_BYTES:], dtype='<u2')
            indices = codes[escaped] >> 4
            if not np.array_equal(indices, np.arange(exceptions.size, dtype=np.uint16)):
                raise ValueError('Invalid or noncanonical exception indices')
            if np.any(delta[normal] > base) or np.any(base - delta[normal] == 255):
                raise ValueError('Invalid normal exponent')
            values = (((codes >> 11) << 15) | (((base - delta) & 255) << 7) | ((codes >> 4) & 127)).astype('<u2')
            values[escaped] = exceptions
        else:
            raise ValueError('Unknown block mode')
        begin = index * BLOCK_WORDS
        output[begin:begin + count] = values
        previous_end = next_offset
    return output
