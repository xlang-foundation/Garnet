"""Byte-only comparison of BF16 carriers and BF16 values stored as FP32."""


def expand_bf16_carrier(raw):
    if not isinstance(raw,bytes) or len(raw)%4:raise ValueError('Expected complete FP32-carrier words')
    # Little-endian BF16 words become high16 bits of little-endian FP32 words.
    # No floating arithmetic, native offsets or checkpoint code is shared here.
    expanded=bytearray(len(raw)*2)
    expanded[2::4]=raw[0::2];expanded[3::4]=raw[1::2]
    return bytes(expanded)


def require_bf16_roundtrip(raw):
    if not isinstance(raw,bytes) or len(raw)%4:raise ValueError('Expected complete FP32 words')
    if any(raw[0::4]) or any(raw[1::4]):raise ValueError('FP32 reference cache contains a value outside BF16 roundtrip precision')
    return raw
