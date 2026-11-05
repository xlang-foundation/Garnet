// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>

namespace Garnet {
// Exact IEEE binary16 expansion; no intermediate BF16 rounding. The numeric
// value of every finite FP16 coefficient is representable exactly in FP32.
inline float Float16ToFloat32(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 31u, fraction = value & 1023u, bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            int unbiased = -14;
            while (!(fraction & 1024u)) { fraction <<= 1; --unbiased; }
            bits = sign | (uint32_t(unbiased + 127) << 23) | ((fraction & 1023u) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((exponent + 112) << 23) | (fraction << 13);
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}

// Round-to-nearest, ties-to-even, including FP16 subnormals and signed zero.
// IEEE overflow returns infinity; callers admitting model data must reject it.
inline uint16_t Float32ToFloat16RNE(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 255u, fraction = bits & 0x7fffffu;
    if (exponent == 255) return uint16_t(sign | (fraction ? 0x7e00u | (fraction >> 13) : 0x7c00u));
    const int unbiased = int(exponent) - 127;
    if (unbiased > 15) return uint16_t(sign | 0x7c00u);
    if (unbiased >= -14) {
        const uint32_t rounded = fraction + 0xfffu + ((fraction >> 13) & 1u);
        return uint16_t(sign | (uint32_t(unbiased + 15) << 10) + (rounded >> 13));
    }
    if (unbiased < -25) return sign;
    const unsigned shift = unsigned(-unbiased - 1);
    const uint32_t significand = 0x800000u | fraction;
    const uint32_t rounded = significand + ((uint32_t(1) << (shift - 1)) - 1u) + ((significand >> shift) & 1u);
    return uint16_t(sign | (rounded >> shift));
}
}
