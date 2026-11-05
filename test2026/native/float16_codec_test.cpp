// SPDX-License-Identifier: Apache-2.0
// CPU-only differential driver; compare its complete output with NumPy's
// independent float16 codec, including subnormal and halfway boundaries.
#include "../../src/model/float16_codec.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: float16_codec_test F32_INPUT FP16_OUTPUT DECODE_OUTPUT");
        std::ofstream decoded(argv[3], std::ios::binary);
        if (!decoded) throw std::runtime_error("cannot open decode output");
        size_t roundtrips = 0;
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            float value = Garnet::Float16ToFloat32(uint16_t(bits));
            decoded.write(reinterpret_cast<const char*>(&value), sizeof(value));
            if (std::isfinite(value)) {
                if (Garnet::Float32ToFloat16RNE(value) != bits) throw std::runtime_error("finite FP16 roundtrip mismatch");
                ++roundtrips;
            }
        }
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() < 0 || input.tellg() % sizeof(float)) throw std::runtime_error("invalid FP32 input");
        const auto count = size_t(input.tellg()) / sizeof(float); input.seekg(0);
        std::vector<float> values(count); input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
        if (!input) throw std::runtime_error("incomplete FP32 input");
        std::ofstream output(argv[2], std::ios::binary);
        if (!output) throw std::runtime_error("cannot open encode output");
        for (float value : values) {
            auto half = Garnet::Float32ToFloat16RNE(value);
            output.write(reinterpret_cast<const char*>(&half), sizeof(half));
        }
        if (!decoded || !output) throw std::runtime_error("incomplete output");
        std::cout << "float16_codec decoded=65536 finite_roundtrips=" << roundtrips << " encoded=" << count << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
