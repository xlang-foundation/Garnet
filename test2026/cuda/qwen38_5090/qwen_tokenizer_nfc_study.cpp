// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0
// Private CPU study entry point; deliberately absent from the Garnet target.
#include "qwen_unicode_normalization.h"
#include <cstdint>
#include <exception>
#if defined(_WIN32)
#define STUDY_EXPORT __declspec(dllexport)
#else
#define STUDY_EXPORT
#endif
extern "C" STUDY_EXPORT int GarnetStudyNormalizeNfc(
    const std::uint32_t* input, int count, std::uint32_t* output, int capacity, int* size)
{
    if (count<0 || capacity<0 || !size || (count && !input)) return 1;
    try {
        std::vector<std::uint32_t> values;
        if (count) values.assign(input,input+count);
        auto normalized=Garnet::Tokenization::Unicode::NormalizeNfc(values);
        *size=static_cast<int>(normalized.size());
        if (capacity<*size || (*size && !output)) return 2;
        for (int i=0; i<*size; ++i) output[i]=normalized[i];
        return 0;
    } catch (const std::exception&) { return 3; }
}
