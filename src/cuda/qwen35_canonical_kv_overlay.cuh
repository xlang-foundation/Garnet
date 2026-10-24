// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen35_attention_core.cuh"
#include <cstddef>
namespace Garnet {
// Separate experiment: input4/5 point at K/V offsets inside readonly canonical
// packed state. Batch stride is FP32 words, >=one cache row and <=INT_MAX.
// Caller validates complete spans/offsets, controls, unique active slots and
// non-overlap of output/workspace/inputs. Geometry remains exactly B4/T4.
int RunQwen35AttentionKVOverlayStrided(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int,std::size_t,cudaStream_t) noexcept;
// Explicit final write, after all readers. Accepted counts are validated active
// prefix lengths. Only selected K/V slots change; unrelated canonical state is
// untouched. This does not relax the existing Garnet plugin no-alias contract.
int CommitQwen35AttentionKVPrefixStrided(const Qwen35AttentionCoreConfig&,const void* const*,const float*,void*,void*,std::size_t,cudaStream_t) noexcept;
}
