// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "qwen35_tree_metadata.cuh"
namespace Garnet {
struct Qwen35TreeCommitConfig {
    int batch=0,nodes=0,capacity=0,key_heads=0,value_heads=0,kernel=0,kv_heads=0,head_dim=0;
    int linear_layers=0,full_layers=0;
    int64_t state_width=0,packed_width=0,hidden_width=0,status_offset=0,metadata_offset=0;
};
// ThreeINT64 words; no padding/opaque float transport.
struct Qwen35TreeCommitLayer {int64_t state=0,payload=0,index=0;};
static_assert(sizeof(Qwen35TreeCommitLayer)==24,"Tree commit layout must be threeINT64 words");
struct Qwen35TreeCommitSelection {int prefix[8]{},last[8]{};};
bool ValidQwen35TreeCommitConfig(const Qwen35TreeCommitConfig&) noexcept;
size_t Qwen35TreeCommitWorkspace(const Qwen35TreeCommitConfig&) noexcept;
// Rebuilds private metadata, validates the COMPLETE finite packet/export/status
// and all layouts, then commits only selected active ancestors. Any GPU control
// error blocks ALL batch writes and sets status. Caller must synchronize/read
// status and poison canonical state on unknown launch/completion failures.
// Inputs/metadata/layouts/parameters must remain immutable until completion.
// Selection lives in device storage at a stable address, so captured graphs can
// change accepted paths/prefix counters between replays.
int RunQwen35TreeCommit(const Qwen35TreeCommitConfig&,const Qwen35TreeCommitSelection*,
    const float* packed,float* canonical,const float* parents,const float* active,const float* prefix,
    const Qwen35TreeCommitLayer* linear,const Qwen35TreeCommitLayer* full,const float* parameters,
    void* workspace,int* status,cudaStream_t) noexcept;
}
