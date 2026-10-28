// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace Garnet {
// Prospective learned-MTP D3/T4 runner control. No model math is implemented here.
// The native owner must retain every operand and serialize the captured graph.
struct MTPDraftDeviceConfig {
    int batch=0,hidden=0,vocab=0,capacity=0,stopCount=0,stops[4]{};
};
struct MTPDraftDeviceState {
    int error=0,count=0,stage=0,anyAdvance=0,modelCalls=0,proposed=0;
    int active[8]{},remaining[8]{},positions[8]{},slots[8]{},stopped[8]{},advance[8]{},stepSlots[8]{};
};
static_assert(sizeof(MTPDraftDeviceState)<=512,"Bounded draft control storage");
// Controls: INT32[batch,4] = active,remaining,nextPosition,MTPnextSlot.
// Proposals: INT64[batch,3]. nextIds: INT64[batch] from unchanged indexed head.
cudaError_t InitMTPDraftDevice(const MTPDraftDeviceConfig&,const int*,MTPDraftDeviceState*,long long*,cudaStream_t) noexcept;
cudaError_t PrepareMTPDraftDevice(const MTPDraftDeviceConfig&,int,const long long*,long long*,MTPDraftDeviceState*,cudaGraphConditionalHandle,cudaStream_t) noexcept;
// Writes the same T4 inputs as _step/_call only if a learned model call is needed.
// Even inactive-row embeddings/positions retain the original caller semantics.
cudaError_t InputsMTPDraftDevice(const MTPDraftDeviceConfig&,int,const void*,const long long*,const MTPDraftDeviceState*,const float*,float*,float*,float*,float*,float*,cudaStream_t) noexcept;
cudaError_t FinishMTPDraftDevice(const MTPDraftDeviceConfig&,const long long*,long long*,MTPDraftDeviceState*,cudaStream_t) noexcept;
}
