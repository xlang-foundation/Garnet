// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace Garnet {
// Unwired copy-only prototype; tensor ownership and serving integration unproven.
struct Qwen35CompactKVConfig {int batch,kv,capacity,tokens,dim;};
struct Qwen35CompactKVSpans {int start[8]{},count[8]{};};
cudaError_t PackQwen35CompactKV(Qwen35CompactKVConfig,const float* key,const float* value,
    const float* slots,const float* active,float* compactKey,float* compactValue,int* invalidControl,cudaStream_t);
cudaError_t CommitQwen35CompactKV(Qwen35CompactKVConfig,Qwen35CompactKVSpans,
    const float* compactKey,const float* compactValue,float* key,float* value,cudaStream_t);
cudaError_t ReportQwen35CompactKVControl(const int* invalidControl,float* perBatch,int batch,cudaStream_t);
}
