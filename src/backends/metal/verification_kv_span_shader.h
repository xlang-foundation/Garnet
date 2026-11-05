// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Copy raw FP32 storage words so signed zeros and NaN payloads survive exactly.
// Only the submitted four-token span is backed up/restored, never the full cache.
inline constexpr const char* kVerificationKVSpanShader=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct VerifyKVSpanParams {
    ulong batch,chunk,heads,dim,capacity,cacheStride,slotStride,
          submittedStride,acceptedStride,backupStride;
};
kernel void verification_kv_backup(device const uint* cache [[buffer(0)]],
    device const float* slots [[buffer(1)]],device const float* submitted [[buffer(2)]],
    device uint* backup [[buffer(3)]],constant VerifyKVSpanParams& p [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.dim;
    if(ulong(id)>=p.batch*p.chunk*width)return;
    ulong flat=ulong(id)/width,col=ulong(id)%width,row=flat/p.chunk,token=flat%p.chunk;
    uint word=0;
    if(submitted[row*p.submittedStride+token]!=0.0f) {
        ulong slot=ulong(slots[row*p.slotStride+token]);
        word=cache[row*p.cacheStride+((col/p.dim)*p.capacity+slot)*p.dim+col%p.dim];
    }
    backup[flat*p.backupStride+col]=word;
}
kernel void verification_kv_restore(device const uint* backup [[buffer(0)]],
    device const float* slots [[buffer(1)]],device const float* submitted [[buffer(2)]],
    device const float* accepted [[buffer(3)]],device uint* cache [[buffer(4)]],
    constant VerifyKVSpanParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.dim;
    if(ulong(id)>=p.batch*p.chunk*width)return;
    ulong flat=ulong(id)/width,col=ulong(id)%width,row=flat/p.chunk,token=flat%p.chunk;
    if(submitted[row*p.submittedStride+token]!=0.0f&&accepted[row*p.acceptedStride+token]==0.0f) {
        ulong slot=ulong(slots[row*p.slotStride+token]);
        cache[row*p.cacheStride+((col/p.dim)*p.capacity+slot)*p.dim+col%p.dim]=backup[flat*p.backupStride+col];
    }
}
)METAL";
}
