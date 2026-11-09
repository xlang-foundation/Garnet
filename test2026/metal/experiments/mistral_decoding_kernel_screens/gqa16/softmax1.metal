kernel void verification_gqa16_softmax(device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
    device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_VERIFY_PARAMS;
    const ulong id=group.x;if(id>=p.batch*p.chunk*p.heads)return;
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk;
    ulong kv=head/(p.heads/p.kvHeads),base=flatRow*p.pStride+head*p.capacity;
    if(active[row*p.aStride+token]==0.0f){for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
    for(ulong s=tid;s<visible;s+=128)scratch[s]=output[base+s]*rsqrt(float(p.dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
