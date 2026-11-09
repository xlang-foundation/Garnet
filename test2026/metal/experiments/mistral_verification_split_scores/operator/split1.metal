
constant uint verify_score_chunk [[function_constant(1)]];
kernel void verification_split_score_dots(device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
    device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_VERIFY_PARAMS;
    const ulong id=group.x;if(id>=p.batch*p.chunk*p.heads)return;
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk;
    ulong kv=head/(p.heads/p.kvHeads),base=flatRow*p.pStride+head*p.capacity;
    if(active[row*p.aStride+token]==0.0f)return;
    const ulong first=ulong(slots[row*p.sStride]);
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    const ulong begin=ulong(group.y)*verify_score_chunk,end=min(visible,begin+verify_score_chunk);
    for(ulong s=begin+simd;s<end;s+=4) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong j=lane;j<p.dim;j+=32) {
                float k=s<first?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+j]:key[(row*p.chunk+s-first)*p.kStride+kv*p.dim+j];
                score+=query[flatRow*p.qStride+head*p.dim+j]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0)output[base+s]=score;
    }
}
kernel void verification_split_score_softmax(device const float* query [[buffer(0)]],
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
    for(ulong s=tid;s<visible;s+=128)scratch[s]=output[base+s];
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
