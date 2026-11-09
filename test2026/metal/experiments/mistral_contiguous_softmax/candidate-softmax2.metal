kernel void chunk_mps_softmax_contiguous(device float* cacheScores [[buffer(0)]],device const float* liveScores [[buffer(1)]],
    device float* liveProb [[buffer(2)]],device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
    constant ChunkAttentionParams& runtime [[buffer(5)]],constant ChunkMPSPitch& pitch [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint query [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const auto p=chunk_mps_params(runtime);
    uint t=query%p.chunk,b=query/(p.chunk*p.heads);
#ifdef GARNET_CHUNK_ATTENTION_BUCKETS
    const uint extent=chunk_mps_extent;
#else
    const uint extent=p.capacity;
#endif
    if(active[b*p.aStride+t]==0) {
        for(uint s=tid;s<extent;s+=128)cacheScores[query*pitch.cacheScore+s]=0;
        for(uint j=tid;j<p.chunk;j+=128)liveProb[query*pitch.liveScore+j]=0;
        return;
    }
    uint visible=uint(min(ulong(extent),ulong(slots[b*p.sStride+t])+1));
    // Admitted only after completed native controls prove contiguous active prefixes.
    const int first=int(slots[b*p.sStride]);
    threadgroup float partial[4];
    for(uint s=tid;s<visible;s+=128) {
        int latest=-1;
        int candidate=int(s)-first;if(candidate>=0&&candidate<=int(t))latest=candidate;
        scratch[s]=(latest<0?cacheScores[query*pitch.cacheScore+s]:liveScores[query*pitch.liveScore+uint(latest)])*rsqrt(float(p.dim));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(uint s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0;
    for(uint s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    // All scratch reads below follow the last writes before the reduction's
    // threadgroup barrier. Each output probability has one owning thread.
    for(uint s=tid;s<extent;s+=128)
        cacheScores[query*pitch.cacheScore+s]=s<visible&&(int(s)<first||int(s)>first+int(t))?scratch[s]/denominator:0;
    for(uint j=tid;j<p.chunk;j+=128) {
        float position=slots[b*p.sStride+j];
        bool valid=j<=t&&active[b*p.aStride+j]!=0&&isfinite(position)&&position>=0&&position<float(visible)&&position==floor(position);
        liveProb[query*pitch.liveScore+j]=valid?scratch[uint(position)]/denominator:0;
    }
}
