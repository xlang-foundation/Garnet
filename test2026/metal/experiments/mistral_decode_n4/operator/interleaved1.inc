// Research candidate: requires the same fixed affine profile prelude and rows
// function constant as the native blocked kernel. Unbuilt/unqualified.
struct InterleavedQ4Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_interleaved4(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant InterleavedQ4Params& runtime [[buffer(5)]],
#ifdef GARNET_BLOCKED_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(InterleavedQ4Params);
    const AffineIndex block=AffineIndex(group)*4+simd,first=block*4;
    if(first>=p.columns)return;
    AffineIndex physical[4];float4 sums[4];
#pragma clang loop unroll(full)
    for(uint row=0;row<4;++row) {
#ifdef GARNET_BLOCKED_INDEXED
        physical[row]=row<GARNET_BLOCKED_ROWS?indices[row]:0;
#else
        physical[row]=row;
#endif
        sums[row]=float4(0.0f);
    }
    for(AffineIndex word=lane;word<p.inner/8;word+=32) {
        const uint4 values=*reinterpret_cast<device const uint4*>(packed+(block*(p.inner/8)+word)*4);
        const AffineIndex coefficient=(block*(p.inner/64)+word/8)*4;
        const ushort4 sc=*reinterpret_cast<device const ushort4*>(scales+coefficient);
        const ushort4 bi=*reinterpret_cast<device const ushort4*>(biases+coefficient);
        const float4 scale=float4(as_type<half4>(sc)),bias=float4(as_type<half4>(bi));
        float4 low[4],high[4];
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS) {
            const AffineIndex base=physical[row]*p.aStride+word*8;
            low[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base));
            high[row]=float4(*reinterpret_cast<device const packed_float4*>(a+base+4));
        }
#pragma clang loop unroll(full)
        for(uint j=0;j<8;++j) {
            const float4 weight=fma(float4((values>>(4*j))&uint4(15u)),scale,bias);
#pragma clang loop unroll(full)
            for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS)
                sums[row]=fma(float4(j<4?low[row][j]:high[row][j-4]),weight,sums[row]);
        }
    }
#pragma clang loop unroll(full)
    for(uint column=0;column<4;++column)if(first+column<p.columns) {
#pragma clang loop unroll(full)
        for(uint row=0;row<4;++row)if(row<GARNET_BLOCKED_ROWS) {
            const float total=simd_sum(sums[row][column]);
            if(lane==0)out[physical[row]*p.oStride+first+column]=total;
        }
    }
}
