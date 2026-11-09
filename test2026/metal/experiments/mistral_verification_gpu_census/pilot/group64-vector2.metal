// Experimental FP32 centered affine-group dot. Separate arithmetic policy.
// Including library supplies the original affine prelude and function constants.
constant uint group64_columns [[function_constant(21)]];
struct Group64Params {ulong rows,inner,columns,aStride,oStride;};
kernel void centered_input_sums(device const float* a [[buffer(0)]],
 constant Group64Params& runtime [[buffer(5)]],
#if GROUP64_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 device float* sums [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    GARNET_AFFINE_PARAMETERS(Group64Params);
    const uint groups=p.inner/64,logical=id/groups,g=id%groups;
    if(logical>=GROUP64_ROWS)return;
#if GROUP64_INDEXED
    const uint row=indices[logical];
#else
    const uint row=logical;
#endif
    float sum=0;
#pragma clang loop unroll(disable)
    for(uint j=0;j<64;++j)sum=sum+a[row*p.aStride+g*64+j];
    sums[row*groups+g]=sum;
}
kernel void centered_q4_group64(device const float* a [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* out [[buffer(4)]],
 constant Group64Params& runtime [[buffer(5)]],
#if GROUP64_INDEXED
 constant uint* indices [[buffer(6)]],
#endif
 device const float* input_sums [[buffer(7)]],
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
    GARNET_AFFINE_PARAMETERS(Group64Params);
    const uint first=(group*4+simd)*group64_columns,groups=p.inner/64;
    if(first>=p.columns)return;
    uint physical[4];float total[4][4];
#pragma clang loop unroll(full)
    for(uint r=0;r<4;++r){
#if GROUP64_INDEXED
        physical[r]=r<GROUP64_ROWS?indices[r]:0;
#else
        physical[r]=r;
#endif
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)total[r][c]=0;
    }
    for(uint g=lane;g<groups;g+=32){
        float scale[4],residual[4],dot[4][4];int zero[4];uint4 wordsLow[4],wordsHigh[4];
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
            const uint coefficient=(first+c)*groups+g;
            scale[c]=garnet_affine_coefficient(scales[coefficient]);
            const float bias=garnet_affine_coefficient(biases[coefficient]);
            const float z=scale[c]!=0?-bias/scale[c]:-1;
            zero[c]=(z>=0&&z<=15)?int(rint(z)):0;
            residual[c]=fma(float(zero[c]),scale[c],bias);
            const device uint* wp=packed+(first+c)*(p.inner/8)+g*8;
            wordsLow[c]=uint4(*reinterpret_cast<device const packed_uint4*>(wp));
            wordsHigh[c]=uint4(*reinterpret_cast<device const packed_uint4*>(wp+4));
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)dot[r][c]=0;
        }
#pragma clang loop unroll(full)
        for(uint word=0;word<8;++word){
            float4 low[4],high[4];
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)if(r<GROUP64_ROWS){
                const uint base=physical[r]*p.aStride+g*64+word*8;
                low[r]=float4(*reinterpret_cast<device const packed_float4*>(a+base));
                high[r]=float4(*reinterpret_cast<device const packed_float4*>(a+base+4));
            }
#pragma clang loop unroll(full)
            for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
                const uint values=word<4?wordsLow[c][word]:wordsHigh[c][word-4];
#pragma clang loop unroll(full)
                for(uint j=0;j<8;++j){
                    const float q=float(int((values>>(j*4))&15u)-zero[c]);
#pragma clang loop unroll(full)
                    for(uint r=0;r<4;++r)if(r<GROUP64_ROWS)
                        dot[r][c]=fma(j<4?low[r][j]:high[r][j-4],q,dot[r][c]);
                }
            }
        }
#pragma clang loop unroll(full)
        for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
#pragma clang loop unroll(full)
            for(uint r=0;r<4;++r)if(r<GROUP64_ROWS)
                total[r][c]=fma(dot[r][c],scale[c],fma(input_sums[physical[r]*groups+g],residual[c],total[r][c]));
        }
    }
#pragma clang loop unroll(full)
    for(uint c=0;c<4;++c)if(c<group64_columns&&first+c<p.columns){
#pragma clang loop unroll(full)
        for(uint r=0;r<4;++r)if(r<GROUP64_ROWS){
            const float result=simd_sum(total[r][c]);
            if(lane==0)out[physical[r]*p.oStride+first+c]=result;
        }
    }
}
