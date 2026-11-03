// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kAffineInt4MPPShader=R"METAL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct MPPQ4Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_mpp_fp32(device const float* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant MPPQ4Params& p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
    // Dequantize one bounded tile, never a whole projection/checkpoint. Both
    // operands and accumulation stay FP32; relaxed precision is explicitly off.
    threadgroup float aTile[32*32],bTile[32*33];
    auto a=tensor(aTile,extents<int,32,32>{},array<int,2>{1,32});
    auto b=tensor(bTile,extents<int,32,32>{},array<int,2>{1,33});
    constexpr auto descriptor=matmul2d_descriptor(32,32,32,false,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor,execution_simdgroups<4>> multiply;
    auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<32*32;i+=128) {
            const ulong row=ulong(group.y)*32+i/32,k=base+i%32;
            aTile[i]=(row<p.rows&&k<p.inner)?input[row*p.aStride+k]:0.0f;
        }
        const uint localN=tid/4,localK=(tid%4)*8;
        const ulong n=ulong(group.x)*32+localN,k=base+localK;
        uint values=0;float scale=0,bias=0;
        if(n<p.columns&&k<p.inner) {
            const ulong coefficient=n*(p.inner/64)+k/64;
            values=packed[n*(p.inner/8)+k/8];
            scale=as_type<float>(uint(scales[coefficient])<<16);
            bias=as_type<float>(uint(biases[coefficient])<<16);
        }
        for(uint j=0;j<8;++j)bTile[(localK+j)*33+localN]=
            (n<p.columns&&k+j<p.inner)?fma(float((values>>(4*j))&15u),scale,bias):0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply.run(a,b,accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Dynamic extents keep the final partial row/column tile inside its output.
    auto destination=tensor(output,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
    accumulator.store(destination.slice(int(group.x)*32,int(group.y)*32));
}
)METAL";
}
