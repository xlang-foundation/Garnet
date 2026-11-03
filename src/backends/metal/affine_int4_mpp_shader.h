// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kAffineInt4MPPShader=R"METAL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#ifdef GARNET_MPP_BF16
using Operand=bfloat;
#else
using Operand=float;
#endif
#ifndef GARNET_MPP_K
#define GARNET_MPP_K 32
#endif
constant constexpr uint BK=GARNET_MPP_K;
struct MPPQ4Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_mpp(device const float* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant MPPQ4Params& p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
    // Dequantize one bounded tile, never a whole projection/checkpoint.
    // Operand type is an explicit build profile, accumulation is FP32 and
    // relaxed precision is always off.
    threadgroup Operand aTile[32*BK],bTile[BK*33];
    auto a=tensor(aTile,extents<int,BK,32>{},array<int,2>{1,int(BK)});
    auto b=tensor(bTile,extents<int,32,BK>{},array<int,2>{1,33});
    constexpr auto descriptor=matmul2d_descriptor(32,32,BK,false,false,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor,execution_simdgroups<4>> multiply;
    auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
    for(ulong base=0;base<p.inner;base+=BK) {
        for(uint i=tid;i<32*BK;i+=128) {
            const ulong row=ulong(group.y)*32+i/BK,k=base+i%BK;
            aTile[i]=Operand((row<p.rows&&k<p.inner)?input[row*p.aStride+k]:0.0f);
        }
        // Keep the original coalesced 32-K word mapping for each wave.
        for(uint wave=0;wave<BK;wave+=32) {
            const uint localN=tid/4,localK=wave+(tid%4)*8;
            const ulong n=ulong(group.x)*32+localN,k=base+localK;
            uint values=0;float scale=0,bias=0;
            if(n<p.columns&&k<p.inner) {
                const ulong coefficient=n*(p.inner/64)+k/64;
                values=packed[n*(p.inner/8)+k/8];
                scale=as_type<float>(uint(scales[coefficient])<<16);
                bias=as_type<float>(uint(biases[coefficient])<<16);
            }
            for(uint j=0;j<8;++j)bTile[(localK+j)*33+localN]=
                Operand((n<p.columns&&k+j<p.inner)?fma(float((values>>(4*j))&15u),scale,bias):0.0f);
        }
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
