// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// BF16 operands, FP32 accumulator/output storage; tensor precision is explicit.
// Cooperative weight staging shares
// each unpacked tile across64 query rows; indexed loads preserve physical state.
// Per-SIMD register multiplication avoids cross-SIMD tensor redistribution.
inline constexpr const char* kAffineInt4Reuse64Shader=R"METAL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Reuse64Params {ulong rows,inner,columns,aStride,oStride;};
kernel void affine_q4_reuse64(device const float* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant Reuse64Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    GARNET_AFFINE_PARAMETERS(Reuse64Params);
    // Pad each64-element weight row by16 bytes to separate shared-memory banks.
    threadgroup bfloat weights[64*72];
    constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,GARNET_REUSE64_RELAXED,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor,execution_simdgroup> multiply;
    auto left=multiply.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto right=multiply.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto c0=multiply.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
    auto c1=multiply.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
    for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0.0f;c1[i]=0.0f;}
    const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3);
    const uint fragmentColumn=((quad&2)|(lane&1))*4;
    const ulong firstRow=ulong(group.y)*64+(simd/2)*32;
    const ulong firstColumn=ulong(group.x)*64+(simd%2)*32;
    for(ulong base=0;base<p.inner;base+=64) {
        for(uint word=tid;word<64*8;word+=128) {
            const uint localN=word/8,localK=(word%8)*8;
            const ulong n=ulong(group.x)*64+localN,k=base+localK;
            uint values=0;float scale=0.0f,bias=0.0f;
            if(n<p.columns&&k<p.inner) {
                values=packed[n*(p.inner/8)+k/8];
                const ulong coefficient=n*(p.inner/64)+k/64;
                scale=as_type<float>(uint(scales[coefficient])<<16);
                bias=as_type<float>(uint(biases[coefficient])<<16);
            }
            for(uint j=0;j<8;++j)weights[localN*72+localK+j]=
                bfloat((n<p.columns&&k+j<p.inner)?fma(float((values>>(4*j))&15u),scale,bias):0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint local=0;local<64;local+=16) {
            #pragma clang loop unroll(full)
            for(uint tilePart=0;tilePart<2;++tilePart) {
                #pragma clang loop unroll(full)
                for(uint i=0;i<8;++i) {
                    const uint n=(simd%2)*32+tilePart*16+fragmentRow+(i/4)*8;
                    right[tilePart*8+i]=weights[n*72+local+fragmentColumn+i%4];
                }
            }
            #pragma clang loop unroll(full)
            for(uint i=0;i<8;++i) {
                const ulong logical=firstRow+fragmentRow+(i/4)*8,k=base+local+fragmentColumn+i%4;
                left[i]=bfloat((logical<p.rows&&k<p.inner)?input[ulong(indices[logical])*p.aStride+k]:0.0f);
            }
            multiply.run(left,right,c0);
            #pragma clang loop unroll(full)
            for(uint i=0;i<8;++i) {
                const ulong logical=firstRow+16+fragmentRow+(i/4)*8,k=base+local+fragmentColumn+i%4;
                left[i]=bfloat((logical<p.rows&&k<p.inner)?input[ulong(indices[logical])*p.aStride+k]:0.0f);
            }
            multiply.run(left,right,c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint tilePart=0;tilePart<2;++tilePart)for(uint i=0;i<8;++i) {
        const ulong logical=firstRow+fragmentRow+(i/4)*8,n=firstColumn+tilePart*16+fragmentColumn+i%4;
        if(logical<p.rows&&n<p.columns)output[ulong(indices[logical])*p.oStride+n]=c0[tilePart*8+i];
        if(logical+16<p.rows&&n<p.columns)output[ulong(indices[logical+16])*p.oStride+n]=c1[tilePart*8+i];
    }
}
)METAL";
}
