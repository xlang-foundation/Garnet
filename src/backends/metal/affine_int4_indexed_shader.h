// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kAffineInt4IndexedShader=R"METAL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct IndexedQ4Params {ulong rows,inner,columns,aStride,oStride;};
// Logical token rows are compact; storage remains in the request graph layout.
// Strict BF16 operands match the register-prefill precision contract even for
// one or two selected tokens. The caller zeros inactive output rows first.
kernel void affine_q4_indexed(device const float* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant IndexedQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint2 group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]]) {
    GARNET_AFFINE_PARAMETERS(IndexedQ4Params);
    constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor,execution_simdgroup> multiply;
    auto left=multiply.get_left_input_cooperative_tensor<bfloat,bfloat,float>();
    auto right=multiply.get_right_input_cooperative_tensor<bfloat,bfloat,float>();
    auto accumulator=multiply.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
    for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
    const uint quad=lane>>2;
    const uint fragmentRow=(quad&4)|((lane>>1)&3);
    const uint fragmentColumn=((quad&2)|(lane&1))*4;
    const AffineIndex firstRow=AffineIndex(group.y)*64+AffineIndex(simd)*16;
    const AffineIndex firstColumn=AffineIndex(group.x)*32;
    if(firstRow>=p.rows)return;
    for(AffineIndex block=0;block<p.inner;block+=GARNET_INDEXED_K)
        for(uint local=0;local<GARNET_INDEXED_K;local+=16) {
        #pragma clang loop unroll(full)
        for(uint i=0;i<8;++i) {
            const AffineIndex k=block+local+fragmentColumn+i%4;
            const AffineIndex logical=firstRow+fragmentRow+(i/4)*8;
            left[i]=bfloat((logical<p.rows&&k<p.inner)?input[AffineIndex(indices[logical])*p.aStride+k]:0.0f);
        }
        #pragma clang loop unroll(full)
        for(uint tilePart=0;tilePart<2;++tilePart) {
            #pragma clang loop unroll(full)
            for(uint rowPart=0;rowPart<2;++rowPart) {
                const AffineIndex n=firstColumn+fragmentRow+rowPart*8+tilePart*16;
                const AffineIndex k=block+local+fragmentColumn;
                uint word=0;float scale=0.0f,bias=0.0f;
                if(n<p.columns&&k<p.inner) {
                    word=packed[n*(p.inner/8)+k/8];
                    const AffineIndex coefficient=n*(p.inner/64)+k/64;
                    scale=as_type<float>(uint(scales[coefficient])<<16);
                    bias=as_type<float>(uint(biases[coefficient])<<16);
                }
                #pragma clang loop unroll(full)
                for(uint j=0;j<4;++j)right[tilePart*8+rowPart*4+j]=
                    bfloat(fma(float((word>>(4*((k%8)+j)))&15u),scale,bias));
            }
        }
        multiply.run(left,right,accumulator);
    }
    for(uint tilePart=0;tilePart<2;++tilePart)for(uint i=0;i<8;++i) {
        const AffineIndex logical=firstRow+fragmentRow+(i/4)*8;
        const AffineIndex n=firstColumn+tilePart*16+fragmentColumn+i%4;
        if(logical<p.rows&&n<p.columns)
            output[AffineIndex(indices[logical])*p.oStride+n]=accumulator[tilePart*8+i];
    }
}
)METAL";
}
