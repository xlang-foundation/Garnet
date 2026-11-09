// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
namespace Garnet {
inline bool PackedAffineLayoutMatches(const uint32_t* words,size_t count) {
    if(!words||count!=32)return false;
    for(size_t i=0;i<count;++i)if(words[i]!=0)return false;
    return true;
}
// Raw cooperative-register indexing is qualified for this exact layout.
// Compile and execute the guard in the SAME library as the packed kernels:
// SDK/offline and OS/runtime compilers can expose different operand layouts.
inline constexpr const char* kPackedAffineLayoutGuardShader=R"METAL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;using namespace mpp::tensor_ops;
kernel void packed_affine_layout_guard(device uint* status [[buffer(0)]],
 uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> op;
 auto left=op.get_left_input_cooperative_tensor<half,half,float>();
 auto right=op.get_right_input_cooperative_tensor<half,half,float>();
 auto result=op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 uint flags=0;
 if(left.get_capacity()!=8)flags|=1;
 if(right.get_capacity()!=16)flags|=2;
 if(result.get_capacity()!=16)flags|=4;
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),column=((quad&2)|(lane&1))*4;
 for(ushort i=0;i<8;++i) {
  if(i>=left.get_capacity()){flags|=8;continue;}
  auto xy=left.get_multidimensional_index(i);
  if(!left.is_valid_element(i)||xy[0]!=column+i%4||xy[1]!=row+(i/4)*8)flags|=8;
 }
 for(ushort i=0;i<16;++i) {
  if(i>=right.get_capacity()){flags|=16;continue;}
  auto xy=right.get_multidimensional_index(i);
  if(!right.is_valid_element(i)||xy[0]!=column+i%4||xy[1]!=row+(i/4)*8)flags|=16;
 }
 for(ushort i=0;i<16;++i) {
  if(i>=result.get_capacity()){flags|=32;continue;}
  auto xy=result.get_multidimensional_index(i);
  if(!result.is_valid_element(i)||xy[0]!=column+(i/8)*16+i%4||xy[1]!=row+((i/4)%2)*8)flags|=32;
 }
 status[lane]=flags;
}
)METAL";
}
