// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {inline constexpr const char* kFusedMistralMlpShader=R"METAL(
constant bool fused_shadow [[function_constant(20)]];
constant bool fused_writers [[function_constant(21)]];

kernel void fused_m128k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 device const uint* packedUp [[buffer(8)]],device const ushort* scalesUp [[buffer(9)]],device const ushort* biasesUp [[buffer(10)]],device float* outputUp [[buffer(11)]],device half* outputHalf [[buffer(13)]],device atomic_uint* writers [[buffer(14)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=136,packet=32;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto left0=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto gate0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left0)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto up0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left0)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<gate0.get_capacity();++i){gate0[i]=0;up0[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+simd*16,firstColumn=group.x*32;
 const uint stream=tid/128,localTid=tid%128,localN=localTid/4,localK=(localTid%4)*packet,n=group.x*32+localN;
 device const uint* codesInput=stream?packedUp:packed;
 device const ushort* scaleInput=stream?scalesUp:scales;
 device const ushort* biasInput=stream?biasesUp:biases;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scaleInput[coefficient]),bias=garnet_affine_coefficient(biasInput[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(codesInput+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+(stream*32+localN)*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left0[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[(0+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   multiply.run(left0,right,gate0);
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[(32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   multiply.run(left0,right,up0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate0[ci],gate0[ci+1],gate0[ci+2],gate0[ci+3]);}
    for(uint j=0;j<4;++j){float g=gate0[ci+j];float value=(g/(1.0f+exp(-g)))*up0[ci+j];outputHalf[row*p.columns+col+j]=half(value);if(fused_writers)atomic_fetch_add_explicit(writers+row*p.columns+col+j,1u,memory_order_relaxed);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up0[ci],up0[ci+1],up0[ci+2],up0[ci+3]);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void fused_m256k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 device const uint* packedUp [[buffer(8)]],device const ushort* scalesUp [[buffer(9)]],device const ushort* biasesUp [[buffer(10)]],device float* outputUp [[buffer(11)]],device half* outputHalf [[buffer(13)]],device atomic_uint* writers [[buffer(14)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=136,packet=32;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto left0=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto gate0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left0)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto up0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left0)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<gate0.get_capacity();++i){gate0[i]=0;up0[i]=0;}
 auto left1=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto gate1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left1)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto up1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left1)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<gate1.get_capacity();++i){gate1[i]=0;up1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+simd*32,firstColumn=group.x*32;
 const uint stream=tid/128,localTid=tid%128,localN=localTid/4,localK=(localTid%4)*packet,n=group.x*32+localN;
 device const uint* codesInput=stream?packedUp:packed;
 device const ushort* scaleInput=stream?scalesUp:scales;
 device const ushort* biasInput=stream?biasesUp:biases;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scaleInput[coefficient]),bias=garnet_affine_coefficient(biasInput[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(codesInput+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+(stream*32+localN)*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left0[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left1[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[(0+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   multiply.run(left0,right,gate0);
   multiply.run(left1,right,gate1);
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[(32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   multiply.run(left0,right,up0);
   multiply.run(left1,right,up1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate0[ci],gate0[ci+1],gate0[ci+2],gate0[ci+3]);}
    for(uint j=0;j<4;++j){float g=gate0[ci+j];float value=(g/(1.0f+exp(-g)))*up0[ci+j];outputHalf[row*p.columns+col+j]=half(value);if(fused_writers)atomic_fetch_add_explicit(writers+row*p.columns+col+j,1u,memory_order_relaxed);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+16+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate1[ci],gate1[ci+1],gate1[ci+2],gate1[ci+3]);}
    for(uint j=0;j<4;++j){float g=gate1[ci+j];float value=(g/(1.0f+exp(-g)))*up1[ci+j];outputHalf[row*p.columns+col+j]=half(value);if(fused_writers)atomic_fetch_add_explicit(writers+row*p.columns+col+j,1u,memory_order_relaxed);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up0[ci],up0[ci+1],up0[ci+2],up0[ci+3]);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+16+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    if(fused_shadow){*reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up1[ci],up1[ci+1],up1[ci+2],up1[ci+3]);}
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

)METAL";}
