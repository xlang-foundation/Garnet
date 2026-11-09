#define GARNET_AFFINE_PROFILE 1
#define GARNET_AFFINE_INDEX32 1
#define GARNET_AFFINE_FP16_COEFFICIENTS 1

#include <metal_stdlib>
inline float garnet_affine_coefficient(ushort value) {
#ifdef GARNET_AFFINE_FP16_COEFFICIENTS
    return float(as_type<half>(value));
#else
    return as_type<float>(uint(value)<<16);
#endif
}
#if defined(GARNET_AFFINE_INDEX32) && !defined(GARNET_AFFINE_PROFILE)
#error Bounded_indexes_require_specialized_profiles
#endif
#ifdef GARNET_AFFINE_INDEX32
using AffineIndex=uint;
static_assert(sizeof(AffineIndex)==4,"Compiled affine index width differs");
struct BoundedAffineParams {uint rows,inner,columns,aStride,oStride;};
#else
using AffineIndex=ulong;
#endif
#ifdef GARNET_AFFINE_PROFILE
constant uint affine_profile_inner [[function_constant(10)]];
constant uint affine_profile_columns [[function_constant(11)]];
constant uint affine_profile_a_stride [[function_constant(12)]];
constant uint affine_profile_o_stride [[function_constant(13)]];
#ifdef GARNET_AFFINE_INDEX32
#define GARNET_AFFINE_PARAMETERS(TYPE) const BoundedAffineParams p{uint(runtime.rows),affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) const TYPE p{runtime.rows,affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#endif
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) constant TYPE& p=runtime
#endif

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct VectorQ4Params {ulong rows,inner,columns,aStride,oStride;};
kernel void packed_affine_gather(device const float* input [[buffer(0)]],
 device half* compact [[buffer(1)]],constant uint* indices [[buffer(2)]],
 constant uint4& shape [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<shape.x*shape.y)compact[i]=half(input[indices[i/shape.y]*shape.z+i%shape.y]);
}
kernel void half_m128k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+(simd/2)*32,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void half_m256k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

kernel void pair_m128k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 device const uint* packedUp [[buffer(8)]],device const ushort* scalesUp [[buffer(9)]],device const ushort* biasesUp [[buffer(10)]],device float* outputUp [[buffer(11)]],
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate0[ci],gate0[ci+1],gate0[ci+2],gate0[ci+3]);
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
    *reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up0[ci],up0[ci+1],up0[ci+2],up0[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void pair_m256k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 device const uint* packedUp [[buffer(8)]],device const ushort* scalesUp [[buffer(9)]],device const ushort* biasesUp [[buffer(10)]],device float* outputUp [[buffer(11)]],
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate0[ci],gate0[ci+1],gate0[ci+2],gate0[ci+3]);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(gate1[ci],gate1[ci+1],gate1[ci+2],gate1[ci+3]);
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
    *reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up0[ci],up0[ci+1],up0[ci+2],up0[ci+3]);
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
    *reinterpret_cast<device float4*>(outputUp+indices[row]*p.oStride+col)=float4(up1[ci],up1[ci+1],up1[ci+2],up1[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

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


constant bool fragment_writers [[function_constant(22)]];

constant bool prepared_writers [[function_constant(23)]];
kernel void prepare_half_weights(device const uint* packed [[buffer(0)]],device const ushort* scales [[buffer(1)]],
 device const ushort* biases [[buffer(2)]],device half* output [[buffer(3)]],constant uint4& dims [[buffer(4)]],
 device atomic_uint* writers [[buffer(5)]],uint gid [[thread_position_in_grid]]) {
 uint count=dims.x*dims.y;if(dims.z)count=count/128*136;
 const uint first=gid*4;if(first>=count)return;
 for(uint j=0;j<4;++j){uint at=first+j,n,k;bool padding=false;
  if(dims.z){uint block=at/(64*136),local=at%(64*136);n=(block/(dims.x/128))*64+local/136;uint kk=local%136; k=(block%(dims.x/128))*128+kk;padding=kk>=128;}
  else{n=at/dims.x;k=at%dims.x;}
  half value=half(0);
  if(!padding){uint word=packed[n*(dims.x/8)+k/8],code=(word>>((k%8)*4))&15u,coefficient=n*(dims.x/64)+k/64;
   float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);value=half(fma(float(code),scale,bias));}
  output[at]=value;if(prepared_writers)atomic_fetch_add_explicit(writers+at,1u,memory_order_relaxed);
 }
}

kernel void cached_mode0_m128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+(simd/2)*32,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode0_m256(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode1_m128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+(simd/2)*32,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=n*p.inner+base+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode1_m256(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=n*p.inner+base+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode2_m128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+(simd/2)*32,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=((group.x*(p.inner/128)+base/128)*64+localN)*136+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode2_m256(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=((group.x*(p.inner/128)+base/128)*64+localN)*136+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode3_m128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto rightNext=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto leftNext=multiply.get_left_input_cooperative_tensor<half,half,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*128+(simd/2)*32,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=((group.x*(p.inner/128)+base/128)*64+localN)*136+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=32) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){
    const uint at=((simd%2)*32+part*16+fragmentRow)*weightStride+local+fragmentColumn;
    const half4 lo=*reinterpret_cast<threadgroup const half4*>(weights+at);
    const half4 hi=*reinterpret_cast<threadgroup const half4*>(weights+at+8*weightStride);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){right[part*8+j]=lo[j];right[part*8+4+j]=hi[j];}
   }
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){
    const uint at=((simd%2)*32+part*16+fragmentRow)*weightStride+(local+16)+fragmentColumn;
    const half4 lo=*reinterpret_cast<threadgroup const half4*>(weights+at);
    const half4 hi=*reinterpret_cast<threadgroup const half4*>(weights+at+8*weightStride);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){rightNext[part*8+j]=lo[j];rightNext[part*8+4+j]=hi[j];}
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   multiply.run(leftNext,rightNext,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   multiply.run(leftNext,rightNext,c1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}
kernel void cached_mode3_m256(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],device const half* prepared [[buffer(8)]],device atomic_uint* writers [[buffer(7)]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto rightNext=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto leftNext=multiply.get_left_input_cooperative_tensor<half,half,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint cacheAt=((group.x*(p.inner/128)+base/128)*64+localN)*136+localK;
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i){
   const half4 value=*reinterpret_cast<device const half4*>(prepared+cacheAt+4*i);
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=32) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){
    const uint at=((simd%2)*32+part*16+fragmentRow)*weightStride+local+fragmentColumn;
    const half4 lo=*reinterpret_cast<threadgroup const half4*>(weights+at);
    const half4 hi=*reinterpret_cast<threadgroup const half4*>(weights+at+8*weightStride);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){right[part*8+j]=lo[j];right[part*8+4+j]=hi[j];}
   }
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){
    const uint at=((simd%2)*32+part*16+fragmentRow)*weightStride+(local+16)+fragmentColumn;
    const half4 lo=*reinterpret_cast<threadgroup const half4*>(weights+at);
    const half4 hi=*reinterpret_cast<threadgroup const half4*>(weights+at+8*weightStride);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){rightNext[part*8+j]=lo[j];rightNext[part*8+4+j]=hi[j];}
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   multiply.run(leftNext,rightNext,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   multiply.run(leftNext,rightNext,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   multiply.run(leftNext,rightNext,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart){
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+(local+16)+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)leftNext[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
   multiply.run(leftNext,rightNext,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    if(fragment_writers)for(uint w=0;w<4;++w)atomic_fetch_add_explicit(writers+indices[row]*p.columns+col+w,1u,memory_order_relaxed);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

// Experimental direct device-tensor operands; no manual cooperative-fragment ABI.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template<int TM,int TN>
inline void tensor_projection(device half* input,device half* weights,
 device float* output,uint rows,uint inner,uint columns,uint2 block,uint simd) {
 // Two by two SIMD groups. Each group owns one independent output tile.
 const int row=int(block.y)*(2*TM)+int(simd/2)*TM;
 const int column=int(block.x)*(2*TN)+int(simd%2)*TN;
 auto a=tensor(input,dextents<int,2>{int(inner),int(rows)},array<int,2>{1,int(inner)});
 auto b=tensor(weights,dextents<int,2>{int(inner),int(columns)},array<int,2>{1,int(inner)});
 auto d=tensor(output,dextents<int,2>{int(columns),int(rows)},array<int,2>{1,int(columns)});
 constexpr auto spec=matmul2d_descriptor(TM,TN,128,false,true,false,
  matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<spec,execution_simdgroup> multiply;
 auto accumulator=multiply.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
 for(int k=0;k<int(inner);k+=128) {
  threadgroup_barrier(mem_flags::mem_none);
  auto left=a.template slice<128,TM>(k,row);
  auto right=b.template slice<128,TN>(k,column);
  multiply.run(left,right,accumulator);
 }
 auto destination=d.template slice<TN,TM>(column,row);
 accumulator.store(destination);
}

#define TENSOR_ENTRY(NAME,M,N) \
kernel void NAME(device half* input [[buffer(0)]],device half* weights [[buffer(1)]], \
 device float* output [[buffer(2)]],constant VectorQ4Params& runtime [[buffer(3)]], \
 uint2 block [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]]) { \
 GARNET_AFFINE_PARAMETERS(VectorQ4Params); \
 tensor_projection<M,N>(input,weights,output,p.rows,p.inner,p.columns,block,simd); \
}
TENSOR_ENTRY(tensor_m32n32,32,32)
TENSOR_ENTRY(tensor_m64n32,64,32)
TENSOR_ENTRY(tensor_m32n64,32,64)

kernel void tensor_scatter(device const float* compact [[buffer(0)]],
 device float* output [[buffer(1)]],constant uint* indices [[buffer(2)]],
 constant uint2& shape [[buffer(3)]],uint index [[thread_position_in_grid]]) {
 if(index<shape.x*shape.y)output[indices[index/shape.y]*shape.y+index%shape.y]=compact[index];
}

// Direct tensor operands over the original bounded Q4 shared-memory tile.
kernel void shared_tensor_m256(device half* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint2 block [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 // Dense identity row map only. The host retains the original route for
 // sparse/permuted/partial profiles; no request is padded to enter this path.
 constexpr uint BK=128,STRIDE=136,PACKET=32;
 threadgroup half decoded[64*STRIDE];
 const uint row=block.y*256+simd*32,column=block.x*64;
 const uint localN=tid/4,localK=(tid%4)*PACKET,n=column+localN;
 auto a=tensor(input,dextents<int,2>{int(p.inner),int(p.rows)},array<int,2>{1,int(p.inner)});
 auto b=tensor(decoded,extents<int,BK,64>{},array<int,2>{1,int(STRIDE)});
 auto d=tensor(output,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
 constexpr auto spec=matmul2d_descriptor(32,64,128,false,true,false,
  matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<spec,execution_simdgroup> multiply;
 auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
 for(uint base=0;base<p.inner;base+=BK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8;
  const uint coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<PACKET/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(decoded+localN*STRIDE+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto left=a.slice<128,32>(int(base),int(row));
  multiply.run(left,b,accumulator);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 auto destination=d.slice<64,32>(int(column),int(row));
 accumulator.store(destination);
}

// Experimental cache-local dispatch permutations; arithmetic bodies unchanged.

kernel void half_rowfast_m256k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 dispatchGroup [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 const uint rowTiles=(p.rows+255)/256,linear=dispatchGroup.y*(p.columns/64)+dispatchGroup.x;
 const uint2 group=uint2(linear/rowTiles,linear%rowTiles);

 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

kernel void half_pair_m256k128(device const half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],uint2 dispatchGroup [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 const uint rowTiles=(p.rows+255)/256,linear=dispatchGroup.y*(p.columns/64)+dispatchGroup.x;
 const uint pairIndex=linear/(2*rowTiles),within=linear%(2*rowTiles);
 const uint2 group=uint2(pairIndex*2+(within%2),within/2);

 constexpr uint stageK=128,weightStride=stageK+8,packet=64*stageK/256;
 threadgroup half weights[64*weightStride];
 constexpr auto descriptor=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<descriptor,execution_simdgroup> multiply;
 auto left=multiply.get_left_input_cooperative_tensor<half,half,float>();
 auto right=multiply.get_right_input_cooperative_tensor<half,half,float>();
 auto c0=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c1=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c2=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 auto c3=multiply.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,metal::remove_addrspace_t<decltype(right)>,float>();
 for(ushort i=0;i<c0.get_capacity();++i){c0[i]=0;c1[i]=0;c2[i]=0;c3[i]=0;}
 const uint quad=lane>>2,fragmentRow=(quad&4)|((lane>>1)&3),fragmentColumn=((quad&2)|(lane&1))*4;
 const uint firstRow=group.y*256+(simd/2)*64,firstColumn=group.x*64+(simd%2)*32;
 const uint localN=tid/(stageK/packet),localK=(tid%(stageK/packet))*packet,n=group.x*64+localN;
 for(uint base=0;base<p.inner;base+=stageK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<packet/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(weights+localN*weightStride+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint local=0;local<stageK;local+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part) {
    #pragma clang loop unroll(full)
    for(uint i=0;i<8;++i)right[part*8+i]=weights[((simd%2)*32+part*16+fragmentRow+(i/4)*8)*weightStride+local+fragmentColumn+i%4];
   }
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+0+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c0);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+16+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c1);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+32+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c2);
   #pragma clang loop unroll(full)
   for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
    const uint row=firstRow+48+fragmentRow+fragmentPart*8,k=base+local+fragmentColumn;
    const half4 value=row<p.rows?*reinterpret_cast<device const half4*>(input+row*p.inner+k):half4(0);
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)left[fragmentPart*4+j]=value[j];
   }
   multiply.run(left,right,c3);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+0+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c0[ci],c0[ci+1],c0[ci+2],c0[ci+3]);
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
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c1[ci],c1[ci+1],c1[ci+2],c1[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+32+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c2[ci],c2[ci+1],c2[ci+2],c2[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
 #pragma clang loop unroll(full)
 for(uint part=0;part<2;++part) {
  #pragma clang loop unroll(full)
  for(uint fragmentPart=0;fragmentPart<2;++fragmentPart) {
   const uint row=firstRow+48+fragmentRow+fragmentPart*8,col=firstColumn+part*16+fragmentColumn,ci=part*8+fragmentPart*4;
   if(row<p.rows) {
    *reinterpret_cast<device float4*>(output+indices[row]*p.oStride+col)=float4(c3[ci],c3[ci+1],c3[ci+2],c3[ci+3]);
    for(uint j=0;j<4;++j){}
   }
  }
 }
}

kernel void shared_tensor_rowfast(device half* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint2 dispatchGroup [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 const uint rowTiles=(p.rows+255)/256,linear=dispatchGroup.y*(p.columns/64)+dispatchGroup.x;
 const uint2 block=uint2(linear/rowTiles,linear%rowTiles);

 // Dense identity row map only. The host retains the original route for
 // sparse/permuted/partial profiles; no request is padded to enter this path.
 constexpr uint BK=128,STRIDE=136,PACKET=32;
 threadgroup half decoded[64*STRIDE];
 const uint row=block.y*256+simd*32,column=block.x*64;
 const uint localN=tid/4,localK=(tid%4)*PACKET,n=column+localN;
 auto a=tensor(input,dextents<int,2>{int(p.inner),int(p.rows)},array<int,2>{1,int(p.inner)});
 auto b=tensor(decoded,extents<int,BK,64>{},array<int,2>{1,int(STRIDE)});
 auto d=tensor(output,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
 constexpr auto spec=matmul2d_descriptor(32,64,128,false,true,false,
  matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<spec,execution_simdgroup> multiply;
 auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
 for(uint base=0;base<p.inner;base+=BK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8;
  const uint coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<PACKET/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(decoded+localN*STRIDE+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto left=a.slice<128,32>(int(base),int(row));
  multiply.run(left,b,accumulator);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 auto destination=d.slice<64,32>(int(column),int(row));
 accumulator.store(destination);
}

kernel void shared_tensor_pair(device half* input [[buffer(0)]],
 device const uint* packed [[buffer(1)]],device const ushort* scales [[buffer(2)]],
 device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 uint2 dispatchGroup [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 const uint rowTiles=(p.rows+255)/256,linear=dispatchGroup.y*(p.columns/64)+dispatchGroup.x;
 const uint pairIndex=linear/(2*rowTiles),within=linear%(2*rowTiles);
 const uint2 block=uint2(pairIndex*2+(within%2),within/2);

 // Dense identity row map only. The host retains the original route for
 // sparse/permuted/partial profiles; no request is padded to enter this path.
 constexpr uint BK=128,STRIDE=136,PACKET=32;
 threadgroup half decoded[64*STRIDE];
 const uint row=block.y*256+simd*32,column=block.x*64;
 const uint localN=tid/4,localK=(tid%4)*PACKET,n=column+localN;
 auto a=tensor(input,dextents<int,2>{int(p.inner),int(p.rows)},array<int,2>{1,int(p.inner)});
 auto b=tensor(decoded,extents<int,BK,64>{},array<int,2>{1,int(STRIDE)});
 auto d=tensor(output,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
 constexpr auto spec=matmul2d_descriptor(32,64,128,false,true,false,
  matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<spec,execution_simdgroup> multiply;
 auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<accumulator.get_capacity();++i)accumulator[i]=0.0f;
 for(uint base=0;base<p.inner;base+=BK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8;
  const uint coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scales[coefficient]),bias=garnet_affine_coefficient(biases[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(packed+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<PACKET/4;++i) {
   const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;
   *reinterpret_cast<threadgroup half4*>(decoded+localN*STRIDE+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto left=a.slice<128,32>(int(base),int(row));
  multiply.run(left,b,accumulator);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 auto destination=d.slice<64,32>(int(column),int(row));
 accumulator.store(destination);
}
