// Experimental direct device-tensor operands; no manual cooperative-fragment ABI.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template<int TM,int TN>
inline void tensor_projection(device const half* input,device const half* weights,
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
 auto accumulator=multiply.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
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
kernel void NAME(device const half* input [[buffer(0)]],device const half* weights [[buffer(1)]], \
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
