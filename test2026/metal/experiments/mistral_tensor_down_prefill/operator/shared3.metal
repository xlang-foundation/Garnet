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
