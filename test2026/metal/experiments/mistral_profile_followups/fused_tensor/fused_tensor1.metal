// Dense/full512 profile: same two Q4 streams and shared footprint, direct MPP operands.
kernel void fused_tensor_m256(device half* input [[buffer(0)]],device const uint* packed [[buffer(1)]],
 device const ushort* scales [[buffer(2)]],device const ushort* biases [[buffer(3)]],device float* output [[buffer(4)]],
 constant VectorQ4Params& runtime [[buffer(5)]],constant uint* indices [[buffer(6)]],
 device const uint* packedUp [[buffer(8)]],device const ushort* scalesUp [[buffer(9)]],device const ushort* biasesUp [[buffer(10)]],device float* outputUp [[buffer(11)]],device half* outputHalf [[buffer(13)]],device atomic_uint* writers [[buffer(14)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_AFFINE_PARAMETERS(VectorQ4Params);
 constexpr uint BK=128,STRIDE=136,PACKET=32;
 threadgroup half weights[64*STRIDE];
 const uint row=group.y*256+simd*32,column=group.x*32;
 const uint stream=tid/128,localTid=tid%128,localN=localTid/4,localK=(localTid%4)*PACKET,n=column+localN;
 device const uint* codesInput=stream?packedUp:packed;
 device const ushort* scaleInput=stream?scalesUp:scales;
 device const ushort* biasInput=stream?biasesUp:biases;
 auto a=tensor(input,dextents<int,2>{int(p.inner),int(p.rows)},array<int,2>{1,int(p.inner)});
 auto b=tensor(weights,extents<int,BK,64>{},array<int,2>{1,int(STRIDE)});
 auto gateWeights=b.slice<128,32>(0,0),upWeights=b.slice<128,32>(0,32);
 constexpr auto spec=matmul2d_descriptor(32,32,128,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<spec,execution_simdgroup> multiply;
 auto gate=multiply.get_destination_cooperative_tensor<decltype(a),decltype(gateWeights),float>();
 auto up=multiply.get_destination_cooperative_tensor<decltype(a),decltype(upWeights),float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<gate.get_capacity();++i){gate[i]=0.0f;up[i]=0.0f;}
 for(uint base=0;base<p.inner;base+=BK) {
  const uint offset=n*(p.inner/8)+(base+localK)/8,coefficient=n*(p.inner/64)+(base+localK)/64;
  const float scale=garnet_affine_coefficient(scaleInput[coefficient]),bias=garnet_affine_coefficient(biasInput[coefficient]);
  const uint4 words=*reinterpret_cast<device const uint4*>(codesInput+offset);
  #pragma clang loop unroll(full)
  for(uint i=0;i<PACKET/4;++i){const uint4 codes=(uint4(words[i/2])>>(uint4(0,4,8,12)+(i%2)*16))&15u;*reinterpret_cast<threadgroup half4*>(weights+(stream*32+localN)*STRIDE+localK+4*i)=half4(fma(float4(codes),float4(scale),float4(bias)));}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto left=a.slice<128,32>(int(base),int(row));
  multiply.run(left,gateWeights,gate);multiply.run(left,upWeights,up);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(fused_shadow){
  auto g=tensor(output,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
  auto u=tensor(outputUp,dextents<int,2>{int(p.columns),int(p.rows)},array<int,2>{1,int(p.oStride)});
  auto gout=g.slice<32,32>(int(column),int(row)),uout=u.slice<32,32>(int(column),int(row));gate.store(gout);up.store(uout);
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<gate.get_capacity();++i){
  auto coordinate=gate.get_multidimensional_index(i);const uint r=row+uint(coordinate[1]),c=column+uint(coordinate[0]);
  const float g=gate[i];const float value=(g/(1.0f+exp(-g)))*up[i];
  outputHalf[r*p.columns+c]=half(value);
  if(fused_writers)atomic_fetch_add_explicit(writers+r*p.columns+c,1u,memory_order_relaxed);
 }
}
