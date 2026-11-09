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
