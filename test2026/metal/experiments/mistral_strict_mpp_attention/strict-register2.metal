
// Strict FP32 register softmax follow-up. Same descriptor precision; explicit
// same-library guards qualify only the score/left-P/output register mapping.
kernel void strict_mpp_register_layout(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<float,float,float>();auto b=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<float,float,float>();auto y=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),col=((quad&2)|(lane&1))*2;uint flags=0;
 if(c.get_capacity()!=16||x.get_capacity()!=8||z.get_capacity()!=16)flags|=1;
 for(ushort i=0;i<c.get_capacity();++i){auto xy=c.get_multidimensional_index(i);if(!c.is_valid_element(i)||xy[0]!=col+((i/2)%4)*8+i%2||xy[1]!=row+(i/8)*8)flags|=2;}
 for(ushort i=0;i<x.get_capacity();++i){auto xy=x.get_multidimensional_index(i);if(!x.is_valid_element(i)||xy[0]!=col+((i/2)%2)*8+i%2||xy[1]!=row+(i/4)*8)flags|=4;}
 for(ushort i=0;i<z.get_capacity();++i){auto xy=z.get_multidimensional_index(i);if(!z.is_valid_element(i)||xy[0]!=col+((i/2)%4)*8+i%2||xy[1]!=row+(i/8)*8)flags|=8;}
 out[lane]=flags;
}
kernel void strict_mpp_register_attention(device const float* q [[buffer(0)]],device const float* kc [[buffer(1)]],
 device const float* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],
 constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],
 uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 const auto p=chunk_mps_params(runtime);const uint b=group.z,h=group.x,kh=h/4,baseQ=group.y*64+sg*16;
 if(baseQ>=p.chunk)return;
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),col=((quad&2)|(lane&1))*2;
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto qa=qk.get_left_input_cooperative_tensor<float,float,float>();auto kb=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto score=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
 auto pa=pv.get_left_input_cooperative_tensor<float,float,float>();auto vb=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto acc0=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)acc0[i]=0;
 auto acc1=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)acc1[i]=0;
 auto acc2=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)acc2[i]=0;
 auto acc3=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)acc3[i]=0;
 float maximum[2]={-INFINITY,-INFINITY},denominator[2]={0,0};
 const uint count=controls.count[b],first=controls.first[b],end=first+min(count,baseQ+16);
 for(uint start=0;baseQ<count&&start<end;start+=32) {
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i)score[i]=0;
  #pragma clang loop unroll_count(4)
  for(uint d=0;d<128;d+=16) {
   #pragma clang loop unroll(full)
   for(ushort i=0;i<qa.get_capacity();++i){auto xy=qa.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]);qa[i]=qa.is_valid_element(i)&&t<count?q[(b*p.chunk+t)*p.qStride+h*128+d+uint(xy[0])]:0;}
   #pragma clang loop unroll(full)
   for(ushort i=0;i<kb.get_capacity();++i){auto xy=kb.get_multidimensional_index(i);uint s=start+uint(xy[1]);kb[i]=kb.is_valid_element(i)&&s<end?kc[b*p.kcStride+(kh*p.capacity+s)*128+d+uint(xy[0])]:0;}
   qk.run(qa,kb,score);
  }
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){uint t=baseQ+row+(i/8)*8,s=start+col+((i/2)%4)*8+i%2;score[i]=(t<count&&s<=first+t&&s<end)?score[i]*rsqrt(128.0f):-INFINITY;}
  float factor[2];
  #pragma clang loop unroll(full)
  for(uint part=0;part<2;++part){
   float mx=-INFINITY;
   #pragma clang loop unroll(full)
   for(uint j=0;j<8;++j)mx=max(mx,score[part*8+j]);
   mx=max(mx,simd_shuffle_xor(mx,ushort(1)));mx=max(mx,simd_shuffle_xor(mx,ushort(8)));mx=max(mx,maximum[part]);
   bool valid=baseQ+row+part*8<count;factor[part]=valid?exp(maximum[part]-mx):0;maximum[part]=mx;float sum=0;
   #pragma clang loop unroll(full)
   for(uint j=0;j<8;++j){float a=valid?exp(score[part*8+j]-mx):0;score[part*8+j]=a;sum+=a;}
   sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));denominator[part]=denominator[part]*factor[part]+sum;
  }
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i)acc0[i]*=factor[i/8];
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i)acc1[i]*=factor[i/8];
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i)acc2[i]*=factor[i/8];
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i)acc3[i]*=factor[i/8];
  #pragma clang loop unroll(full)
  for(uint part=0;part<2;++part){
   #pragma clang loop unroll(full)
   for(ushort i=0;i<8;++i)pa[i]=score[(i/4)*8+part*4+i%4];
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+0*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc0);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+1*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc1);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+2*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc2);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+3*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc3);
  }
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+(i/8)*8,d=0*32+col+((i/2)%4)*8+i%2;
  if(t<p.chunk){float value=0;if(t<count){value=acc0[i]/denominator[i/8];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+(i/8)*8,d=1*32+col+((i/2)%4)*8+i%2;
  if(t<p.chunk){float value=0;if(t<count){value=acc1[i]/denominator[i/8];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+(i/8)*8,d=2*32+col+((i/2)%4)*8+i%2;
  if(t<p.chunk){float value=0;if(t<count){value=acc2[i]/denominator[i/8];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+(i/8)*8,d=3*32+col+((i/2)%4)*8+i%2;
  if(t<p.chunk){float value=0;if(t<count){value=acc3[i]/denominator[i/8];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
}
