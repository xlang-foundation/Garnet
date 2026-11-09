// SPDX-License-Identifier: Apache-2.0
// Two-half decomposition with FP32 high/correction accumulators; strict MPP.

// Strict FP32 register softmax follow-up. Same descriptor precision; explicit
// same-library guards qualify only the score/left-P/output register mapping.
kernel void compensated_mpp_layout(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<half,half,float>();auto b=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<half,half,float>();auto y=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),col=((quad&2)|(lane&1))*4;uint flags=0;
 if(c.get_capacity()!=16||x.get_capacity()!=8||z.get_capacity()!=16)flags|=1;
 for(ushort i=0;i<c.get_capacity();++i){auto xy=c.get_multidimensional_index(i);if(!c.is_valid_element(i)||xy[0]!=col+(i/8)*16+i%4||xy[1]!=row+((i/4)%2)*8)flags|=2;}
 for(ushort i=0;i<x.get_capacity();++i){auto xy=x.get_multidimensional_index(i);if(!x.is_valid_element(i)||xy[0]!=col+i%4||xy[1]!=row+(i/4)*8)flags|=4;}
 for(ushort i=0;i<z.get_capacity();++i){auto xy=z.get_multidimensional_index(i);if(!z.is_valid_element(i)||xy[0]!=col+(i/8)*16+i%4||xy[1]!=row+((i/4)%2)*8)flags|=8;}
 out[lane]=flags;
}
template<bool LOW_LOW> void compensated_mpp_attention_impl(device const float* q,device const float* kc,
 device const float* vc,device const float* gate,device float* output,
 constant ChunkAttentionParams& runtime,constant MergedKVControls& controls,
 uint3 group,uint sg,uint lane) {
 const auto p=chunk_mps_params(runtime);const uint b=group.z,h=group.x,kh=h/4,baseQ=group.y*64+sg*16;
 if(baseQ>=p.chunk)return;
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),col=((quad&2)|(lane&1))*4;
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto qa=qk.get_left_input_cooperative_tensor<half,half,float>();auto kb=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto score=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
 auto qaLow=qk.get_left_input_cooperative_tensor<half,half,float>();auto kbLow=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto correction=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
 auto pa=pv.get_left_input_cooperative_tensor<half,half,float>();auto vb=pv.get_right_input_cooperative_tensor<half,half,float>();
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
 auto paLow=pv.get_left_input_cooperative_tensor<half,half,float>();auto vbLow=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto corr0=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)corr0[i]=0;
 auto corr1=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)corr1[i]=0;
 auto corr2=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)corr2[i]=0;
 auto corr3=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i)corr3[i]=0;
 float maximum[2]={-INFINITY,-INFINITY},denominator[2]={0,0};
 const uint count=controls.count[b],first=controls.first[b],end=first+min(count,baseQ+16);
 for(uint start=0;baseQ<count&&start<end;start+=32) {
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){score[i]=0;correction[i]=0;}
  #pragma clang loop unroll_count(4)
  for(uint d=0;d<128;d+=16) {
   #pragma clang loop unroll(full)
   for(ushort i=0;i<qa.get_capacity();++i){auto xy=qa.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]);float value=qa.is_valid_element(i)&&t<count?q[(b*p.chunk+t)*p.qStride+h*128+d+uint(xy[0])]*16.0f:0;qa[i]=half(value);qaLow[i]=half(value-float(qa[i]));}
   #pragma clang loop unroll(full)
   for(ushort i=0;i<kb.get_capacity();++i){auto xy=kb.get_multidimensional_index(i);uint s=start+uint(xy[1]);float value=kb.is_valid_element(i)&&s<end?kc[b*p.kcStride+(kh*p.capacity+s)*128+d+uint(xy[0])]*16.0f:0;kb[i]=half(value);kbLow[i]=half(value-float(kb[i]));}
   qk.run(qa,kb,score);qk.run(qaLow,kb,correction);qk.run(qa,kbLow,correction);if(LOW_LOW)qk.run(qaLow,kbLow,correction);
  }
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,s=start+col+(i/8)*16+i%4;score[i]=(t<count&&s<=first+t&&s<end)?(score[i]+correction[i])*(rsqrt(128.0f)/256.0f):-INFINITY;}
  float factor[2];
  #pragma clang loop unroll(full)
  for(uint part=0;part<2;++part){
   float mx=-INFINITY;
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;++j)mx=max(mx,max(score[part*4+j],score[8+part*4+j]));
   mx=max(mx,simd_shuffle_xor(mx,ushort(1)));mx=max(mx,simd_shuffle_xor(mx,ushort(8)));mx=max(mx,maximum[part]);
   bool valid=baseQ+row+part*8<count;factor[part]=valid?exp(maximum[part]-mx):0;maximum[part]=mx;float sum=0;
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;++j){float a=valid?exp(score[part*4+j]-mx):0,b=valid?exp(score[8+part*4+j]-mx):0;score[part*4+j]=a;score[8+part*4+j]=b;sum+=a+b;}
   sum+=simd_shuffle_xor(sum,ushort(1));sum+=simd_shuffle_xor(sum,ushort(8));denominator[part]=denominator[part]*factor[part]+sum;
  }
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc0[i]*=factor[(i/4)%2];corr0[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc1[i]*=factor[(i/4)%2];corr1[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc2[i]*=factor[(i/4)%2];corr2[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc3[i]*=factor[(i/4)%2];corr3[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(uint part=0;part<2;++part){
   #pragma clang loop unroll(full)
   for(ushort i=0;i<8;++i){float value=score[part*8+i]*32768.0f;pa[i]=half(value);paLow[i]=half(value-float(pa[i]));}
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);float value=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+0*32+uint(xy[0])]*16.0f:0;vb[i]=half(value);vbLow[i]=half(value-float(vb[i]));}
   pv.run(pa,vb,acc0);pv.run(paLow,vb,corr0);pv.run(pa,vbLow,corr0);if(LOW_LOW)pv.run(paLow,vbLow,corr0);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);float value=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+1*32+uint(xy[0])]*16.0f:0;vb[i]=half(value);vbLow[i]=half(value-float(vb[i]));}
   pv.run(pa,vb,acc1);pv.run(paLow,vb,corr1);pv.run(pa,vbLow,corr1);if(LOW_LOW)pv.run(paLow,vbLow,corr1);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);float value=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+2*32+uint(xy[0])]*16.0f:0;vb[i]=half(value);vbLow[i]=half(value-float(vb[i]));}
   pv.run(pa,vb,acc2);pv.run(paLow,vb,corr2);pv.run(pa,vbLow,corr2);if(LOW_LOW)pv.run(paLow,vbLow,corr2);
   #pragma clang loop unroll(full)
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);float value=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+3*32+uint(xy[0])]*16.0f:0;vb[i]=half(value);vbLow[i]=half(value-float(vb[i]));}
   pv.run(pa,vb,acc3);pv.run(paLow,vb,corr3);pv.run(pa,vbLow,corr3);if(LOW_LOW)pv.run(paLow,vbLow,corr3);
  }
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=0*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc0[i]+corr0[i])/(denominator[(i/4)%2]*524288.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=1*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc1[i]+corr1[i])/(denominator[(i/4)%2]*524288.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=2*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc2[i]+corr2[i])/(denominator[(i/4)%2]*524288.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=3*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc3[i]+corr3[i])/(denominator[(i/4)%2]*524288.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
}
kernel void compensated_mpp_three(device const float* q [[buffer(0)]],device const float* kc [[buffer(1)]],device const float* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){compensated_mpp_attention_impl<false>(q,kc,vc,gate,output,runtime,controls,group,sg,lane);}
kernel void compensated_mpp_four(device const float* q [[buffer(0)]],device const float* kc [[buffer(1)]],device const float* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){compensated_mpp_attention_impl<true>(q,kc,vc,gate,output,runtime,controls,group,sg,lane);}
kernel void compensated_mpp_capacities(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<half,half,float>();auto b=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<half,half,float>();auto y=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 out[lane*6+0]=a.get_capacity();out[lane*6+1]=b.get_capacity();out[lane*6+2]=c.get_capacity();out[lane*6+3]=x.get_capacity();out[lane*6+4]=y.get_capacity();out[lane*6+5]=z.get_capacity();
}
kernel void compensated_mpp_coordinates(device atomic_uint* out [[buffer(0)]]){
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<half,half,float>();auto b=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<half,half,float>();auto y=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 strict_coop_count(a,out,out+2560,16,16);strict_coop_count(b,out+256,out+2560,16,32);strict_coop_count(c,out+768,out+2560,32,16);
 strict_coop_count(x,out+1280,out+2560,16,16);strict_coop_count(y,out+1536,out+2560,32,16);strict_coop_count(z,out+2048,out+2560,32,16);
}
