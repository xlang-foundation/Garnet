// SPDX-License-Identifier: Apache-2.0
// Half-rounded QKV, FP32 accumulation/softmax/output; one/two half P components.

// Strict FP32 register softmax follow-up. Same descriptor precision; explicit
// same-library guards qualify only the score/left-P/output register mapping.
kernel void half_policy_layout(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
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

 for(ushort i=0;i<a.get_capacity();++i){auto xy=a.get_multidimensional_index(i);if(!a.is_valid_element(i)||xy[0]!=col+i%4||xy[1]!=row+(i/4)*8)flags|=16;}
 for(ushort i=0;i<b.get_capacity();++i){auto xy=b.get_multidimensional_index(i);if(!b.is_valid_element(i)||xy[0]!=col+i%4||xy[1]!=row+(i/4)*8)flags|=32;}
 for(ushort i=0;i<y.get_capacity();++i){auto xy=y.get_multidimensional_index(i);if(!y.is_valid_element(i)||xy[0]!=col+(i/8)*16+i%4||xy[1]!=row+((i/4)%2)*8)flags|=64;}
 out[lane]=flags;
}
template<bool HIGH_LOW> void half_policy_attention_impl(device const half* q,device const half* kc,
 device const half* vc,device const float* gate,device float* output,
 constant ChunkAttentionParams& runtime,constant MergedKVControls& controls,
 uint3 group,uint sg,uint lane) {
 const auto p=chunk_mps_params(runtime);const uint b=group.z,h=group.x,kh=h/4,baseQ=group.y*64+sg*16;
 if(baseQ>=p.chunk)return;
 const uint quad=lane>>2,row=(quad&4)|((lane>>1)&3),col=((quad&2)|(lane&1))*4;
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto qa=qk.get_left_input_cooperative_tensor<half,half,float>();auto kb=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto score=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
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
 auto paLow=pv.get_left_input_cooperative_tensor<half,half,float>();
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
  for(ushort i=0;i<16;++i){score[i]=0;}
  #pragma clang loop unroll_count(4)
  for(uint d=0;d<128;d+=16) {
   #pragma clang loop unroll(full)
   for(uint part=0;part<2;++part){uint t=baseQ+row+part*8;half4 value=t<count?*reinterpret_cast<device const half4*>(q+(b*p.chunk+t)*p.qStride+h*128+d+col):half4(0);for(uint j=0;j<4;++j)qa[part*4+j]=value[j];}
   #pragma clang loop unroll(full)
   for(uint part=0;part<4;++part){uint token=start+row+part*8;half4 value=token<end?*reinterpret_cast<device const half4*>(kc+b*p.kcStride+(kh*p.capacity+token)*128+d+col):half4(0);for(uint j=0;j<4;++j)kb[part*4+j]=value[j];}
   qk.run(qa,kb,score);
  }
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,s=start+col+(i/8)*16+i%4;score[i]=(t<count&&s<=first+t&&s<end)?score[i]*rsqrt(128.0f):-INFINITY;}
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
  for(ushort i=0;i<16;++i){acc0[i]*=factor[(i/4)%2];if(HIGH_LOW)corr0[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc1[i]*=factor[(i/4)%2];if(HIGH_LOW)corr1[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc2[i]*=factor[(i/4)%2];if(HIGH_LOW)corr2[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(ushort i=0;i<16;++i){acc3[i]*=factor[(i/4)%2];if(HIGH_LOW)corr3[i]*=factor[(i/4)%2];}
  #pragma clang loop unroll(full)
  for(uint part=0;part<2;++part){
   #pragma clang loop unroll(full)
   for(ushort i=0;i<8;++i){float value=score[part*8+i]*32768.0f;pa[i]=half(value);if(HIGH_LOW)paLow[i]=half(value-float(pa[i]));}
   #pragma clang loop unroll(full)
   for(uint frag=0;frag<4;++frag){uint token=start+part*16+row+(frag%2)*8,feature=0*32+col+(frag/2)*16;half4 value=token<end?*reinterpret_cast<device const half4*>(vc+b*p.vcStride+(kh*p.capacity+token)*128+feature):half4(0);for(uint j=0;j<4;++j)vb[frag*4+j]=value[j];}
   pv.run(pa,vb,acc0);if(HIGH_LOW)pv.run(paLow,vb,corr0);
   #pragma clang loop unroll(full)
   for(uint frag=0;frag<4;++frag){uint token=start+part*16+row+(frag%2)*8,feature=1*32+col+(frag/2)*16;half4 value=token<end?*reinterpret_cast<device const half4*>(vc+b*p.vcStride+(kh*p.capacity+token)*128+feature):half4(0);for(uint j=0;j<4;++j)vb[frag*4+j]=value[j];}
   pv.run(pa,vb,acc1);if(HIGH_LOW)pv.run(paLow,vb,corr1);
   #pragma clang loop unroll(full)
   for(uint frag=0;frag<4;++frag){uint token=start+part*16+row+(frag%2)*8,feature=2*32+col+(frag/2)*16;half4 value=token<end?*reinterpret_cast<device const half4*>(vc+b*p.vcStride+(kh*p.capacity+token)*128+feature):half4(0);for(uint j=0;j<4;++j)vb[frag*4+j]=value[j];}
   pv.run(pa,vb,acc2);if(HIGH_LOW)pv.run(paLow,vb,corr2);
   #pragma clang loop unroll(full)
   for(uint frag=0;frag<4;++frag){uint token=start+part*16+row+(frag%2)*8,feature=3*32+col+(frag/2)*16;half4 value=token<end?*reinterpret_cast<device const half4*>(vc+b*p.vcStride+(kh*p.capacity+token)*128+feature):half4(0);for(uint j=0;j<4;++j)vb[frag*4+j]=value[j];}
   pv.run(pa,vb,acc3);if(HIGH_LOW)pv.run(paLow,vb,corr3);
  }
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=0*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc0[i]+(HIGH_LOW?corr0[i]:0.0f))/(denominator[(i/4)%2]*32768.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=1*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc1[i]+(HIGH_LOW?corr1[i]:0.0f))/(denominator[(i/4)%2]*32768.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=2*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc2[i]+(HIGH_LOW?corr2[i]:0.0f))/(denominator[(i/4)%2]*32768.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 #pragma clang loop unroll(full)
 for(ushort i=0;i<16;++i){uint t=baseQ+row+((i/4)%2)*8,d=3*32+col+(i/8)*16+i%4;
  if(t<p.chunk){float value=0;if(t<count){value=(acc3[i]+(HIGH_LOW?corr3[i]:0.0f))/(denominator[(i/4)%2]*32768.0f);if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
}
kernel void half_policy_p16(device const half* q [[buffer(0)]],device const half* kc [[buffer(1)]],device const half* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){half_policy_attention_impl<false>(q,kc,vc,gate,output,runtime,controls,group,sg,lane);}
kernel void half_policy_p22(device const half* q [[buffer(0)]],device const half* kc [[buffer(1)]],device const half* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){half_policy_attention_impl<true>(q,kc,vc,gate,output,runtime,controls,group,sg,lane);}
kernel void half_policy_capacities(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<half,half,float>();auto b=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<half,half,float>();auto y=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 out[lane*6+0]=a.get_capacity();out[lane*6+1]=b.get_capacity();out[lane*6+2]=c.get_capacity();out[lane*6+3]=x.get_capacity();out[lane*6+4]=y.get_capacity();out[lane*6+5]=z.get_capacity();
}
kernel void half_policy_coordinates(device atomic_uint* out [[buffer(0)]]){
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<half,half,float>();auto b=qk.get_right_input_cooperative_tensor<half,half,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<half,half,float>();auto y=pv.get_right_input_cooperative_tensor<half,half,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 strict_coop_count(a,out,out+2560,16,16);strict_coop_count(b,out+256,out+2560,16,32);strict_coop_count(c,out+768,out+2560,32,16);
 strict_coop_count(x,out+1280,out+2560,16,16);strict_coop_count(y,out+1536,out+2560,32,16);strict_coop_count(z,out+2048,out+2560,32,16);
}

kernel void half_policy_pack_q(device const float* q [[buffer(0)]],device half* packed [[buffer(1)]],constant ChunkAttentionParams& runtime [[buffer(2)]],uint n [[thread_position_in_grid]]){
 const auto p=chunk_mps_params(runtime);uint row=n/4096,d=n%4096;if(row<p.batch*p.chunk)packed[row*p.qStride+d]=half(q[row*p.qStride+d]);
}

kernel void half_policy_pack_kv(device const float* k [[buffer(0)]],device const float* v [[buffer(1)]],device half* hk [[buffer(2)]],device half* hv [[buffer(3)]],constant ChunkAttentionParams& runtime [[buffer(4)]],constant MergedKVControls& controls [[buffer(5)]],constant uint& extent [[buffer(6)]],uint n [[thread_position_in_grid]]){
 const auto p=chunk_mps_params(runtime);uint d=n%128,row=n/128,t=row%extent,head=(row/extent)%8,b=row/(extent*8);if(b>=p.batch)return;
 uint at=b*p.kcStride+(head*p.capacity+t)*128+d;bool valid=controls.count[b]&&t<controls.first[b]+controls.count[b];hk[at]=valid?half(k[at]):half(0);hv[at]=valid?half(v[at]):half(0);
}
