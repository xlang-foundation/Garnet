// SPDX-License-Identifier: Apache-2.0
// Strict FP32 MPP online attention. Uses public cooperative coordinate APIs,
// not the relaxed-precision lane map from vLLM's NAX implementation.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
kernel void strict_mpp_query_pack(device const float* q [[buffer(0)]],device const float* active [[buffer(1)]],
 device float* qp [[buffer(2)]],constant ChunkAttentionParams& runtime [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 const auto p=chunk_mps_params(runtime);uint d=id%p.dim,r=id/p.dim,t=r%p.chunk,h=(r/p.chunk)%p.heads,b=r/(p.chunk*p.heads);
 if(b<p.batch)qp[id]=active[b*p.aStride+t]!=0?q[(b*p.chunk+t)*p.qStride+h*p.dim+d]:0;
}
kernel void strict_mpp_layout(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]) {
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<float,float,float>();auto b=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<float,float,float>();auto y=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 out[lane*6+0]=a.get_capacity();out[lane*6+1]=b.get_capacity();out[lane*6+2]=c.get_capacity();out[lane*6+3]=x.get_capacity();out[lane*6+4]=y.get_capacity();out[lane*6+5]=z.get_capacity();
}
kernel void strict_mpp_attention(device const float* q [[buffer(0)]],device const float* kc [[buffer(1)]],
 device const float* vc [[buffer(2)]],device const float* gate [[buffer(3)]],device float* output [[buffer(4)]],
 constant ChunkAttentionParams& runtime [[buffer(5)]],constant MergedKVControls& controls [[buffer(6)]],
 uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 const auto p=chunk_mps_params(runtime);const uint b=group.z,h=group.x,kh=h/4,baseQ=group.y*64+sg*16;
 if(baseQ>=p.chunk)return;
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto qa=qk.get_left_input_cooperative_tensor<float,float,float>();auto kb=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto score=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qa)>,metal::remove_addrspace_t<decltype(kb)>,float>();
 auto pa=pv.get_left_input_cooperative_tensor<float,float,float>();auto vb=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto acc0=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 for(ushort i=0;i<acc0.get_capacity();++i)acc0[i]=0;
 auto acc1=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 for(ushort i=0;i<acc1.get_capacity();++i)acc1[i]=0;
 auto acc2=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 for(ushort i=0;i<acc2.get_capacity();++i)acc2[i]=0;
 auto acc3=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>,metal::remove_addrspace_t<decltype(vb)>,float>();
 for(ushort i=0;i<acc3.get_capacity();++i)acc3[i]=0;
 threadgroup float scores[4*512],factors[4*16],denominators[4*16];
 const uint row=lane/2,sub=lane%2,query=baseQ+row;
 float maximum=-INFINITY,denominator=0;
 const uint count=controls.count[b],first=controls.first[b],end=first+min(count,baseQ+16);
 for(uint start=0;baseQ<count&&start<end;start+=32) {
  for(ushort i=0;i<score.get_capacity();++i)score[i]=0;
  for(uint d=0;d<128;d+=16) {
   for(ushort i=0;i<qa.get_capacity();++i){auto xy=qa.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]);qa[i]=qa.is_valid_element(i)&&t<count?q[(b*p.chunk+t)*p.qStride+h*128+d+uint(xy[0])]:0;}
   for(ushort i=0;i<kb.get_capacity();++i){auto xy=kb.get_multidimensional_index(i);uint s=start+uint(xy[1]);kb[i]=kb.is_valid_element(i)&&s<end?kc[b*p.kcStride+(kh*p.capacity+s)*128+d+uint(xy[0])]:0;}
   qk.run(qa,kb,score);
  }
  for(ushort i=0;i<score.get_capacity();++i)if(score.is_valid_element(i)){auto xy=score.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]),s=start+uint(xy[0]);scores[sg*512+uint(xy[1])*32+uint(xy[0])]=(t<count&&s<=first+t&&s<end)?score[i]*rsqrt(128.0f):-INFINITY;}
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float blockMax=-INFINITY;for(uint s=sub;s<32;s+=2)blockMax=max(blockMax,scores[sg*512+row*32+s]);
  blockMax=max(blockMax,simd_shuffle_xor(blockMax,ushort(1)));
  float next=max(maximum,blockMax),factor=query<count?exp(maximum-next):0,part=0;
  for(uint s=sub;s<32;s+=2){float e=query<count?exp(scores[sg*512+row*32+s]-next):0;scores[sg*512+row*32+s]=e;part+=e;}
  part+=simd_shuffle_xor(part,ushort(1));denominator=denominator*factor+part;maximum=next;
  if(!sub){factors[sg*16+row]=factor;denominators[sg*16+row]=denominator;}
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for(ushort i=0;i<acc0.get_capacity();++i)if(acc0.is_valid_element(i)){auto xy=acc0.get_multidimensional_index(i);acc0[i]*=factors[sg*16+uint(xy[1])];}
  for(ushort i=0;i<acc1.get_capacity();++i)if(acc1.is_valid_element(i)){auto xy=acc1.get_multidimensional_index(i);acc1[i]*=factors[sg*16+uint(xy[1])];}
  for(ushort i=0;i<acc2.get_capacity();++i)if(acc2.is_valid_element(i)){auto xy=acc2.get_multidimensional_index(i);acc2[i]*=factors[sg*16+uint(xy[1])];}
  for(ushort i=0;i<acc3.get_capacity();++i)if(acc3.is_valid_element(i)){auto xy=acc3.get_multidimensional_index(i);acc3[i]*=factors[sg*16+uint(xy[1])];}
  for(uint part=0;part<2;++part) {
   for(ushort i=0;i<pa.get_capacity();++i){auto xy=pa.get_multidimensional_index(i);pa[i]=pa.is_valid_element(i)?scores[sg*512+uint(xy[1])*32+part*16+uint(xy[0])]:0;}
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+0*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc0);
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+1*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc1);
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+2*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc2);
   for(ushort i=0;i<vb.get_capacity();++i){auto xy=vb.get_multidimensional_index(i);uint s=start+part*16+uint(xy[1]);vb[i]=vb.is_valid_element(i)&&s<end?vc[b*p.vcStride+(kh*p.capacity+s)*128+3*32+uint(xy[0])]:0;}
   pv.run(pa,vb,acc3);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(ushort i=0;i<acc0.get_capacity();++i)if(acc0.is_valid_element(i)){
  auto xy=acc0.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]),d=0*32+uint(xy[0]);
  if(t<p.chunk){float value=0;if(t<count){value=acc0[i]/denominators[sg*16+uint(xy[1])];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 for(ushort i=0;i<acc1.get_capacity();++i)if(acc1.is_valid_element(i)){
  auto xy=acc1.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]),d=1*32+uint(xy[0]);
  if(t<p.chunk){float value=0;if(t<count){value=acc1[i]/denominators[sg*16+uint(xy[1])];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 for(ushort i=0;i<acc2.get_capacity();++i)if(acc2.is_valid_element(i)){
  auto xy=acc2.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]),d=2*32+uint(xy[0]);
  if(t<p.chunk){float value=0;if(t<count){value=acc2[i]/denominators[sg*16+uint(xy[1])];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
 for(ushort i=0;i<acc3.get_capacity();++i)if(acc3.is_valid_element(i)){
  auto xy=acc3.get_multidimensional_index(i);uint t=baseQ+uint(xy[1]),d=3*32+uint(xy[0]);
  if(t<p.chunk){float value=0;if(t<count){value=acc3[i]/denominators[sg*16+uint(xy[1])];if(p.gated)value/=1+exp(-gate[(b*p.chunk+t)*p.gStride+h*128+d]);}output[(b*p.chunk+t)*p.oStride+h*128+d]=value;}
 }
}
