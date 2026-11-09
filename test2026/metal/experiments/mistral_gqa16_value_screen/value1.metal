// Fixed B1/B4,T4,H32,KV8,D128,C4096,ungated. Each output preserves key order.
constant uint verification_value_tile [[function_constant(2)]];
kernel void verification_gqa16_context(device const float* probabilities [[buffer(0)]],
 device const float* value [[buffer(1)]],device const float* valueCache [[buffer(2)]],
 device const float* gate [[buffer(3)]],device const float* slots [[buffer(4)]],
 device const float* active [[buffer(5)]],device float* output [[buffer(6)]],
 constant VerifyAttentionParams& runtime [[buffer(7)]],
 threadgroup float* shared [[threadgroup(0)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
 GARNET_VERIFY_PARAMS;
 const uint row=group.x/8,kv=group.x%8,column=group.y*32+(tid%8)*4;
 if(row>=p.batch)return;
 const uint query=tid/8,token=query/4,head=kv*4+query%4,flat=row*4+token;
 auto out=reinterpret_cast<device packed_float4*>(output+flat*uint(p.oStride)+head*128+column);
 uint visible[4],maximum=0;bool enabled[4];
 #pragma clang loop unroll(full)
 for(uint t=0;t<4;++t){enabled[t]=active[row*uint(p.aStride)+t]!=0.0f;visible[t]=0;if(enabled[t])visible[t]=uint(slots[row*uint(p.sStride)+t])+1;maximum=max(maximum,visible[t]);}
 if(maximum==0){*out=packed_float4(0.0f);return;}
 const uint first=uint(slots[row*uint(p.sStride)]),tile=verification_value_tile;
 threadgroup float* values=shared;threadgroup float* probs=shared+tile*32;
 float4 result=float4(0.0f);
 for(uint begin=0;begin<maximum;begin+=tile){
  for(uint index=tid;index<tile*32;index+=128){
   const uint s=begin+index/32,d=group.y*32+index%32;
   values[index]=s<maximum?(s<first?valueCache[row*uint(p.vcStride)+(kv*4096+s)*128+d]:value[(row*4+s-first)*uint(p.vStride)+kv*128+d]):0.0f;
  }
  for(uint index=tid;index<16*tile;index+=128){
   const uint q=index/tile,s=begin+index%tile,t=q/4,h=kv*4+q%4;
   probs[index]=s<visible[t]?probabilities[(row*4+t)*uint(p.pStride)+h*4096+s]:0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(enabled[token])for(uint s=0;s<min(tile,maximum-begin);++s)if(begin+s<visible[token]){
   const float4 v=float4(*reinterpret_cast<threadgroup const packed_float4*>(values+s*32+(tid%8)*4));
   result+=probs[query*tile+s]*v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 *out=packed_float4(result);
}
