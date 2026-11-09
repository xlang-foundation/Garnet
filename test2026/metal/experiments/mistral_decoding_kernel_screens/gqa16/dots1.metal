// Fixed Mistral B1/B4,T4,H32,KV8,D128,C4096. Shared K tile,16 Q rows.
// Raw dot products retain lane order; the original full-row softmax loads scale.
kernel void verification_gqa16_dots(device const float* query [[buffer(0)]],
 device const float* key [[buffer(1)]],device const float* keyCache [[buffer(2)]],
 device const float* slots [[buffer(3)]],device const float* active [[buffer(4)]],
 device float* output [[buffer(5)]],constant VerifyAttentionParams& runtime [[buffer(6)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 GARNET_VERIFY_PARAMS;
 const uint row=group.x/8,kv=group.x%8,tile=group.y*32;
 if(row>=p.batch)return;
 const uint first=uint(slots[row*uint(p.sStride)]);
 uint visible[4];bool enabled[4];uint maximum=0;
 #pragma clang loop unroll(full)
 for(uint t=0;t<4;++t){enabled[t]=active[row*uint(p.aStride)+t]!=0.0f;visible[t]=enabled[t]?min(4096u,uint(slots[row*uint(p.sStride)+t])+1):0;maximum=max(maximum,visible[t]);}
 threadgroup float keys[32*128];
 for(uint index=tid;index<32*128;index+=128){
  const uint s=tile+index/128,j=index%128;
  keys[index]=s<maximum?(s<first?keyCache[row*uint(p.kcStride)+(kv*4096+s)*128+j]:key[(row*4+s-first)*uint(p.kStride)+kv*128+j]):0.0f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float4 queries[16];
 #pragma clang loop unroll(full)
 for(uint q=0;q<16;++q){const uint t=q/4,head=kv*4+q%4,base=(row*4+t)*uint(p.qStride)+head*128+lane;queries[q]=enabled[t]?float4(query[base],query[base+32],query[base+64],query[base+96]):float4(0.0f);}
 for(uint local=simd;local<32;local+=4){
  const uint s=tile+local;
  const float4 k=float4(keys[local*128+lane],keys[local*128+lane+32],keys[local*128+lane+64],keys[local*128+lane+96]);
  #pragma clang loop unroll(full)
  for(uint q=0;q<16;++q){
   const uint t=q/4,head=kv*4+q%4;
   if(s<visible[t]){
    float score=0.0f;
    score+=queries[q].x*k.x;score+=queries[q].y*k.y;
    score+=queries[q].z*k.z;score+=queries[q].w*k.w;
    score=simd_sum(score);
    if(lane==0)output[(row*4+t)*uint(p.pStride)+head*4096+s]=score;
   }
  }
 }
}
