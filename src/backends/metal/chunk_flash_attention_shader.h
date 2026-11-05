// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Register Q/output tiling and online softmax follow the FlashAttention design.
// The 8x8 fragment lane mapping was cross-checked against vLLM-Metal's Apache2
// pagedattention_tiled.metal at a44be48731005932a24dd30b55c21e992dc90b9a.
// This kernel uses Garnet's FP32 dense old/live KV operands and prepared native
// owner controls. It neither downcasts operands nor stores global scores.
inline constexpr const char* kChunkFlashAttentionShader=R"METAL(
struct ChunkFlashControls {uint first[4];uint count[4];};
inline float chunk_flash_row_max(float x) {
    x=max(x,simd_shuffle_xor(x,ushort(1)));
    return max(x,simd_shuffle_xor(x,ushort(8)));
}
inline float chunk_flash_row_sum(float x) {
    x+=simd_shuffle_xor(x,ushort(1));return x+simd_shuffle_xor(x,ushort(8));
}
kernel void chunk_flash_attention_f32(
    device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* oldKey [[buffer(3)]],
    device const float* oldValue [[buffer(4)]],device const float* gate [[buffer(5)]],
    device float* output [[buffer(6)]],constant ChunkAttentionParams& p [[buffer(7)]],
    constant ChunkFlashControls& controls [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D=128,LD=132,BQ=32,BK=16,TD=16,TK=2;
    const uint b=group.z,h=group.x,start=group.y*BQ,kh=h/(uint(p.heads)/uint(p.kvHeads));
    if(start>=controls.count[b]) {
        for(uint i=tid;i<BQ*D;i+=128) {
            uint token=start+i/D;
            if(token<uint(p.chunk))output[(b*uint(p.chunk)+token)*uint(p.oStride)+h*D+i%D]=0;
        }
        return; // uniform for the complete threadgroup
    }
    const uint valid=min(BQ,controls.count[b]-start),first=controls.first[b];
    threadgroup float scratch[BQ*LD];
    for(uint i=tid;i<BQ*D;i+=128) {
        uint row=i/D,d=i%D;
        scratch[row*LD+d]=row<valid?query[((b*uint(p.chunk)+start+row)*uint(p.qStride))+h*D+d]:0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 qr[TD],accum[TD];
    #pragma unroll
    for(uint d=0;d<TD;++d) {
        simdgroup_matrix<float,8,8> m;
        simdgroup_load(m,scratch+simd*8*LD+d*8,LD);
        qr[d]=reinterpret_cast<thread float2&>(m.thread_elements());accum[d]=float2(0);
    }
    // Q now lives in registers. Reuse its complete shared allocation for K/V.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* ks=scratch;threadgroup float* vs=scratch+BK*LD;
    const uint laneGroup=lane/4;
    const uint row=(laneGroup&4)+(lane/2)%4,col=(laneGroup&2)*2+(lane%2)*2;
    const uint t=start+simd*8+row;
    const bool masked=simd*8+row>=valid;
    float maximum=-INFINITY,denominator=0;
    const uint stop=first+start+valid;
    const float scale=rsqrt(float(D));
    for(uint base=0;base<stop;base+=BK) {
        for(uint i=tid;i<BK*D;i+=128) {
            uint r=i/D,d=i%D,pos=base+r;
            float k=0,v=0;
            if(pos<stop) {
                if(pos>=first) {
                    uint token=pos-first;
                    k=key[(b*uint(p.chunk)+token)*uint(p.kStride)+kh*D+d];
                    v=value[(b*uint(p.chunk)+token)*uint(p.vStride)+kh*D+d];
                }else {
                    k=oldKey[b*uint(p.kcStride)+(kh*uint(p.capacity)+pos)*D+d];
                    v=oldValue[b*uint(p.vcStride)+(kh*uint(p.capacity)+pos)*D+d];
                }
            }
            ks[r*LD+d]=k;vs[r*LD+d]=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 scores[TK];
        #pragma unroll
        for(uint k=0;k<TK;++k)scores[k]=float2(0);
        #pragma unroll
        for(uint d=0;d<TD;++d) {
            simdgroup_matrix<float,8,8> q;
            reinterpret_cast<thread float2&>(q.thread_elements())=qr[d];
            #pragma unroll
            for(uint k=0;k<TK;++k) {
                simdgroup_matrix<float,8,8> keyTile,s;
                simdgroup_load(keyTile,ks+k*8*LD+d*8,LD,ulong2(0),true);
                reinterpret_cast<thread float2&>(s.thread_elements())=scores[k];
                simdgroup_multiply_accumulate(s,q,keyTile,s);
                scores[k]=reinterpret_cast<thread float2&>(s.thread_elements());
            }
        }
        float local=-INFINITY;
        for(uint k=0;k<TK;++k)for(uint j=0;j<2;++j) {
            uint pos=base+k*8+col+j;
            scores[k][j]=masked||pos>first+t?-INFINITY:scores[k][j]*scale;
            local=max(local,scores[k][j]);
        }
        float next=max(maximum,chunk_flash_row_max(local));
        float factor=next>maximum?(maximum==-INFINITY?0:exp(maximum-next)):1;
        float sum=0;
        for(uint k=0;k<TK;++k)for(uint j=0;j<2;++j) {
            float v=scores[k][j]==-INFINITY?0:exp(scores[k][j]-next);
            scores[k][j]=v;sum+=v;
        }
        maximum=next;denominator=denominator*factor+chunk_flash_row_sum(sum);
        #pragma unroll
        for(uint d=0;d<TD;++d)accum[d]*=factor;
        #pragma unroll
        for(uint k=0;k<TK;++k) {
            simdgroup_matrix<float,8,8> probability;
            reinterpret_cast<thread float2&>(probability.thread_elements())=scores[k];
            #pragma unroll
            for(uint d=0;d<TD;++d) {
                simdgroup_matrix<float,8,8> v,o;
                simdgroup_load(v,vs+k*8*LD+d*8,LD);
                reinterpret_cast<thread float2&>(o.thread_elements())=accum[d];
                simdgroup_multiply_accumulate(o,probability,v,o);
                accum[d]=reinterpret_cast<thread float2&>(o.thread_elements());
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint d=0;d<TD;++d) {
        simdgroup_matrix<float,8,8> o;
        reinterpret_cast<thread float2&>(o.thread_elements())=masked?float2(0):accum[d]/denominator;
        simdgroup_store(o,scratch+simd*8*LD+d*8,LD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<BQ*D;i+=128) {
        uint r=i/D,d=i%D,token=start+r;
        if(token>=uint(p.chunk))continue;
        uint index=(b*uint(p.chunk)+token)*uint(p.oStride)+h*D+d;
        float result=r<valid?scratch[r*LD+d]:0;
        if(p.gated&&r<valid)result/=1+exp(-gate[(b*uint(p.chunk)+token)*uint(p.gStride)+h*D+d]);
        output[index]=result;
    }
}
)METAL";
}
