// SPDX-License-Identifier: Apache-2.0
// STAGED ONLY: consumed by a future retained-context learned MTP draft runner.
#include "qwen35_mtp_draft_control.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace Garnet { namespace {
__host__ __device__ bool ValidConfig(const MTPDraftDeviceConfig& c){
    if((c.batch!=1&&c.batch!=2&&c.batch!=4&&c.batch!=8)||(c.hidden!=2048&&c.hidden!=2560)||
       c.vocab<4||c.vocab>(1<<24)||c.capacity<32||c.capacity>8192||c.stopCount<1||c.stopCount>4)return false;
    for(int i=0;i<c.stopCount;++i)if(c.stops[i]<0||c.stops[i]>=c.vocab)return false;
    return true;
}
__device__ bool IsStop(const MTPDraftDeviceConfig& c,long long id){
    for(int i=0;i<c.stopCount;++i)if(id==c.stops[i])return true;return false;
}
__device__ void Fail(MTPDraftDeviceState& s,int code){if(!s.error)s.error=code;s.anyAdvance=0;}
__global__ void Initialize(MTPDraftDeviceConfig c,const int* controls,MTPDraftDeviceState* state,long long* proposals){
    if(blockIdx.x||threadIdx.x)return;
    auto& s=*state;int* words=reinterpret_cast<int*>(&s);
    for(int i=0;i<int(sizeof(s)/sizeof(int));++i)words[i]=0;
    if(!ValidConfig(c)){s.error=1;return;}
    int maximum=0;
    for(int b=0;b<c.batch;++b){
        const int* row=controls+4*b;int active=row[0],remaining=row[1],position=row[2],slot=row[3];
        if((active!=0&&active!=1)||remaining<0||remaining>128||position<0||position>1048576-128||slot<0||slot>c.capacity||
           (active&&remaining<2)){Fail(s,2);return;}
        s.active[b]=active;s.remaining[b]=remaining;s.positions[b]=position;s.slots[b]=slot;
        if(active&&remaining>maximum)maximum=remaining;
        for(int t=0;t<3;++t)proposals[b*3+t]=0;
    }
    s.count=maximum-1;if(s.count>3)s.count=3;
    if(s.count<1){Fail(s,3);return;}
}
// Produce exactly one _draft iteration. Model calls exist only for j<count;
// inactive/short rows must still get the ordinary head's unused predictions.
__device__ void PrepareBody(const MTPDraftDeviceConfig& c,int stage,const long long* next,
    long long* proposals,MTPDraftDeviceState& s){
    s.anyAdvance=0;
    for(int b=0;b<c.batch;++b)s.advance[b]=0;
    if(s.error)return;
    if(stage<1||stage>3||s.stage!=stage-1){Fail(s,4);return;}
    s.stage=stage;if(stage>s.count)return;
    for(int b=0;b<c.batch;++b)if(next[b]<0||next[b]>=c.vocab){Fail(s,5);return;}
    for(int b=0;b<c.batch;++b){
        proposals[b*3+stage-1]=s.active[b]&&!s.stopped[b]?next[b]:0;
        const bool eligible=s.active[b]&&!s.stopped[b]&&stage<s.remaining[b];
        if(eligible)++s.proposed;
        if(eligible&&IsStop(c,next[b]))s.stopped[b]=1;
        s.advance[b]=int(stage<s.count&&s.active[b]&&!s.stopped[b]&&stage<s.remaining[b]-1);
        s.anyAdvance|=s.advance[b];
    }
    if(!s.anyAdvance)return;
    for(int b=0;b<c.batch;++b)if(s.advance[b]&&s.slots[b]>=c.capacity){Fail(s,6);return;}
    for(int b=0;b<c.batch;++b){s.stepSlots[b]=s.slots[b];s.slots[b]+=s.advance[b];}
    ++s.modelCalls;
}
__global__ void Prepare(MTPDraftDeviceConfig c,int stage,const long long* next,long long* proposals,
    MTPDraftDeviceState* state,cudaGraphConditionalHandle handle){
    if(blockIdx.x||threadIdx.x)return;
    auto& s=*state;
    if(!ValidConfig(c)){Fail(s,1);if(handle)cudaGraphSetConditional(handle,0);return;}
    PrepareBody(c,stage,next,proposals,s);if(handle)cudaGraphSetConditional(handle,s.error?0:s.anyAdvance);
}
__global__ void PrepareInputs(MTPDraftDeviceConfig c,int stage,const __nv_bfloat16* table,const long long* proposals,
    const MTPDraftDeviceState* state,const float* last,float* target,float* embedding,float* positions,float* slots,float* active){
    if(state->error||!state->anyAdvance)return;
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;const size_t total=size_t(c.batch)*4*c.hidden;
    if(index<total){
        int b=int(index/(4*c.hidden)),t=int(index/c.hidden)%4,d=int(index%c.hidden);
        target[index]=t==0?last[size_t(b)*c.hidden+d]:0.f;
        long long token=proposals[b*3+stage-1];
        embedding[index]=t==0?__bfloat162float(table[size_t(token)*c.hidden+d]):0.f;
    }
    if(index<size_t(c.batch)*4){
        int b=int(index/4),t=int(index%4);
        slots[index]=t==0?float(state->stepSlots[b]):float(state->slots[b]);
        active[index]=t==0?float(state->advance[b]):0.f;
        for(int axis=0;axis<3;++axis)positions[index*3+axis]=t==0?float(state->positions[b]+stage):0.f;
    }
}
__global__ void Finish(MTPDraftDeviceConfig c,const long long* next,long long* proposals,MTPDraftDeviceState* state){
    if(blockIdx.x||threadIdx.x)return;
    auto& s=*state;if(!ValidConfig(c)){Fail(s,1);return;}
    // j3 is proposal-only: preserve the original last hidden/cache/input buffers.
    PrepareBody(c,3,next,proposals,s);
    if(!s.error&&(s.anyAdvance||s.stage!=3||s.modelCalls<0||s.modelCalls>2))Fail(s,7);
}
}
cudaError_t InitMTPDraftDevice(const MTPDraftDeviceConfig& c,const int* controls,MTPDraftDeviceState* state,long long* proposals,cudaStream_t stream) noexcept {
    if(!ValidConfig(c)||!controls||!state||!proposals)return cudaErrorInvalidValue;
    Initialize<<<1,1,0,stream>>>(c,controls,state,proposals);return cudaPeekAtLastError();
}
cudaError_t PrepareMTPDraftDevice(const MTPDraftDeviceConfig& c,int stage,const long long* next,long long* proposals,
    MTPDraftDeviceState* state,cudaGraphConditionalHandle handle,cudaStream_t stream) noexcept {
    if(!ValidConfig(c)||(stage!=1&&stage!=2)||!next||!proposals||!state)return cudaErrorInvalidValue;
    Prepare<<<1,1,0,stream>>>(c,stage,next,proposals,state,handle);return cudaPeekAtLastError();
}
cudaError_t InputsMTPDraftDevice(const MTPDraftDeviceConfig& c,int stage,const void* table,const long long* proposals,
    const MTPDraftDeviceState* state,const float* last,float* target,float* embedding,float* positions,float* slots,float* active,cudaStream_t stream) noexcept {
    if(!ValidConfig(c)||(stage!=1&&stage!=2)||!table||!proposals||!state||!last||!target||!embedding||!positions||!slots||!active)return cudaErrorInvalidValue;
    size_t count=size_t(c.batch)*4*c.hidden;
    PrepareInputs<<<unsigned((count+255)/256),256,0,stream>>>(c,stage,static_cast<const __nv_bfloat16*>(table),proposals,state,last,target,embedding,positions,slots,active);
    return cudaPeekAtLastError();
}
cudaError_t FinishMTPDraftDevice(const MTPDraftDeviceConfig& c,const long long* next,long long* proposals,MTPDraftDeviceState* state,cudaStream_t stream) noexcept {
    if(!ValidConfig(c)||!next||!proposals||!state)return cudaErrorInvalidValue;
    Finish<<<1,1,0,stream>>>(c,next,proposals,state);return cudaPeekAtLastError();
}
}
