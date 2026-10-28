// SPDX-License-Identifier: Apache-2.0
// Separate reference: unchanged legacy attention math and independent FP32 cache
// storage, with explicit BF16 current-K/V rounding. Not registered or deployed.
#include "qwen38_bf16_kv_float_reference.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
namespace Garnet { namespace {
bool valid(const Qwen35AttentionCoreConfig& c){
    return c.batch==4&&(c.tokens==4||c.tokens==32)&&c.heads==24&&c.kv==4&&c.capacity==4096&&c.dim==256;
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    const auto maximum=(std::numeric_limits<uintptr_t>::max)();
    return !a||!b||x>maximum-n||y>maximum-m||(x<y+m&&y<x+n);
}
__global__ void roundCurrent(const float* key,const float* value,float* roundedKey,float* roundedValue,size_t words){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=2*words)return;
    const size_t j=i<words?i:i-words;
    const float source=i<words?key[j]:value[j];
    const float rounded=__bfloat162float(__float2bfloat16_rn(source));
    if(i<words)roundedKey[j]=rounded;else roundedValue[j]=rounded;
}
}
size_t Qwen38BF16KVFloatReferenceWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    if(!valid(c))return 0;
    const size_t base=Qwen35AttentionCoreWorkspace(c);
    return base+2*size_t(c.batch)*c.tokens*c.kv*c.dim*sizeof(float);
}
int RunQwen38BF16KVFloatReference(const Qwen35AttentionCoreConfig& c,
    const void* const* input,void* const* output,void* workspace,cudaStream_t stream) noexcept {
    if(!valid(c)||!input||!output||!workspace)return 1;
    const size_t query=size_t(c.batch)*c.tokens*c.heads*c.dim*sizeof(float);
    const size_t chunk=size_t(c.batch)*c.tokens*c.kv*c.dim*sizeof(float);
    const size_t cache=size_t(c.batch)*c.kv*c.capacity*c.dim*sizeof(float);
    const size_t control=size_t(c.batch)*c.tokens*sizeof(float);
    const size_t sizes[8]={query,chunk,chunk,query,cache,cache,control,control};
    const size_t outSizes[3]={query,cache,cache};
    const size_t scratch=Qwen38BF16KVFloatReferenceWorkspace(c);
    for(int i=0;i<8;++i){
        if(!input[i]||reinterpret_cast<uintptr_t>(input[i])%alignof(float)||overlap(workspace,scratch,input[i],sizes[i]))return 1;
        for(int j=0;j<3;++j)if(overlap(output[j],outSizes[j],input[i],sizes[i]))return 1;
    }
    if(reinterpret_cast<uintptr_t>(workspace)%alignof(float))return 1;
    for(int j=0;j<3;++j){
        if(!output[j]||reinterpret_cast<uintptr_t>(output[j])%alignof(float)||overlap(output[j],outSizes[j],workspace,scratch))return 1;
        for(int k=j+1;k<3;++k)if(overlap(output[j],outSizes[j],output[k],outSizes[k]))return 1;
    }
    const size_t base=Qwen35AttentionCoreWorkspace(c),words=chunk/sizeof(float);
    auto* roundedKey=reinterpret_cast<float*>(static_cast<unsigned char*>(workspace)+base);
    auto* roundedValue=roundedKey+words;
    roundCurrent<<<unsigned((2*words+255)/256),256,0,stream>>>((const float*)input[1],(const float*)input[2],roundedKey,roundedValue,words);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    const void* roundedInputs[8]={input[0],roundedKey,roundedValue,input[3],input[4],input[5],input[6],input[7]};
    return RunQwen35AttentionCore(c,roundedInputs,output,workspace,stream);
}
}
