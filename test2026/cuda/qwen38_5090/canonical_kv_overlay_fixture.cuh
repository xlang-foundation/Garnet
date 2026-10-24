// SPDX-License-Identifier: Apache-2.0
// Independent CPU oracle + full-cache checks before component-only A/B timing.
#include "qwen35_attention_core.cuh"
#include "attention_kv_delta_core.cuh"
#include "attention_kv_delta_cached_core.cuh"
#include "attention_kv_delta_strided_core.cuh"
#include "attention_kv_canonical_layout.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
 float* base=nullptr;size_t count;static constexpr size_t guard=64;
 explicit Buffer(size_t n):count(n){Check(cudaMalloc(reinterpret_cast<void**>(&base),(n+2*guard)*4));std::vector<float> v(n+2*guard,1234567.f);Check(cudaMemcpy(base,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
 ~Buffer(){if(base)cudaFree(base);}float* data(){return base+guard;}
 void upload(const std::vector<float>& v){if(v.size()!=count)throw std::runtime_error("Input size mismatch");Check(cudaMemcpy(data(),v.data(),count*4,cudaMemcpyHostToDevice));}
 std::vector<float> read(){std::vector<float> v(count);Check(cudaMemcpy(v.data(),data(),count*4,cudaMemcpyDeviceToHost));return v;}
 void guards(){float v[128];Check(cudaMemcpy(v,base,256,cudaMemcpyDeviceToHost));Check(cudaMemcpy(v+64,data()+count,256,cudaMemcpyDeviceToHost));for(float x:v)if(x!=1234567.f)throw std::runtime_error("Buffer overrun");}
};
struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}};
bool Equal(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
void Save(const std::string& p,const std::vector<float>& v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);if(!f)throw std::runtime_error("Cannot retain raw context");}
float Value(size_t i,int salt){return (int((i*17+salt)%97)-48)/256.f;}
// One actual-sized canonical state allocation, with canaries in every unrelated
// word. Only the selected layer K/V windows are initialized with fixture data.
__global__ void checkCanonicalPadding(const unsigned* base,size_t stride,size_t keyOffset,size_t valueOffset,size_t cacheRow,unsigned* failed){
 const size_t guard=64,b=blockIdx.y;
 for(size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;p<stride;p+=size_t(gridDim.x)*blockDim.x){
  if((p>=keyOffset&&p<keyOffset+cacheRow)||(p>=valueOffset&&p<valueOffset+cacheRow))continue;
  if(base[guard+b*stride+p]!=0x42424242u)atomicExch(failed,1u);
 }
 if(b==0&&blockIdx.x==0&&threadIdx.x<guard){
  if(base[threadIdx.x]!=0x42424242u||base[guard+4*stride+threadIdx.x]!=0x42424242u)atomicExch(failed,1u);
 }
}
struct CanonicalFixture {
 float* base=nullptr;unsigned* failed=nullptr;size_t stride,keyOffset,valueOffset,cacheRow;static constexpr size_t guard=64;
 CanonicalFixture(size_t s,size_t k,size_t v,size_t row):stride(s),keyOffset(k),valueOffset(v),cacheRow(row){
  if(k+row>s||v+row>s||!(k+row<=v||v+row<=k))throw std::runtime_error("Invalid canonical fixture spans");
  Check(cudaMalloc(reinterpret_cast<void**>(&base),(4*stride+2*guard)*4));Check(cudaMalloc(reinterpret_cast<void**>(&failed),4));
 }
 ~CanonicalFixture(){if(failed)cudaFree(failed);if(base)cudaFree(base);}
 float* key(){return base+guard+keyOffset;}float* value(){return base+guard+valueOffset;}
 void initialize(const std::vector<float>& k,const std::vector<float>& v){
  if(k.size()!=4*cacheRow||v.size()!=4*cacheRow)throw std::runtime_error("Canonical fixture input size mismatch");
  Check(cudaMemset(base,0x42,(4*stride+2*guard)*4));
  Check(cudaMemcpy2D(key(),stride*4,k.data(),cacheRow*4,cacheRow*4,4,cudaMemcpyHostToDevice));
  Check(cudaMemcpy2D(value(),stride*4,v.data(),cacheRow*4,cacheRow*4,4,cudaMemcpyHostToDevice));Check(cudaDeviceSynchronize());
 }
 std::vector<float> read(bool values){std::vector<float> result(4*cacheRow);Check(cudaMemcpy2D(result.data(),cacheRow*4,values?value():key(),stride*4,cacheRow*4,4,cudaMemcpyDeviceToHost));return result;}
 void padding(cudaStream_t stream){
  Check(cudaMemsetAsync(failed,0,4,stream));checkCanonicalPadding<<<dim3(4096,4),256,0,stream>>>((const unsigned*)base,stride,keyOffset,valueOffset,cacheRow,failed);Check(cudaGetLastError());Check(cudaStreamSynchronize(stream));
  unsigned error=0;Check(cudaMemcpy(&error,failed,4,cudaMemcpyDeviceToHost));if(error)throw std::runtime_error("Unrelated canonical state or guard mutated");
 }
};

struct Case {int batch,tokens,heads,kv,capacity,dim,prefix,pattern;};
