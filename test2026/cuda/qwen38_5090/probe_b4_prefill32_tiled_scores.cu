// SPDX-License-Identifier: Apache-2.0
// Standalone synthetic component test. No checkpoint, DLL or serving benchmark.
#include "qwen38_bf16_prefill32_tiled_scores.cuh"
#include "qwen38_bf16_kv_commit.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Buffer {
 void* data=nullptr;size_t bytes;
 explicit Buffer(size_t n):bytes(n){check(cudaMalloc(&data,n));}
 ~Buffer(){if(data)cudaFree(data);}
 Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
struct Stream {cudaStream_t value;Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(value);}};
struct Event {cudaEvent_t value;Event(){check(cudaEventCreate(&value));}~Event(){cudaEventDestroy(value);}};
__global__ void fill(float* p,size_t count,unsigned seed,unsigned multiplier){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
  p[i]=i%97==0?-0.f:float(int((unsigned(i)*multiplier+seed)%257)-128)/256.f;
}
__global__ void fillKV(float* canonical,int64_t keyOffset,int64_t valueOffset,unsigned seed){
 const size_t stride=4ULL*4096*256,total=4*stride;
 auto* key=reinterpret_cast<__nv_bfloat16*>(canonical+keyOffset);auto* value=reinterpret_cast<__nv_bfloat16*>(canonical+valueOffset);
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x){
  const size_t b=i/stride,k=i%stride,offset=b*size_t(Qwen38BF16KVCanonicalWidth)*2+k;
  key[offset]=__float2bfloat16_rn(float(int((unsigned(i)*11+seed)%127)-63)/64.f);
  value[offset]=__float2bfloat16_rn(float(int((unsigned(i)*7+seed)%127)-63)/64.f);
 }
}
__global__ void compareWords(const uint32_t* a,const uint32_t* b,size_t count,unsigned long long* mismatches){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
  if(a[i]!=b[i])atomicAdd(mismatches,1ULL);
}
void save(const std::filesystem::path& p,const std::vector<float>& data){
 require(!std::filesystem::exists(p),"Preserve existing raw output");std::ofstream f(p,std::ios::binary);
 f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(float)));require(bool(f),"Raw output write failed");
}
}
int main(int argc,char** argv){
 try{
  static_assert(sizeof(float)==4,"FP32 output required");require(argc==2,"Output directory required");
  const std::filesystem::path out(argv[1]);require(std::filesystem::is_directory(out),"Owned output directory must exist");
  int count=0;check(cudaGetDeviceCount(&count));require(count==1,"Exactly one visible device required");
  cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));require(prop.major==12&&prop.minor==0,"SM120 required");
  int runtime=0,driver=0;check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
  std::cout<<std::setprecision(9)<<"INFO\tSM120\t"<<runtime<<'\t'<<driver<<std::endl;
  Stream stream;const auto s=stream.value;Qwen35AttentionCoreConfig c{};
  c.batch=4;c.tokens=32;c.heads=24;c.kv=4;c.capacity=4096;c.dim=256;
  const size_t words=4ULL*32*24*256,small=4ULL*32*4*256,controls=4*32;
  Buffer q(words*4),k(small*4),v(small*4),gate(words*4),state(4ULL*Qwen38BF16KVCanonicalWidth*4),slots(controls*4),active(controls*4);
  std::array<Buffer*,7> buffers{&q,&k,&v,&gate,&state,&slots,&active};
  std::vector<Buffer*> backups;for(auto* b:buffers)backups.push_back(new Buffer(b->bytes));
  struct Release {std::vector<Buffer*>& values;~Release(){for(auto* v:values)delete v;}} release{backups};
  std::array<const void*,7> inputs{};for(size_t i=0;i<7;++i)inputs[i]=buffers[i]->data;
  const size_t workspaceBytes=Qwen38BF16KVOverlayWorkspace(c);require(workspaceBytes==Qwen38BF16KVPrefill32TiledScoresWorkspace(c)&&workspaceBytes>0,"Workspace contract differs");
  Buffer workspace(workspaceBytes),oldOutput(words*4),newOutput(words*4),mismatches(sizeof(unsigned long long));
  check(cudaMemsetAsync(state.data,0,state.bytes,s));
  std::array<int,4> prefixes{0,1024,3936,4064},layers{0,7,8,15};
  std::array<std::array<int,4>,4> counts{{{32,17,1,0},{32,32,32,32},{32,32,32,32},{32,20,31,2}}};
  for(int index=0;index<4;++index){
   fill<<<4096,256,0,s>>>(static_cast<float*>(q.data),words,index+3,37);
   fill<<<4096,256,0,s>>>(static_cast<float*>(k.data),small,index+7,11);
   fill<<<4096,256,0,s>>>(static_cast<float*>(v.data),small,index+11,7);
   fill<<<4096,256,0,s>>>(static_cast<float*>(gate.data),words,index+13,13);
   const int64_t keyOffset=2482176LL+6676480LL*layers[index],valueOffset=keyOffset+2097152;
   fillKV<<<4096,256,0,s>>>(static_cast<float*>(state.data),keyOffset,valueOffset,index+19);check(cudaPeekAtLastError());
   std::array<float,128> hostSlots{},hostActive{};
   for(int b=0;b<4;++b)for(int t=0;t<32;++t){hostActive[b*32+t]=t<counts[index][b]?1.f:0.f;hostSlots[b*32+t]=t<counts[index][b]?float(prefixes[index]+t):4096.f;}
   check(cudaMemcpyAsync(slots.data,hostSlots.data(),slots.bytes,cudaMemcpyHostToDevice,s));
   check(cudaMemcpyAsync(active.data,hostActive.data(),active.bytes,cudaMemcpyHostToDevice,s));
   for(size_t i=0;i<7;++i)check(cudaMemcpyAsync(backups[i]->data,buffers[i]->data,buffers[i]->bytes,cudaMemcpyDeviceToDevice,s));
   check(cudaStreamSynchronize(s));
   auto run=[&](int strategy){
    const int result=strategy?RunQwen38BF16KVPrefill32TiledScores(c,inputs.data(),newOutput.data,workspace.data,keyOffset,valueOffset,s)
                             :RunQwen38BF16KVOverlay(c,inputs.data(),oldOutput.data,workspace.data,keyOffset,valueOffset,s);
    require(result==0,"Attention component launch failed");
   };
   for(int strategy=0;strategy<2;++strategy){
    run(strategy);check(cudaMemsetAsync(mismatches.data,0,mismatches.bytes,s));
    for(size_t i=0;i<7;++i)compareWords<<<4096,256,0,s>>>(static_cast<const uint32_t*>(buffers[i]->data),static_cast<const uint32_t*>(backups[i]->data),buffers[i]->bytes/4,static_cast<unsigned long long*>(mismatches.data));
    unsigned long long changed=0;check(cudaMemcpyAsync(&changed,mismatches.data,sizeof(changed),cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));require(changed==0,"Readonly attention input mutated");
   }
   std::vector<float> reference(words),candidate(words);
   check(cudaMemcpyAsync(reference.data(),oldOutput.data,oldOutput.bytes,cudaMemcpyDeviceToHost,s));
   check(cudaMemcpyAsync(candidate.data(),newOutput.data,newOutput.bytes,cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));
   save(out/("case-"+std::to_string(index)+"-reference.f32"),reference);save(out/("case-"+std::to_string(index)+"-candidate.f32"),candidate);
   require(std::memcmp(reference.data(),candidate.data(),words*4)==0,"Native output bytes differ; preserve raw outputs");
   for(float x:reference)require(std::isfinite(x),"Nonfinite native reference output");
   std::cout<<"CASE\t"<<index<<'\t'<<prefixes[index]<<'\t'<<3+4*layers[index]<<'\t';
   for(int b=0;b<4;++b)std::cout<<(b?",":"")<<counts[index][b];
   std::cout<<'\t'<<words<<"\t0\t0"<<std::endl;
   for(int warm=0;warm<2;++warm){run(0);run(1);}check(cudaStreamSynchronize(s));
   for(int trial=0;trial<3;++trial)for(int order=0;order<2;++order){
    const int strategy=(trial+index+order)%2;Event begin,end;check(cudaEventRecord(begin.value,s));
    for(int rep=0;rep<10;++rep)run(strategy);
    check(cudaEventRecord(end.value,s));check(cudaEventSynchronize(end.value));float ms=0;check(cudaEventElapsedTime(&ms,begin.value,end.value));
    require(std::isfinite(ms)&&ms>0,"Invalid CUDA-event elapsed time");
    std::cout<<"TIMING\t"<<index<<'\t'<<trial<<'\t'<<strategy<<"\t10\t"<<ms<<std::endl;
   }
   auto invalid=c;invalid.tokens=4;require(Qwen38BF16KVPrefill32TiledScoresWorkspace(invalid)==0,"Unqualified T4 accepted");
   require(RunQwen38BF16KVPrefill32TiledScores(invalid,inputs.data(),newOutput.data,workspace.data,keyOffset,valueOffset,s)!=0,"Unqualified T4 launch accepted");
   invalid=c;invalid.batch=8;require(Qwen38BF16KVPrefill32TiledScoresWorkspace(invalid)==0,"Unqualified B8 accepted");
   require(RunQwen38BF16KVPrefill32TiledScores(c,inputs.data(),q.data,workspace.data,keyOffset,valueOffset,s)!=0,"Output/input alias accepted");
   require(RunQwen38BF16KVPrefill32TiledScores(c,inputs.data(),newOutput.data,workspace.data,0,0,s)!=0,"Invalid layout offsets accepted");
   check(cudaMemsetAsync(mismatches.data,0,mismatches.bytes,s));
   for(size_t i=0;i<7;++i)compareWords<<<4096,256,0,s>>>(static_cast<const uint32_t*>(buffers[i]->data),static_cast<const uint32_t*>(backups[i]->data),buffers[i]->bytes/4,static_cast<unsigned long long*>(mismatches.data));
   compareWords<<<4096,256,0,s>>>(static_cast<const uint32_t*>(oldOutput.data),static_cast<const uint32_t*>(newOutput.data),words,static_cast<unsigned long long*>(mismatches.data));
   unsigned long long changed=0;check(cudaMemcpyAsync(&changed,mismatches.data,sizeof(changed),cudaMemcpyDeviceToHost,s));check(cudaStreamSynchronize(s));require(changed==0,"Rejected call mutated inputs/output");
   std::cout<<"CONTROL\t"<<index<<"\treadonly_all7_inputs_full_words\tinvalid_no_mutation"<<std::endl;
  }
  std::cout<<"COMPLETE\t4\t24\t4"<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<"COMPONENT_FAILED "<<e.what()<<std::endl;return 1;}
}
