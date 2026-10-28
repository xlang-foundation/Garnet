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
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("Strided canonical correctness requires one raw output prefix");
 const bool timing=false;
 std::vector<Case> cases={{4,4,24,4,4096,256,32,0},{4,4,24,4,4096,256,128,0},{4,4,24,4,4096,256,512,0},{4,4,24,4,4096,256,3968,0},{4,4,24,4,4096,256,4096,0},{4,4,24,4,4096,256,128,1},{4,4,24,4,4096,256,128,2},{4,4,24,4,4096,256,128,3},{4,4,24,4,4096,256,511,0},{4,4,24,4,4096,256,513,0},{4,4,24,4,4096,256,1024,0},{4,4,24,4,4096,256,2048,0}};
 cudaStream_t stream;Check(cudaStreamCreate(&stream));cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
 for(size_t ci=0;ci<cases.size();++ci){const auto a=cases[ci];Garnet::Qwen35AttentionCoreConfig c{a.batch,a.tokens,a.heads,a.kv,a.capacity,a.dim};
  size_t qsize=size_t(c.batch)*c.tokens*c.heads*c.dim,kvsize=size_t(c.batch)*c.tokens*c.kv*c.dim,cache=size_t(c.batch)*c.kv*c.capacity*c.dim;
  std::vector<std::vector<float>> host(8);host[0].resize(qsize);host[1].resize(kvsize);host[2].resize(kvsize);host[3].resize(qsize);host[4].resize(cache);host[5].resize(cache);host[6].resize(c.batch*c.tokens);host[7].resize(c.batch*c.tokens);
  for(int i=0;i<6;++i)for(size_t j=0;j<host[i].size();++j)host[i][j]=Value(j,11+i*19)*(i==3?16.f:1.f);
  auto isActive=[&](int b,int t){return a.pattern==0 || (a.pattern==1 && b!=3 && t%2==0) || (a.pattern==3 && (b==0 || (b==1 && t==0) || (b==2 && t%2==0)));};
  for(int b=0;b<c.batch;++b){
   int count=0;for(int t=0;t<c.tokens;++t)count+=isActive(b,t)?1:0;
   const int mixedPrefixes[4]={32,128,512,3968};
   int prefix=a.pattern==3?mixedPrefixes[b]:a.prefix,slot=std::min(c.capacity-std::max(count,1),prefix-1);
   for(int t=0;t<c.tokens;++t){bool active=isActive(b,t);host[6][b*c.tokens+t]=float(std::min(slot,c.capacity-1));host[7][b*c.tokens+t]=active?1.f:0.f;if(active)++slot;}
  }
  printf("B4_ATTENTION_CONTROLS case=%zu pattern=%d slots=",ci,a.pattern);
  for(float v:host[6])printf("%.0f,",v);printf(" active=");for(float v:host[7])printf("%.0f,",v);printf("\n");fflush(stdout);
  auto nextKey=host[4],nextValue=host[5];
  for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(host[7][b*c.tokens+t])for(int h=0;h<c.kv;++h)for(int d=0;d<c.dim;++d){size_t from=((size_t(b)*c.tokens+t)*c.kv+h)*c.dim+d,to=((size_t(b)*c.kv+h)*c.capacity+int(host[6][b*c.tokens+t]))*c.dim+d;nextKey[to]=host[1][from];nextValue[to]=host[2][from];}
  std::vector<float> expected(qsize,0);
  for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(host[7][b*c.tokens+t])for(int h=0;h<c.heads;++h){
   int end=int(host[6][b*c.tokens+t])+1,kh=h/(c.heads/c.kv);size_t row=(size_t(b)*c.tokens+t)*c.heads+h;std::vector<double> scores(end);double maximum=-std::numeric_limits<double>::infinity(),total=0;
   for(int p=0;p<end;++p){double sum=0;size_t base=((size_t(b)*c.kv+kh)*c.capacity+p)*c.dim;for(int d=0;d<c.dim;++d)sum+=double(host[0][row*c.dim+d])*nextKey[base+d];scores[p]=sum/std::sqrt(double(c.dim));maximum=std::max(maximum,scores[p]);}
   for(double& s:scores){s=std::exp(s-maximum);total+=s;}
   for(int d=0;d<c.dim;++d){double sum=0;for(int p=0;p<end;++p)sum+=(scores[p]/total)*nextValue[((size_t(b)*c.kv+kh)*c.capacity+p)*c.dim+d];expected[row*c.dim+d]=float(sum/(1+std::exp(-double(host[3][row*c.dim+d]))));}
  }
  Save(std::string(argv[1])+"-case"+std::to_string(ci)+"-cpu.f32",expected);
  std::vector<std::unique_ptr<Buffer>> inputs;const void* in[8];for(int i=0;i<8;++i){inputs.emplace_back(new Buffer(host[i].size()));inputs.back()->upload(host[i]);in[i]=inputs.back()->data();}
  std::vector<std::unique_ptr<Buffer>> outputs;void* out[3];for(size_t n:{qsize,cache,cache})outputs.emplace_back(new Buffer(n));for(int i=0;i<3;++i)out[i]=outputs[i]->data();Buffer workspace(Garnet::Qwen35AttentionCoreWorkspace(c)/4),splitWorkspace(Garnet::Qwen35AttentionSplitWorkspace(c)/4);
  const size_t which=ci%2;
  CanonicalFixture canonical(kCanonicalStride,kCanonicalKeyOffsets[which],kCanonicalValueOffsets[which],size_t(c.kv)*c.capacity*c.dim);canonical.initialize(host[4],host[5]);
  const void* stridedIn[8];for(int i=0;i<8;++i)stridedIn[i]=in[i];stridedIn[4]=canonical.key();stridedIn[5]=canonical.value();
  printf("STRIDED_CANONICAL case=%zu stride=%zu layer=%d key_offset=%zu value_offset=%zu\n",ci,kCanonicalStride,kCanonicalLayers[which],kCanonicalKeyOffsets[which],kCanonicalValueOffsets[which]);fflush(stdout);
  Buffer accepted(c.batch);std::vector<float> fullCounts(c.batch,0);for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)fullCounts[b]+=host[7][b*c.tokens+t]!=0;accepted.upload(fullCounts);
  if(Garnet::RunQwen35AttentionKVOverlay(c,in,out[0],stream))throw std::runtime_error("Read-only overlay enqueue failed");Check(cudaStreamSynchronize(stream));
  if(!Equal(inputs[4]->read(),host[4])||!Equal(inputs[5]->read(),host[5]))throw std::runtime_error("Overlay mutated canonical KV before commit");
  const auto readonlyReference=outputs[0]->read();
  for(int strategy=0;strategy<3;++strategy){
   if(Garnet::RunQwen35AttentionKVOverlayCached(c,in,out[0],workspace.data(),strategy,stream))throw std::runtime_error("Cached readonly overlay failed");Check(cudaStreamSynchronize(stream));
   if(!Equal(outputs[0]->read(),readonlyReference)||!Equal(inputs[4]->read(),host[4])||!Equal(inputs[5]->read(),host[5]))throw std::runtime_error("Cached overlay differs or mutates canonical KV");
  }
  for(int strategy=0;strategy<3;++strategy){
   if(Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),strategy,kCanonicalStride,stream))throw std::runtime_error("Strided readonly overlay failed");Check(cudaStreamSynchronize(stream));
   if(!Equal(outputs[0]->read(),readonlyReference)||!Equal(canonical.read(false),host[4])||!Equal(canonical.read(true),host[5]))throw std::runtime_error("Strided readonly output differs or mutates KV");canonical.padding(stream);
  }
  Graph prefixGraph;
  Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  if(Garnet::CommitQwen35AttentionKVPrefixStrided(c,stridedIn,accepted.data(),canonical.key(),canonical.value(),kCanonicalStride,stream))throw std::runtime_error("Strided prefix graph capture failed");
  Check(cudaStreamEndCapture(stream,&prefixGraph.graph));Check(cudaGraphInstantiate(&prefixGraph.executable,prefixGraph.graph,nullptr,nullptr,0));
  for(int trial=0;trial<3;++trial){
   canonical.initialize(host[4],host[5]);
   inputs[4]->upload(host[4]);inputs[5]->upload(host[5]);std::vector<float> counts(c.batch);const int mixed[4]={4,0,3,1};
   for(int b=0;b<c.batch;++b)counts[b]=trial==0?0:std::min(int(fullCounts[b]),trial==1?b:mixed[b]);accepted.upload(counts);
   if(Garnet::CommitQwen35AttentionKVPrefix(c,in,accepted.data(),inputs[4]->data(),inputs[5]->data(),stream))throw std::runtime_error("Prefix commit enqueue failed");Check(cudaStreamSynchronize(stream));
   auto expectedKey=host[4],expectedValue=host[5];
   for(int b=0;b<c.batch;++b){int rank=0;for(int t=0;t<c.tokens;++t)if(host[7][b*c.tokens+t]){const bool take=rank++<int(counts[b]);if(take)for(int h=0;h<c.kv;++h)for(int d=0;d<c.dim;++d){size_t from=((size_t(b)*c.tokens+t)*c.kv+h)*c.dim+d,to=((size_t(b)*c.kv+h)*c.capacity+int(host[6][b*c.tokens+t]))*c.dim+d;expectedKey[to]=host[1][from];expectedValue[to]=host[2][from];}}}
   if(!Equal(inputs[4]->read(),expectedKey)||!Equal(inputs[5]->read(),expectedValue))throw std::runtime_error("Zero/partial prefix full KV mismatch");
   if(Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),trial,kCanonicalStride,stream))throw std::runtime_error("Strided prefix target failed");Check(cudaStreamSynchronize(stream));
   if(!Equal(outputs[0]->read(),readonlyReference))throw std::runtime_error("Strided partial target context differs");
   if(Garnet::CommitQwen35AttentionKVPrefixStrided(c,stridedIn,accepted.data(),canonical.key(),canonical.value(),kCanonicalStride,stream))throw std::runtime_error("Strided prefix commit failed");Check(cudaStreamSynchronize(stream));
   if(!Equal(canonical.read(false),expectedKey)||!Equal(canonical.read(true),expectedValue))throw std::runtime_error("Strided zero/partial full KV mismatch");canonical.padding(stream);
   canonical.initialize(host[4],host[5]);Check(cudaGraphLaunch(prefixGraph.executable,stream));Check(cudaStreamSynchronize(stream));
   if(!Equal(canonical.read(false),expectedKey)||!Equal(canonical.read(true),expectedValue))throw std::runtime_error("Captured dynamic-prefix full KV mismatch");canonical.padding(stream);
   Save(std::string(argv[1])+"-case"+std::to_string(ci)+"-commit"+std::to_string(trial)+"-counts.f32",counts);
  }
  inputs[4]->upload(host[4]);inputs[5]->upload(host[5]);accepted.upload(fullCounts);canonical.initialize(host[4],host[5]);
  auto run=[&](int mode){if(mode>=8){if(Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),mode-8,kCanonicalStride,stream))return 1;return Garnet::CommitQwen35AttentionKVPrefixStrided(c,stridedIn,accepted.data(),canonical.key(),canonical.value(),kCanonicalStride,stream);}if(mode>=4){int code=mode==4?Garnet::RunQwen35AttentionKVOverlay(c,in,out[0],stream):Garnet::RunQwen35AttentionKVOverlayCached(c,in,out[0],workspace.data(),mode-5,stream);if(code)return code;return Garnet::CommitQwen35AttentionKVPrefix(c,in,accepted.data(),inputs[4]->data(),inputs[5]->data(),stream);}return mode==3?Garnet::RunQwen35AttentionSplit(c,in,out,splitWorkspace.data(),stream):mode==2?Garnet::RunQwen35AttentionCompactTiled(c,in,out,nullptr,stream):mode==1?Garnet::RunQwen35AttentionCompact(c,in,out,nullptr,stream):Garnet::RunQwen35AttentionCore(c,in,out,workspace.data(),stream);};
  auto readKey=[&](int mode){return mode>=8?canonical.read(false):(mode>=4?inputs[4]:outputs[1])->read();};
  auto readValue=[&](int mode){return mode>=8?canonical.read(true):(mode>=4?inputs[5]:outputs[2])->read();};
  Graph graphs[11];std::vector<std::vector<float>> witnessed(11);
  const char* variants[11]={"baseline","compact","tiled","split","overlay","cached-compact","cached-legacy","hybrid","strided-compact","strided-legacy","strided-hybrid"};std::vector<float> baseline;float maxError=0,splitDelta=0;
  for(int mode=0;mode<11;++mode){
   if(run(mode))throw std::runtime_error("Direct attention enqueue failed");Check(cudaStreamSynchronize(stream));auto actual=outputs[0]->read();witnessed[mode]=actual;
   Save(std::string(argv[1])+"-case"+std::to_string(ci)+("-"+std::string(variants[mode])+".f32"),actual);
   for(size_t i=0;i<actual.size();++i){float error=std::fabs(actual[i]-expected[i]);maxError=std::max(maxError,error);if(!std::isfinite(actual[i])||error>2e-5f+2e-4f*std::fabs(expected[i])){fprintf(stderr,"case=%zu mode=%d index=%zu actual=%g expected=%g error=%g\n",ci,mode,i,actual[i],expected[i],error);throw std::runtime_error("Independent CPU attention oracle failed");}}
   if(mode==0)baseline=actual;
   else if(mode<3||mode>=4){if(!Equal(actual,baseline))throw std::runtime_error("Compact differs bitwise from baseline");}
   else for(size_t i=0;i<actual.size();++i){float error=std::fabs(actual[i]-baseline[i]);splitDelta=std::max(splitDelta,error);if(error>2e-5f+2e-4f*std::fabs(baseline[i]))throw std::runtime_error("Split exceeds predeclared legacy numerical tolerance");}
   if(!Equal(readKey(mode),nextKey)||!Equal(readValue(mode),nextValue))throw std::runtime_error("Full KV output mismatch");
   Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(run(mode))throw std::runtime_error("Capture failed");Check(cudaStreamEndCapture(stream,&graphs[mode].graph));Check(cudaGraphInstantiate(&graphs[mode].executable,graphs[mode].graph,nullptr,nullptr,0));
   for(int replay=0;replay<3;++replay){Check(cudaGraphLaunch(graphs[mode].executable,stream));Check(cudaStreamSynchronize(stream));if(!Equal(outputs[0]->read(),actual)||!Equal(readKey(mode),nextKey)||!Equal(readValue(mode),nextValue))throw std::runtime_error("Graph replay output mismatch");}
  }
  for(int i=0;i<8;++i){inputs[i]->guards();if(!Equal(inputs[i]->read(),i==4?nextKey:i==5?nextValue:host[i]))throw std::runtime_error("Unexpected input mutation");}
  if(!Equal(canonical.read(false),nextKey)||!Equal(canonical.read(true),nextValue))throw std::runtime_error("Strided full committed KV mismatch");canonical.padding(stream);for(auto& p:outputs)p->guards();workspace.guards();splitWorkspace.guards();accepted.guards();
  const auto beforeInvalid=outputs[0]->read();auto invalid=c;invalid.dim=0;
  if(Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,canonical.key(),workspace.data(),0,kCanonicalStride,stream)==0||Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),0,size_t(2147483648ULL),stream)==0||Garnet::CommitQwen35AttentionKVPrefixStrided(c,stridedIn,accepted.data(),canonical.key(),canonical.value(),size_t(2147483648ULL),stream)==0||Garnet::RunQwen35AttentionKVOverlayStrided(invalid,stridedIn,out[0],workspace.data(),0,kCanonicalStride,stream)==0||Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),3,kCanonicalStride,stream)==0||Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),0,0,stream)==0||Garnet::RunQwen35AttentionKVOverlayStrided(c,stridedIn,out[0],workspace.data(),0,size_t(c.kv)*c.capacity*c.dim-1,stream)==0||Garnet::CommitQwen35AttentionKVPrefixStrided(c,stridedIn,accepted.data(),canonical.key(),canonical.value(),0,stream)==0||Garnet::RunQwen35AttentionKVOverlayCached(invalid,in,out[0],workspace.data(),0,stream)==0||Garnet::RunQwen35AttentionKVOverlayCached(c,in,out[0],workspace.data(),3,stream)==0||Garnet::RunQwen35AttentionKVOverlay(invalid,in,out[0],stream)==0||Garnet::CommitQwen35AttentionKVPrefix(invalid,in,accepted.data(),inputs[4]->data(),inputs[5]->data(),stream)==0||Garnet::RunQwen35AttentionCompact(invalid,in,out,nullptr,stream)==0||Garnet::RunQwen35AttentionCompactTiled(invalid,in,out,nullptr,stream)==0||Garnet::Qwen35AttentionSplitWorkspace(invalid)!=0||Garnet::RunQwen35AttentionSplit(invalid,in,out,splitWorkspace.data(),stream)==0||!Equal(outputs[0]->read(),beforeInvalid))throw std::runtime_error("Invalid host profile accepted or changed output");
  if(!Equal(canonical.read(false),nextKey)||!Equal(canonical.read(true),nextValue))throw std::runtime_error("Rejected controls changed canonical KV");canonical.padding(stream);
  if(!timing){printf("{\"case\":%zu,\"status\":\"correctness_pass\",\"variants_checked\":11,\"strided_readonly_exact\":true,\"strided_prefix_trials\":3,\"strided_prefix_graph_trials\":3,\"canonical_padding_preserved\":true,\"cached_readonly_legacy_exact\":true,\"overlay_legacy_bit_exact\":true,\"readonly_overlay_preserved_kv\":true,\"zero_partial_prefix_trials\":3,\"cpu_max_abs\":%.9g,\"row_tiled_bit_exact\":true,\"split_within_tolerance\":true,\"full_kv_bit_exact\":true,\"graph_replays\":3}\n",ci,maxError);fflush(stdout);continue;}



 }
 Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));Check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"COMPACT_ATTENTION_FAILURE %s\n",e.what());return 1;}}
