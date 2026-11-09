// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace Garnet {struct MetalCommittedAttentionReference {
 static void Run(MetalContext& gpu,const std::vector<MetalMatrix>& args,const MetalMatrix& out,size_t capacity,size_t chunk,bool gated){gpu.EncodeCommittedAttentionChunk(args,out,32,8,128,capacity,chunk,gated);}
};}
namespace {
void Need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
float Value(size_t i,uint32_t salt){uint32_t x=uint32_t(i)^salt;x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return float(int(x%20001)-10000)*.00003f;}
struct Fixture {
 MetalContext gpu;size_t batch,chunk,cap,first,rows;bool gated;std::vector<float> q,k,v,kc,vc,slots,active,gate;
 std::vector<MetalMatrix> args;MetalMatrix original,candidate,copied,zero;
 size_t oracle=0,words=0;double fraction=0;
 Fixture(size_t b,size_t t,size_t c,size_t f,bool g,bool ragged):batch(b),chunk(t),cap(c),first(f),rows(b*t),gated(g){
  auto data=[](size_t n,uint32_t salt){std::vector<float> a(n);for(size_t i=0;i<n;++i)a[i]=Value(i,salt);return a;};
  q=data(rows*4096,17);k=data(rows*1024,31);v=data(rows*1024,71);kc=data(batch*8*cap*128,101);vc=data(kc.size(),151);gate=data(q.size(),191);
  slots.resize(rows);active.assign(rows,1);for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)slots[b*chunk+t]=float(first+t);
  if(ragged){for(size_t t=chunk-3;t<chunk;++t)active[chunk+t]=0;for(size_t t=1;t<chunk;++t)active[2*chunk+t]=0;for(size_t t=0;t<chunk;++t)active[3*chunk+t]=0;}
  args={gpu.Upload(q,rows,4096),gpu.Upload(k,rows,1024),gpu.Upload(v,rows,1024),gpu.Upload(kc,batch,8*cap*128),gpu.Upload(vc,batch,8*cap*128),gpu.Upload(slots,batch,chunk),gpu.Upload(active,batch,chunk),gpu.Upload(gate,rows,4096)};
  original=gpu.Allocate(rows,4096);candidate=gpu.Allocate(rows,4096);copied=gpu.Allocate(batch,chunk);zero=gpu.Upload(std::vector<float>(rows,0),batch,chunk);
  gpu.PrepareAttentionChunk(args,32,8,128,cap,chunk);
 }
 void Commit(){for(size_t part=0;part<2;++part)gpu.EncodeKVChunk(args[part+1],args[part+3],args[5],args[6],args[part+3],8,128,cap,chunk,true);}
 void Oracle(const std::vector<float>& got){
  for(size_t b=0;b<batch;++b){size_t count=0;while(count<chunk&&active[b*chunk+count])++count;if(!count)continue;
   for(size_t t:std::vector<size_t>{0,count/2,count-1})for(size_t h:{size_t(0),size_t(7),size_t(31)}){
    size_t visible=first+t+1,kh=h/4;std::vector<double> p(visible);double mx=-INFINITY;
    for(size_t s=0;s<visible;++s){double dot=0;for(size_t d=0;d<128;++d){float key=s<first?kc[((b*8+kh)*cap+s)*128+d]:k[((b*chunk+s-first)*8+kh)*128+d];dot+=double(q[((b*chunk+t)*32+h)*128+d])*key;}p[s]=dot/std::sqrt(128.);mx=std::max(mx,p[s]);}
    double sum=0;for(auto& x:p){x=std::exp(x-mx);sum+=x;}
    for(size_t d:{size_t(0),size_t(63),size_t(127)}){double expected=0;for(size_t s=0;s<visible;++s){float val=s<first?vc[((b*8+kh)*cap+s)*128+d]:v[((b*chunk+s-first)*8+kh)*128+d];expected+=p[s]/sum*val;}
     size_t at=((b*chunk+t)*32+h)*128+d;if(gated)expected/=1+std::exp(-double(gate[at]));double f=std::abs(double(got[at])-expected)/(1e-5+2e-4*std::abs(expected));Need(f<=1,"Native committed attention violates original FP64 bound");fraction=std::max(fraction,f);++oracle;}
   }
  }
 }
 void Run(){size_t plans=gpu.ChunkAttentionMPSPlanCount(),scratch=gpu.ChunkAttentionMPSScratchBytes(),activeOwners=0,total=0;
  for(size_t b=0;b<batch;++b){if(active[b*chunk])++activeOwners;for(size_t t=0;t<chunk;++t)total+=size_t(active[b*chunk+t]!=0);}
  std::vector<float> expectedK=kc,expectedV=vc;
  for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)if(active[b*chunk+t])for(size_t h=0;h<8;++h)for(size_t d=0;d<128;++d){size_t out=((b*8+h)*cap+first+t)*128+d,in=((b*chunk+t)*8+h)*128+d;expectedK[out]=k[in];expectedV[out]=v[in];}
  gpu.Begin();gpu.EncodeAttentionChunk(args,original,32,8,128,cap,chunk,gated);Commit();gpu.SubmitAndWait();Need(gpu.LastCommittedKVAttentionDispatches()==0,"Generic attention changed policy");
  const auto old=gpu.Download(original);for(size_t part=0;part<2;++part){auto a=gpu.Download(args[part+3]);const auto& expected=part?expectedV:expectedK;Need(a.size()==expected.size()&&!std::memcmp(a.data(),expected.data(),a.size()*4),"Original committed KV differs");}
  gpu.Update(args[3],kc);gpu.Update(args[4],vc);
  gpu.Begin();Commit();MetalCommittedAttentionReference::Run(gpu,args,candidate,cap,chunk,gated);gpu.Flush();gpu.SubmitAndWait();
  Need(gpu.LastCommittedKVAttentionDispatches()==size_t(total>8),"Committed route/scalar fallback differs");
  Need(gpu.LastChunkAttentionMPSDispatches()==size_t(total>8),"MPS route coverage differs");
  Need(gpu.LastChunkAttentionMPSMatrices()==(total>8?activeOwners*8*2:0),"Committed path did not use two products");
  Need(gpu.ChunkAttentionMPSPlanCount()==plans&&gpu.ChunkAttentionMPSScratchBytes()==scratch,"Execution changed prepared plans/storage");
  auto got=gpu.Download(candidate);Need(got.size()==old.size(),"Output size changed");for(size_t i=0;i<got.size();++i){Need(std::isfinite(got[i]),"Nonfinite committed output");Need(std::abs(double(got[i])-old[i])<=1e-5+2e-4*std::abs(double(old[i])),"Complete output exceeds fixed control bound");if(!active[i/4096])Need(got[i]==0,"Inactive committed output stale");}words=got.size();Oracle(got);Oracle(old);
  for(size_t part=0;part<2;++part){auto a=gpu.Download(args[part+3]);const auto& expected=part?expectedV:expectedK;Need(a.size()==expected.size()&&!std::memcmp(a.data(),expected.data(),a.size()*4),"Committed path changed full KV parent");}
  for(size_t index:{size_t(0),size_t(1),size_t(2),size_t(5),size_t(6),size_t(7)}){auto a=gpu.Download(args[index]);const auto& expected=index==0?q:index==1?k:index==2?v:index==5?slots:index==6?active:gate;Need(!std::memcmp(a.data(),expected.data(),a.size()*4),"Readonly input changed");}
 }
 void RejectControls(){
  auto invoke=[&](){gpu.Begin();bool rejected=false;try{MetalCommittedAttentionReference::Run(gpu,args,candidate,cap,chunk,gated);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();Need(rejected,"Unsafe committed control admitted");};
  auto changed=slots;changed[1]=changed[0];gpu.Update(args[5],changed);invoke();gpu.Update(args[5],slots);
  auto flags=active;flags[1]=0;gpu.Update(args[6],flags);invoke();gpu.Update(args[6],active);
  gpu.Begin();gpu.EncodeAdd(args[5],zero,copied);auto inputs=args;inputs[5]=copied;bool pending=false;try{MetalCommittedAttentionReference::Run(gpu,inputs,candidate,cap,chunk,gated);}catch(const std::invalid_argument&){pending=true;}gpu.Abort();Need(pending,"Pending controls admitted");Need(gpu.LastCommittedKVAttentionDispatches()==0,"Abort retained committed dispatch counter");
 }
};
}
int main(){try{size_t words=0,oracle=0,cells=0;double fraction=0;
 const size_t shapes[][4]={{1,8,128,0},{4,17,256,128},{1,128,4096,1920},{4,128,4096,1920},{4,128,4096,3840},{4,128,4096,0}};
 for(const auto& shape:shapes){{Fixture f(shape[0],shape[1],shape[2],shape[3],cells==1||cells==4,cells==1||cells==4);f.Run();if(cells==1)f.RejectControls();words+=f.words;oracle+=f.oracle;fraction=std::max(fraction,f.fraction);}++cells;}
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Committed reference retained native owners");
 std::cout<<"Native committed KV: "<<cells<<" cells, "<<words<<" complete output words, "<<oracle<<" original-bound FP64 values, maximum fraction "<<fraction<<"; full KV/readonly data, two-products, scalar, controls, partial submit, abort, unchanged storage PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
