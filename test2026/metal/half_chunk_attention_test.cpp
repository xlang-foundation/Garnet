// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "half_attention_profile.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace Garnet;
namespace Garnet {struct MetalHalfAttentionReference {
 static void Run(MetalContext& g,const std::vector<MetalMatrix>& a,const MetalMatrix& o,size_t c,size_t t,bool gated){g.EncodeHalfAttentionChunk(a,o,32,8,128,c,t,gated);}
 static std::array<MetalMatrix,3> Scratch(MetalContext& g){return g.HalfAttentionScratchForTest();}
 static uint32_t PartialStatus(MetalContext& g){return g.HalfAttentionPartialStatusForTest();}
 static MetalMatrix CacheView(MetalContext& g,const MetalMatrix& m,size_t n,size_t c){return g.PrepareKVRowSpan(m,1,n,8,128,c);}
};}
namespace {
bool halfOnlyProfile=false;
void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
float Word(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
uint16_t HalfBits(float x){_Float16 h=(_Float16)x;uint16_t v;std::memcpy(&v,&h,2);return v;}
double Half(float x){return double((_Float16)x);}
float Value(size_t i,uint32_t salt){uint32_t x=uint32_t(i)^salt;x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return float(int(x%20001)-10000)*.00003f;}
std::vector<float> Read(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f),"Missing trained fixture");size_t n=f.tellg();Need(n&&n%4==0,"Invalid fixture bytes");std::vector<float> a(n/4);f.seekg(0);f.read((char*)a.data(),n);Need(bool(f),"Truncated fixture");return a;}
void Save(const std::filesystem::path& p,const std::vector<float>& a){std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*4);Need(bool(f),"Cannot save native replay");}
std::filesystem::path FreshReference(const std::filesystem::path& root){std::filesystem::create_directories(root);auto name="reference-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());auto out=root/name;Need(std::filesystem::create_directory(out),"Fresh reference directory required");std::ofstream f(root/"active.txt");f<<name<<'\n';Need(bool(f),"Cannot select reference run");return out;}
std::filesystem::path ResolveReference(const std::filesystem::path& root){std::ifstream f(root/"active.txt");std::string name;f>>name;Need(bool(f)&&name.find('/')==std::string::npos&&name.find("..") ==std::string::npos&&!name.empty(),"Missing reference run");return root/name;}
struct Fixture {
 MetalContext& gpu;size_t batch,chunk,cap,first,rows;bool gated,offsetCache;std::array<std::vector<float>,8> data;
 std::array<std::vector<float>,8> initialParents;std::array<MetalMatrix,8> parents;std::vector<MetalMatrix> args,rounded;
 MetalMatrix outParent,out,control,oldControl,copiedSlots,zeroSlots;std::vector<float> outPoison;
 size_t fp64=0,checked=0;double maxFraction=0;
 Fixture(MetalContext& g,size_t b,size_t t,size_t c,size_t f,bool gate,bool ragged,int adversary=0,const std::filesystem::path& fixture={},size_t ci=0):gpu(g),batch(b),chunk(t),cap(c),first(f),rows(b*t),gated(gate),offsetCache(b<=2){
  const size_t widths[]={4096,1024,1024,8*cap*128,8*cap*128,chunk,chunk,4096};
  for(size_t i=0;i<8;++i){size_t n=(i<3||i==7?rows:batch)*widths[i];data[i].resize(n);for(size_t j=0;j<n;++j)data[i][j]=Value(j,17+37*i);}
  for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t){data[5][b*chunk+t]=float(first+t);data[6][b*chunk+t]=1;}
  if(ragged){for(size_t t=chunk-3;t<chunk;++t)data[6][chunk+t]=0;for(size_t t=1;t<chunk;++t)data[6][2*chunk+t]=0;for(size_t t=0;t<chunk;++t)data[6][3*chunk+t]=0;}
  if(adversary==1){std::fill(data[0].begin(),data[0].end(),1.00048828125f);for(size_t r=0;r<rows;++r)for(size_t d=0;d<1024;++d){data[1][r*1024+d]=r%2?1.0f:1.00048828125f;data[2][r*1024+d]=r%2?-30.0f:30.0f;}}
  if(adversary==2){for(size_t r=0;r<rows;++r){for(size_t d=0;d<4096;++d)data[0][r*4096+d]=float(d%128==0);for(size_t d=0;d<1024;++d){data[1][r*1024+d]=d%128?0:-float(22*std::sqrt(128.));data[2][r*1024+d]=2048;}}
   for(size_t b=0;b<batch;++b)for(size_t h=0;h<8;++h)for(size_t t=0;t<cap;++t)for(size_t d=0;d<128;++d){size_t at=((b*8+h)*cap+t)*128+d;data[3][at]=t&&d==0?-float(22*std::sqrt(128.)):0;data[4][at]=t?2048:0;}}
  if(!fixture.empty())for(size_t i=0;i<8;++i){auto a=Read(fixture/("case"+std::to_string(ci)+"-input"+std::to_string(i)+".f32"));size_t nr=i<3||i==7?rows:batch,stride=(a.size()-32)/nr;Need(a.size()==32+nr*stride&&stride>=widths[i],"Wrong trained fixture extent");for(size_t r=0;r<nr;++r)std::copy_n(a.begin()+16+r*stride,widths[i],data[i].begin()+r*widths[i]);}
  for(size_t i=0;i<8;++i){size_t nr=i<3||i==7?rows:batch;
   if(i==3||i==4){if(offsetCache){initialParents[i].assign((batch+2)*widths[i],Word(0x7fc12345));std::copy(data[i].begin(),data[i].end(),initialParents[i].begin()+widths[i]);parents[i]=gpu.Upload(initialParents[i],batch+2,widths[i]);args.push_back(MetalHalfAttentionReference::CacheView(gpu,parents[i],batch,cap));}else{initialParents[i]=data[i];parents[i]=gpu.Upload(data[i],batch,widths[i]);args.push_back(parents[i]);}}
   else if(widths[i]%4==0){initialParents[i].assign(data[i].size()+32,Word(0x7fc12345));std::copy(data[i].begin(),data[i].end(),initialParents[i].begin()+16);parents[i]=gpu.Upload(initialParents[i],1,initialParents[i].size());args.push_back(gpu.PrepareDiagnosticDenseView(parents[i],16,nr,widths[i]));}
   else{initialParents[i]=data[i];parents[i]=gpu.Upload(data[i],nr,widths[i]);args.push_back(parents[i]);}
   auto r=data[i];if(i<5)for(auto& x:r)x=float(Half(x));rounded.push_back(gpu.Upload(r,nr,widths[i]));
  }
  outPoison.assign(rows*4096+32,Word(0x7fc12345));outParent=gpu.Upload(outPoison,1,outPoison.size());out=gpu.PrepareDiagnosticDenseView(outParent,16,rows,4096);
  control=gpu.Allocate(rows,4096);oldControl=gpu.Allocate(rows,4096);copiedSlots=gpu.Allocate(batch,chunk);zeroSlots=gpu.Upload(std::vector<float>(batch*chunk,0),batch,chunk);
  gpu.PrepareAttentionChunk(args,32,8,128,cap,chunk,halfOnlyProfile);Need(gpu.HalfAttentionScratchBytes()==68ULL*1024*1024+384,"Half profile scratch size changed");
  if(halfOnlyProfile)Need(gpu.HalfOnlyAttentionPlanCount()>0&&gpu.ChunkAttentionMPSPlanCount()==0&&gpu.ChunkAttentionMPSScratchBytes()==0,"Half-only preparation retained MPS fallback workspaces");
 }
 std::string ReferenceKey() const {uint64_t h=1469598103934665603ULL;for(const auto& a:data)for(float x:a){uint32_t bits;std::memcpy(&bits,&x,4);h^=bits;h*=1099511628211ULL;}return std::to_string(batch)+":"+std::to_string(chunk)+":"+std::to_string(cap)+":"+std::to_string(first)+":"+std::to_string(gated)+":"+std::to_string(h);}
 void MakeReference(const std::filesystem::path& folder,size_t ci){gpu.Begin();gpu.EncodeAttentionChunk(rounded,control,32,8,128,cap,chunk,gated);gpu.EncodeAttentionChunk(args,oldControl,32,8,128,cap,chunk,gated);gpu.SubmitAndWait();Save(folder/("case"+std::to_string(ci)+"-rounded-control.f32"),gpu.Download(control));Save(folder/("case"+std::to_string(ci)+"-original-control.f32"),gpu.Download(oldControl));std::ofstream f(folder/("case"+std::to_string(ci)+"-input-key.txt"));f<<ReferenceKey()<<'\n';Need(bool(f),"Cannot bind reference input");}
 size_t Count(size_t b) const {size_t n=0;while(n<chunk&&data[6][b*chunk+n])++n;return n;}
 void Commit(){CommitOn(gpu);}
 void CommitOn(MetalContext& execution){for(size_t i=0;i<2;++i)execution.EncodeKVChunk(args[1+i],args[3+i],args[5],args[6],args[3+i],8,128,cap,chunk,true);}
 void Poison(){auto scratch=MetalHalfAttentionReference::Scratch(gpu);for(size_t i=0;i<3;++i){const size_t counts[]={512*4096/2+32,4*8*4096*128/2+32,4*8*4096*128/2+32};std::vector<float> a(counts[i],Word(0x7e097e09));gpu.Update(scratch[i],a);}}
 void CheckScratch(bool used){auto scratch=MetalHalfAttentionReference::Scratch(gpu);for(size_t part=0;part<3;++part){auto parent=gpu.Download(scratch[part]);std::vector<uint16_t> expected(parent.size()*2,0x7e09);if(used){
   if(part==0)for(size_t row=0;row<rows;++row)for(size_t d=0;d<4096;++d)expected[32+row*4096+d]=HalfBits(row%chunk<Count(row/chunk)?data[0][row*4096+d]:0);
   else{size_t extent=gpu.LastChunkAttentionBucketCapacity();for(size_t b=0;b<batch;++b)for(size_t h=0;h<8;++h)for(size_t t=0;t<extent;++t)for(size_t d=0;d<128;++d){float x=0;if(Count(b)&&t<first+Count(b))x=t<first?data[2+part][((b*8+h)*cap+t)*128+d]:data[part][((b*chunk+t-first)*8+h)*128+d];expected[32+((b*8+h)*cap+t)*128+d]=HalfBits(x);}}}
   Need(!std::memcmp(parent.data(),expected.data(),parent.size()*4),"Half scratch complete write/guard/tail differs");checked+=expected.size();}}
 std::vector<float> Run(const std::filesystem::path& folder,size_t ci,const std::filesystem::path& references){
  size_t total=0;for(size_t b=0;b<batch;++b)total+=Count(b);bool half=true;
  std::ifstream inputKey(references/("case"+std::to_string(ci)+"-input-key.txt"));std::string key;inputKey>>key;Need(bool(inputKey)&&key==ReferenceKey(),"Reference inputs/shape differ");
  auto originalReference=Read(references/("case"+std::to_string(ci)+"-original-control.f32"));auto reference=half?Read(references/("case"+std::to_string(ci)+"-rounded-control.f32")):originalReference;Need(reference.size()==rows*4096&&originalReference.size()==rows*4096,"Reference output extent differs");gpu.Update(outParent,outPoison);Poison();auto owners=MetalContext::ProfileOwnerStats();size_t plans=gpu.ChunkAttentionMPSPlanCount();
  gpu.Begin();Commit();MetalHalfAttentionReference::Run(gpu,args,out,cap,chunk,gated);gpu.Flush();gpu.SubmitAndWait();
  Need(gpu.LastHalfAttentionDispatches()==size_t(half),"Half selection/scalar fallback counter differs");Need(gpu.LastChunkAttentionMPSDispatches()==0,"Private half path used MPS products");
  auto got=gpu.Download(out);auto parent=gpu.Download(outParent);for(size_t i=0;i<16;++i){Need(!std::memcmp(&parent[i],&outPoison[i],4)&&!std::memcmp(&parent[parent.size()-16+i],&outPoison[outPoison.size()-16+i],4),"Offset output guard changed");}
  std::vector<std::vector<double>> prefix(batch*8);
  for(size_t b=0;b<batch;++b)for(size_t h=0;h<8;++h){double mx=0;for(size_t s=0;s<first+Count(b);++s){for(size_t d=0;d<128;++d){float val=s<first?data[4][((b*8+h)*cap+s)*128+d]:data[2][((b*chunk+s-first)*8+h)*128+d];mx=std::max(mx,std::abs(Half(val)));}prefix[b*8+h].push_back(mx);}}
  for(size_t i=0;i<got.size();++i){Need(std::isfinite(got[i]),"Native half output nonfinite");size_t flat=i/4096,b=flat/chunk,t=flat%chunk,h=i%4096/128;if(t>=Count(b)){Need(got[i]==0,"Inactive output stale");continue;}double bound=1e-5+2e-4*std::abs(double(reference[i]));if(half)bound+=(std::ldexp(1.,-11)+std::ldexp(1.,-40)*(first+t+1))*prefix[b*8+h/4][first+t];Need(std::abs(double(got[i])-reference[i])<=bound,"Complete native half output exceeds fixed new-policy budget");}
  for(size_t b=0;b<batch;++b)if(Count(b))for(size_t t:std::vector<size_t>{0,Count(b)/2,Count(b)-1})for(size_t h:{size_t(0),size_t(7),size_t(8),size_t(31)}){
   size_t visible=first+t+1,kh=h/4;std::vector<double> probs(visible);double maximum=-INFINITY;
   auto operand=[&](float x){return half?Half(x):double(x);};
   for(size_t s=0;s<visible;++s){double dot=0;for(size_t d=0;d<128;++d){float key=s<first?data[3][((b*8+kh)*cap+s)*128+d]:data[1][((b*chunk+s-first)*8+kh)*128+d];dot+=operand(data[0][((b*chunk+t)*32+h)*128+d])*operand(key);}probs[s]=dot/std::sqrt(128.);maximum=std::max(maximum,probs[s]);}
   double sum=0;for(auto& x:probs){x=std::exp(x-maximum);sum+=x;}
   for(size_t d=0;d<128;++d){double expected=0,weighted=0,maxv=0;for(size_t s=0;s<visible;++s){float raw=s<first?data[4][((b*8+kh)*cap+s)*128+d]:data[2][((b*chunk+s-first)*8+kh)*128+d];double val=operand(raw);expected+=probs[s]/sum*val;weighted+=probs[s]/sum*std::abs(val);maxv=std::max(maxv,std::abs(val));}size_t i=((b*chunk+t)*32+h)*128+d;if(gated)expected/=1+std::exp(-double(data[7][i]));double bound=1e-5+2e-4*std::abs(expected);if(half)bound+=std::ldexp(1.,-11)*weighted+std::ldexp(1.,-40)*visible*maxv;double fraction=std::abs(double(got[i])-expected)/bound;Need(fraction<=1,"Native half exceeds fixed FP64 new-policy bound");maxFraction=std::max(maxFraction,fraction);++fp64;}}
  CheckScratch(half);
  for(size_t i=0;i<8;++i){auto expected=initialParents[i];if(i==3||i==4)for(size_t b=0;b<batch;++b)for(size_t t=0;t<Count(b);++t)for(size_t h=0;h<8;++h)for(size_t d=0;d<128;++d){size_t at=((b*8+h)*cap+first+t)*128+d;if(offsetCache)at+=8*cap*128;expected[at]=data[i-2][((b*chunk+t)*8+h)*128+d];}auto actual=gpu.Download(parents[i]);Need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Input/canonical-cache full parent differs");checked+=actual.size();}
  auto after=MetalContext::ProfileOwnerStats();Need(owners.matrices==after.matrices&&owners.pipelineSets==after.pipelineSets&&plans==gpu.ChunkAttentionMPSPlanCount(),"Half execution allocated native owners/plans");
  if(!folder.empty()){Save(folder/("case"+std::to_string(ci)+"-native-output.f32"),got);Save(folder/("case"+std::to_string(ci)+"-rounded-control.f32"),reference);Save(folder/("case"+std::to_string(ci)+"-original-control.f32"),originalReference);Save(folder/("case"+std::to_string(ci)+"-output-parent.f32"),parent);}
  std::cout<<"half_native cell B="<<batch<<" T="<<chunk<<" C="<<cap<<" first="<<first<<" dispatch="<<size_t(half)<<" fp64="<<fp64<<" max_fraction="<<maxFraction<<" checked="<<checked<<" PASS\n";return got;
 }
 void RejectControls(){
#ifdef GARNET_METAL_HALF_ONLY_PREPARATION
  if(halfOnlyProfile){
   gpu.Update(outParent,outPoison);auto before=gpu.Download(outParent);gpu.Begin();bool rejected=false;
   try{gpu.EncodeAttentionChunk(args,out,32,8,128,cap,chunk,gated);}catch(const std::logic_error&){rejected=true;}
   gpu.Abort();Need(rejected,"Half-only profile silently ran FP32 without a prepared fallback");
   auto after=gpu.Download(outParent);Need(after.size()==before.size()&&!std::memcmp(after.data(),before.data(),after.size()*4),"Rejected FP32 fallback changed output parent");
  }
#endif
  auto reject=[&](const std::vector<MetalMatrix>& a,const MetalMatrix& target){gpu.Begin();bool rejected=false;try{MetalHalfAttentionReference::Run(gpu,a,target,cap,chunk,gated);}catch(const std::exception&){rejected=true;}gpu.Abort();Need(rejected,"Invalid half input/control admitted");Need(gpu.LastHalfAttentionDispatches()==0,"Abort retained half counter");};
  auto positions=data[5];positions[1]=positions[0];gpu.Update(args[5],positions);reject(args,out);gpu.Update(args[5],data[5]);
  auto flags=data[6];flags[1]=0;gpu.Update(args[6],flags);reject(args,out);gpu.Update(args[6],data[6]);reject(args,args[0]);
  gpu.Begin();gpu.EncodeAdd(args[5],zeroSlots,copiedSlots);auto pending=args;pending[5]=copiedSlots;bool rejected=false;try{MetalHalfAttentionReference::Run(gpu,pending,out,cap,chunk,gated);}catch(const std::exception&){rejected=true;}gpu.Abort();Need(rejected,"Pending half controls admitted");
 }
 void RejectRange(){for(int mode=0;mode<6;++mode){gpu.Update(outParent,outPoison);auto bad=data[0];if(mode<3)bad[0]=mode==0?2049.0f:mode==1?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity();gpu.Update(args[0],bad);if(mode>=3){size_t index=mode==3?3:mode==4?4:7;auto altered=data[index];altered[0]=mode==3?-std::numeric_limits<float>::infinity():mode==4?2049.0f:std::numeric_limits<float>::quiet_NaN();gpu.Update(args[index],altered);}gpu.Begin();Commit();MetalHalfAttentionReference::Run(gpu,args,out,cap,chunk,gated);if(mode==1)gpu.Flush();bool failed=false;try{gpu.SubmitAndWait();}catch(const std::runtime_error&){failed=true;}Need(failed,"Out-of-domain GPU value admitted");bool unreadable=false;try{gpu.Download(out);}catch(const std::exception&){unreadable=true;}Need(unreadable,"Failed half output became ready");gpu.Abort();Need(gpu.LastHalfAttentionDispatches()==0,"Failed/aborted half counter retained");gpu.Update(args[0],data[0]);for(size_t i:std::vector<size_t>{3,4,7})gpu.Update(args[i],data[i]);}
 }
};
void CheckContextHandoff(Fixture& f,const std::vector<float>& expected){
 // Preparation belongs to f.gpu; this session has never prepared an attention
 // profile. This matches graph preparation followed by decoder execution.
 MetalContext session;auto owners=MetalContext::ProfileOwnerStats();
 f.gpu.Update(f.outParent,f.outPoison);f.Poison();session.Begin();f.CommitOn(session);
 MetalHalfAttentionReference::Run(session,f.args,f.out,f.cap,f.chunk,f.gated);session.Flush();
 Need(MetalHalfAttentionReference::PartialStatus(session)==0,"Valid handoff range status differs");
 bool pending=true;try{f.gpu.Download(f.out);pending=false;}catch(const std::exception&){}Need(pending,"Flushed handoff result published early");
 session.SubmitAndWait();auto got=session.Download(f.out);Need(got.size()==expected.size()&&!std::memcmp(got.data(),expected.data(),got.size()*4),"Graph/session handoff changed complete output");
 Need(session.LastHalfAttentionDispatches()==1,"Session did not use prepared shared half path");f.CheckScratch(true);
 auto after=MetalContext::ProfileOwnerStats();Need(owners.matrices==after.matrices&&owners.pipelineSets==after.pipelineSets,"Cross-context half execution allocated native owners");

 auto bad=f.data[0];bad[0]=std::numeric_limits<float>::quiet_NaN();f.gpu.Update(f.args[0],bad);
 session.Begin();f.CommitOn(session);MetalHalfAttentionReference::Run(session,f.args,f.out,f.cap,f.chunk,f.gated);session.Flush();
 auto failure=MetalHalfAttentionReference::PartialStatus(session);Need(failure==1,"Invalid flushed Q did not set range flag");
 // A completed non-half batch must succeed while another context owns a
 // failed half batch. Begin, Abort and destruction must not clear its flag.
 f.gpu.Begin();f.gpu.EncodeAdd(f.args[5],f.zeroSlots,f.copiedSlots);f.gpu.SubmitAndWait();
 Need(f.gpu.Download(f.copiedSlots)==f.data[5],"Unrelated non-half batch inherited failure");
 {MetalContext other;other.Begin();other.Abort();other.Begin();other.SubmitAndWait();}
 Need(MetalHalfAttentionReference::PartialStatus(session)==failure,"Unrelated context cleared active half failure");
 // A later valid dispatch in the same partially submitted batch must not
 // reset an earlier error, and neither result may become readable.
 f.gpu.Update(f.args[0],f.data[0]);MetalHalfAttentionReference::Run(session,f.args,f.control,f.cap,f.chunk,f.gated);session.Flush();
 Need(MetalHalfAttentionReference::PartialStatus(session)==failure,"Later valid half dispatch cleared batch failure");
 bool rejected=false;try{session.SubmitAndWait();}catch(const std::runtime_error&){rejected=true;}Need(rejected,"Cross-context invalid half batch admitted");
 for(auto m:{f.out,f.control}){bool unreadable=false;try{f.gpu.Download(m);}catch(const std::exception&){unreadable=true;}Need(unreadable,"Failed shared-status output published");}
 session.Abort();Need(session.LastHalfAttentionDispatches()==0,"Session Abort retained half dispatches");
 // Failed batches intentionally invalidate all written owners, including
 // canonical KV. Restore fixture state before subsequent rejection/recovery.
 for(size_t i:{size_t(3),size_t(4)})f.gpu.Update(f.args[i],f.data[i]);
 std::cout<<"half_native graph/session handoff and shared failure isolation PASS\n";
}
void CheckBatchIndependence(MetalContext& gpu){
 Fixture alone(gpu,1,8,256,0,false,false),batched(gpu,4,8,256,0,false,false);
 auto run=[&](Fixture& f){gpu.Begin();f.Commit();MetalHalfAttentionReference::Run(gpu,f.args,f.out,f.cap,f.chunk,f.gated);gpu.SubmitAndWait();Need(gpu.LastHalfAttentionDispatches()==1,"Half tiny tail changed numerical policy");return gpu.Download(f.out);};
 auto expected=run(alone);
 auto compare=[&](){auto got=run(batched);Need(!std::memcmp(got.data(),expected.data(),expected.size()*4),"Same request changed with batch occupancy/context bucket");};
 compare();
 // Unrelated owners extend the selected context bucket from128 to256.
 for(size_t b=1;b<4;++b)for(size_t t=0;t<8;++t)batched.data[5][b*8+t]=float(128+t);
 gpu.Update(batched.args[5],batched.data[5]);compare();Need(gpu.LastChunkAttentionBucketCapacity()==256,"Long-owner bucket case not exercised");
 // The same first owner remains after all other owners retire; total8 now.
 for(size_t b=1;b<4;++b)for(size_t t=0;t<8;++t)batched.data[6][b*8+t]=0;
 gpu.Update(batched.args[6],batched.data[6]);compare();Need(gpu.LastChunkAttentionBucketCapacity()==128,"Short-owner bucket case not exercised");
 std::fill(batched.data[6].begin(),batched.data[6].end(),0);gpu.Update(batched.args[6],batched.data[6]);auto empty=run(batched);for(float x:empty)Need(x==0,"Empty half owners retained stale values");
 std::cout<<"half_native batch occupancy, context bucket, tiny tail and empty-owner policy PASS exact_words="<<3*expected.size()<<'\n';
}

#ifdef GARNET_METAL_HALF_ONLY_PREPARATION
void CheckExplicitFallback(MetalContext& gpu){
 Fixture f(gpu,4,128,256,0,false,false);
 const auto profiles=gpu.HalfOnlyAttentionPlanCount();
 gpu.PrepareAttentionChunk(f.args,32,8,128,f.cap,f.chunk,false);
 Need(gpu.HalfOnlyAttentionPlanCount()==profiles&&gpu.ChunkAttentionMPSPlanCount()==1&&gpu.ChunkAttentionMPSScratchBytes()>0,"Explicit default preparation failed to add an independent MPS fallback");
 auto run=[&](const MetalMatrix& out){gpu.Begin();gpu.EncodeAttentionChunk(f.args,out,32,8,128,f.cap,f.chunk,false);gpu.SubmitAndWait();Need(gpu.LastChunkAttentionMPSDispatches()==1,"Explicit fallback did not execute its MPS route");return gpu.Download(out);};
 const auto a=run(f.control),b=run(f.oldControl);
 Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"Repeated explicit fallback changed its complete output");
 for(auto x:a)Need(std::isfinite(x),"Explicit fallback returned nonfinite output");
 std::cout<<"half_native separate half/default registries and explicit MPS fallback PASS\n";
}
#endif

#ifdef GARNET_METAL_CHUNK_HALF_WIDE
void CheckWidePartition(MetalContext& gpu,size_t first,size_t active,bool gated){
 Fixture whole(gpu,1,512,4096,first,gated,false),piece(gpu,1,128,4096,first,gated,false);
 for(size_t t=active;t<512;++t)whole.data[6][t]=0;gpu.Update(whole.args[6],whole.data[6]);
 auto run=[&](Fixture& f){gpu.Begin();f.Commit();MetalHalfAttentionReference::Run(gpu,f.args,f.out,f.cap,f.chunk,f.gated);gpu.SubmitAndWait();Need(gpu.LastHalfAttentionDispatches()==1,"Wide partition lost half route");return gpu.Download(f.out);};
 const auto expected=run(whole);std::vector<float> joined(expected.size(),0);
 for(size_t start=0;start<active;start+=128){
  for(size_t role:{size_t(0),size_t(1),size_t(2),size_t(7)}){const size_t width=role==0||role==7?4096:1024;
   piece.data[role].assign(whole.data[role].begin()+start*width,whole.data[role].begin()+(start+128)*width);gpu.Update(piece.args[role],piece.data[role]);}
  for(size_t t=0;t<128;++t){piece.data[5][t]=float(first+start+t);piece.data[6][t]=start+t<active?1.f:0.f;}
  gpu.Update(piece.args[5],piece.data[5]);gpu.Update(piece.args[6],piece.data[6]);auto values=run(piece);
  std::copy(values.begin(),values.end(),joined.begin()+start*4096);
 }
 Need(!std::memcmp(expected.data(),joined.data(),joined.size()*4),"Changing512/128 chunk partition changed complete attention output");
 for(size_t role:{size_t(3),size_t(4)}){auto a=gpu.Download(whole.parents[role]),b=gpu.Download(piece.parents[role]);Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"Wide partition changed complete canonical cache parent/guards");}
 std::cout<<"half_native512/128 exact partition first="<<first<<" active="<<active<<" gated="<<gated<<" output_words="<<joined.size()<<" PASS\n";
}
#endif
}
int main(int argc,char** argv){try{
 size_t samples=0,words=0,cells=0;double fraction=0;std::string mode=argc>1?argv[1]:"";
 bool makeReference=mode=="--reference"||mode=="--reference-trained",trained=mode=="--reference-trained"||mode=="--trained";
#ifdef GARNET_METAL_HALF_ONLY_PREPARATION
 halfOnlyProfile=!makeReference;
#endif
 Need((mode=="--reference"&&argc==3)||(mode=="--selftest"&&argc==3)||(mode=="--reference-trained"&&argc==4)||(mode=="--trained"&&argc==5),"Usage: test --reference ROOT | --selftest ROOT | --reference-trained FIXTURE ROOT | --trained FIXTURE ROOT FRESH_OUTPUT");
 std::filesystem::path input=trained?argv[2]:"",root=trained?argv[3]:argv[2],references=makeReference?FreshReference(root):ResolveReference(root),output=mode=="--trained"?argv[4]:"";
 if(!output.empty()){Need(!std::filesystem::exists(output),"Fresh native output required");std::filesystem::create_directories(output);}
 {MetalContext gpu;
 if(trained){for(size_t ci=0;ci<3;++ci){Fixture f(gpu,4,128,4096,1920,false,false,0,input,ci);if(makeReference)f.MakeReference(references,ci);else f.Run(output,ci,references);samples+=f.fp64;words+=f.checked;fraction=std::max(fraction,f.maxFraction);++cells;}}
 else{const size_t shapes[][4]={{1,8,128,0},{4,17,256,128},{1,128,4096,1920},{4,128,4096,1920},{4,128,4096,3840},{4,128,4096,0},{1,17,128,0},{1,17,4096,3840}
#ifdef GARNET_METAL_CHUNK_HALF_WIDE
 ,{1,256,4096,1920},{2,256,4096,0},{1,512,4096,1920},{1,512,4096,3584}
#endif
 };
  for(size_t ci=0;ci<sizeof(shapes)/sizeof(shapes[0]);++ci){auto a=shapes[ci];Fixture f(gpu,a[0],a[1],a[2],a[3],ci==1||ci==4,ci==1||ci==4,ci==6?1:ci==7?2:0);if(makeReference)f.MakeReference(references,ci);else{auto expected=f.Run({},ci,references);if(ci==1){CheckContextHandoff(f,expected);f.RejectControls();f.RejectRange();f.Run({},ci,references);}}samples+=f.fp64;words+=f.checked;fraction=std::max(fraction,f.maxFraction);++cells;}}
 if(!makeReference&&!trained)CheckBatchIndependence(gpu);
#ifdef GARNET_METAL_CHUNK_HALF_WIDE
 if(!makeReference&&!trained){CheckWidePartition(gpu,0,512,false);CheckWidePartition(gpu,1920,257,false);CheckWidePartition(gpu,3584,512,true);}
#endif
#ifdef GARNET_METAL_HALF_ONLY_PREPARATION
 if(!makeReference&&!trained)CheckExplicitFallback(gpu);
#endif
 }
 auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Half attention retained owners");
 auto memory=MetalContext::MemoryStats();if(!makeReference)Need(memory.allocated<=4ULL*1024*1024,"Half attention device allocations remain after close");
 std::cout<<(makeReference?"Native MPS reference PASS":"Native half attention PASS")<<" cells="<<cells<<" fp64_values="<<samples<<" checked_parent_words="<<words<<" max_fraction="<<fraction<<" closed_device_bytes="<<memory.allocated<<" all_owners_closed=true\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
