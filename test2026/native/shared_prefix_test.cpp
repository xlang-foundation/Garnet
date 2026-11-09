// SPDX-License-Identifier: Apache-2.0
#include "../../src/runtime/kv/prefix_cache.h"
#include "../../src/backends/metal/prefix_shared_blocks.h"
#include <iostream>
#include <limits>
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
#include "metal_context.h"
#endif
using namespace Garnet;
void Need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
struct Resource{std::vector<uint8_t> data=std::vector<uint8_t>(4096,37);};
struct State:PrefixSnapshot {
    std::vector<std::shared_ptr<Resource>> owners;std::vector<PrefixSharedResource> resources;
    const int recurrent,hidden;
    State(PrefixCacheIdentity id,size_t n,std::shared_ptr<Resource> resource,int value=1)
        :PrefixSnapshot(std::move(id),n,64),recurrent(value),hidden(value+1) {
        if(resource){owners.push_back(resource);resources.push_back({resource.get(),4096});}
    }
    const std::vector<PrefixSharedResource>& SharedResources()const override{return resources;}
};
PrefixCacheIdentity Id(){return {"weights","shared-block64-fp32","native-tokenizer","text3axis","task-A","text"};}
template<class F>void Reject(F f,const char* m){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,m);}
void Accounting(){
 auto id=Id();auto a=std::make_shared<Resource>();std::weak_ptr<Resource> weak=a;PrefixCache cache(2,65536,true);
 cache.Insert({1},std::make_shared<State>(id,1,a,11));auto one=cache.Stats();
 cache.Insert({2},std::make_shared<State>(id,1,a,22));auto two=cache.Stats();
 Need(one.resourceCount==1&&two.resourceCount==1&&two.resourceBytes==4096,"shared payload counted twice");
 Need(two.privateBytes==2*one.privateBytes&&two.accountedBytes==two.privateBytes+4096,"private/shared union differs");
 auto hit=cache.Lookup(id,{1});a.reset();
 cache.Insert({3},std::make_shared<State>(id,1,std::make_shared<Resource>()));
 cache.Insert({4},std::make_shared<State>(id,1,std::make_shared<Resource>()));
 Need(!weak.expired()&&std::static_pointer_cast<const State>(hit.snapshot)->owners[0]->data[0]==37,"eviction invalidated owning lease");
 cache.Clear();Need(cache.Stats().accountedBytes==0&&cache.Stats().resourceCount==0,"clear retained accounted resources");
 Need(!weak.expired(),"clear released active owner");hit={};Need(weak.expired(),"last hit retained shared resource");
 // Both checkpoints fit when one physical payload is counted once.
 auto r=std::make_shared<Resource>();size_t budget=2*one.privateBytes+4096;PrefixCache tight(4,budget,true);
 Need(tight.Insert({1},std::make_shared<State>(id,1,r))&&tight.Insert({2},std::make_shared<State>(id,1,r)),"shared pair exceeded unique budget");
 Need(tight.Stats().entries==2,"shared pair prematurely evicted");
 Need(tight.Insert({3},std::make_shared<State>(id,1,std::make_shared<Resource>())),"replacement resource rejected");
 Need(tight.Stats().entries==1&&tight.Stats().evictions==2&&tight.Stats().accountedBytes<=budget,"last-reference eviction accounting differs");
}
void Validation(){
 auto id=Id();auto resource=std::make_shared<Resource>();PrefixCache cache(8,65536,true);cache.Insert({1,2,3,4},std::make_shared<State>(id,4,resource,77));auto before=cache.Stats();
 auto hint=cache.FindCommonPrefix(id,{1,2,3});Need(hint.commonTokens==3&&hint.snapshot->Tokens()==4,"common-prefix hint missing");
 auto stats=cache.Stats();Need(stats.hits==before.hits&&stats.misses==before.misses,"hint forged request statistics");
 Need(!cache.Lookup(id,{1,2,3}),"hint was treated as partial recurrent restore");
 auto foreign=id;foreign.isolation="other";Need(!cache.FindCommonPrefix(foreign,{1,2,3,4}).snapshot,"foreign hint accepted");
 for(size_t kind=0;kind<4;kind++){
  auto bad=std::make_shared<State>(id,1,resource);
  if(kind==0)bad->resources[0].bytes=4097;
  if(kind==1)bad->resources[0].identity=nullptr;
  if(kind==2)bad->resources[0].bytes=0;
  if(kind==3)bad->resources[0].bytes=SIZE_MAX;
  auto old=cache.Stats();bool accepted=false;try{accepted=cache.Insert({9},bad);}catch(const std::invalid_argument&){}
  Need(!accepted,"invalid resource accepted");auto now=cache.Stats();Need(old.entries==now.entries&&old.accountedBytes==now.accountedBytes&&old.evictions==now.evictions,"failed admission mutated index");
 }
 PrefixCache legacy(4,65536);Reject([&]{legacy.Insert({1},std::make_shared<State>(id,1,resource));},"legacy mode silently undercounted shared storage");
 cache.Insert({1,2},std::make_shared<State>(id,2,resource,19));auto earlier=std::static_pointer_cast<const State>(cache.Lookup(id,{1,2,9}).snapshot);
 auto later=std::static_pointer_cast<const State>(cache.Lookup(id,{1,2,3,4}).snapshot);
 Need(earlier->recurrent==19&&earlier->hidden==20&&later->recurrent==77&&later->hidden==78,"private boundary state was shared/truncated");
}
void Words(){
 size_t g=3,t=130,w=7;std::vector<float> values(g*t*w),rebuilt(values.size());std::vector<uint32_t> bits(values.size());
 for(size_t i=0;i<bits.size();i++)bits[i]=uint32_t(0x3f000000+i);
 bits[0]=0;bits[1]=0x80000000;bits[3]=0x7fc01234;std::memcpy(values.data(),bits.data(),bits.size()*4);
 for(size_t first=0;first<t;first+=64){size_t n=std::min(size_t(64),t-first);auto b=MakePrefixBlock(values,g,t,w,first,n);Need(PrefixBlockMatches(*b,values,g,t,w,first,n),"block equality failed");WritePrefixBlock(*b,rebuilt,g,t,w,first,n);}
 Need(!std::memcmp(values.data(),rebuilt.data(),values.size()*4),"group-major assembly changed FP32 bits");
 auto b=MakePrefixBlock(values,g,t,w,0,64);auto changed=values;uint32_t flip=0x7fc01235;std::memcpy(changed.data()+3,&flip,4);Need(!PrefixBlockMatches(*b,changed,g,t,w,0,64),"different NaN payload reused");
 changed=values;uint32_t negative=0x80000000;std::memcpy(changed.data(),&negative,4);Need(!PrefixBlockMatches(*b,changed,g,t,w,0,64),"signed zero bytes were conflated");
 Reject([&]{MakePrefixBlock(values,g,t,w,129,2);},"past-end range accepted");
 Reject([&]{MakePrefixBlock(values,g,t,w,0,65);},"oversized page accepted");
 Reject([&]{PrefixWordCount(SIZE_MAX,64,128);},"overflow geometry accepted");
 Reject([&]{DensePrefixBlock bad(64,2,7,std::vector<float>(1));},"bad block extent accepted");
 Reject([&]{WritePrefixBlock(*b,rebuilt,g,t,w,0,63);},"wrong restored extent accepted");
}
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
void DirectCapture(){
 const size_t g=3,cap=193,w=7,t=130,columns=g*cap*w;
 std::vector<float> source(columns,-777),logical(g*t*w),rebuilt(logical.size());
 for(size_t group=0;group<g;++group)for(size_t token=0;token<t;++token)for(size_t lane=0;lane<w;++lane){
  const float value=float(group*10000+token*10+lane+1);source[(group*cap+token)*w+lane]=value;logical[(group*t+token)*w+lane]=value;
 }
 auto first=CapturePrefixBlocks(source.data(),columns,g,cap,w,t,0,{});
 Need(first.blocks.size()==3&&first.newBlocks==3&&first.reusedBlocks==0&&first.copiedBytes==logical.size()*4&&first.comparedBytes==0,"wrong initial direct traffic");
 for(size_t i=0;i<3;++i){auto& b=*first.blocks[i];WritePrefixBlock(b,rebuilt,g,t,w,i*64,b.tokens);Need(b.allFinite,"fresh finite status missing");}
 Need(logical==rebuilt,"direct strided layout differs");
 auto same=CapturePrefixBlocks(source.data(),columns,g,cap,w,t,t,first.blocks);
 Need(same.copiedBytes==0&&same.newBlocks==0&&same.reusedBlocks==3&&same.comparedBytes==logical.size()*4,"identical prefix was recopied");
 for(size_t i=0;i<3;++i)Need(same.blocks[i]==first.blocks[i],"identical block lost identity");
 // Same token lineage with changed numerical state must allocate another block.
 source[(2*cap+70)*w+2]+=1;auto changed=CapturePrefixBlocks(source.data(),columns,g,cap,w,t,t,first.blocks);
 Need(changed.newBlocks==1&&changed.reusedBlocks==2&&changed.copiedBytes==g*64*w*4,"changed numerical history reused");
 Need(changed.blocks[1]!=first.blocks[1]&&changed.blocks[0]==first.blocks[0]&&changed.blocks[2]==first.blocks[2],"wrong changed block");
 auto bounded=CapturePrefixBlocks(source.data(),columns,g,cap,w,t,63,first.blocks);
 Need(bounded.reusedBlocks==0&&bounded.newBlocks==3,"resource hint crossed common-token boundary");
 auto extended=CapturePrefixBlocks(source.data(),columns,g,cap,w,t+1,t,changed.blocks);
 Need(extended.reusedBlocks==2&&extended.newBlocks==1&&extended.blocks[2]->tokens==3,"partial tail block was treated as a full reusable block");
 // A finite immutable block cannot authorize changed infinity or NaN payloads.
 source[1]=std::numeric_limits<float>::infinity();Reject([&]{CapturePrefixBlocks(source.data(),columns,g,cap,w,t,t,first.blocks);},"nonfinite changed block accepted");
 source[1]=std::numeric_limits<float>::quiet_NaN();Reject([&]{CapturePrefixBlocks(source.data(),columns,g,cap,w,t,0,{});},"new nonfinite block accepted");
 auto nonfinite=MakePrefixBlock(std::vector<float>(g*64*w,std::numeric_limits<float>::quiet_NaN()),g,64,w,0,64);
 Need(!nonfinite->allFinite,"nonfinite block marked finite");
 std::vector<float> nanSource(columns,std::numeric_limits<float>::quiet_NaN());Reject([&]{CapturePrefixBlocks(nanSource.data(),columns,g,cap,w,64,64,{nonfinite});},"nonfinite immutable parent bypassed checks");
 source[1]=2;source[cap*w-1]=std::numeric_limits<float>::infinity();
 auto future=CapturePrefixBlocks(source.data(),columns,g,cap,w,t,t,changed.blocks);Need(future.reusedBlocks==3,"uncaptured future slot affected prefix");
 Reject([&]{CapturePrefixBlocks(nullptr,columns,g,cap,w,t,0,{});},"null row accepted");
 Reject([&]{CapturePrefixBlocks(source.data(),columns-1,g,cap,w,t,0,{});},"bad row span accepted");
 Reject([&]{CapturePrefixBlocks(source.data(),columns,g,cap,w,cap+1,0,{});},"oversized prefix accepted");
 Reject([&]{CapturePrefixBlocks(source.data(),columns,g,cap,w,t,t+1,{});},"oversized common range accepted");
 Reject([&]{CapturePrefixBlocks(source.data(),SIZE_MAX,SIZE_MAX,cap,w,t,0,{});},"overflow direct geometry accepted");
 // Exercise the actual completed-MPS-row bridge, including padded row stride,
 // source replacement after capture, read-only isolation, and owner lifetime.
 MetalContext gpu;std::vector<float> rows(4*columns);for(size_t row=0;row<4;++row)for(size_t i=0;i<columns;++i)rows[row*columns+i]=float(row*100000+i+1);
 auto matrix=gpu.Upload(rows,4,columns);auto before=gpu.Download(matrix);
 auto native=gpu.ReadRowPrefixBlocks(matrix,2,g,cap,w,t,0,{});auto repeat=gpu.ReadRowPrefixBlocks(matrix,2,g,cap,w,t,t,native.blocks);
 Need(repeat.reusedBlocks==3&&repeat.copiedBytes==0&&gpu.Download(matrix)==before,"native direct capture mutated source or copied reusable data");
 auto compact=gpu.ReadRowPrefixGroups(matrix,2,g,cap,w,t);std::vector<float> assembled(compact.size());
 for(size_t i=0;i<native.blocks.size();++i)WritePrefixBlock(*native.blocks[i],assembled,g,t,w,i*64,native.blocks[i]->tokens);
 Need(assembled==compact,"native direct capture differs from legacy compact read");
 gpu.ZeroRow(matrix,2);Need(native.blocks[0]->words[0]==before[2*columns],"immutable block borrowed mutable source");
 auto zero=gpu.ReadRowPrefixBlocks(matrix,2,g,cap,w,t,t,native.blocks);Need(zero.newBlocks==3&&zero.reusedBlocks==0,"reset row aliased old resource");
 auto after=gpu.Download(matrix);for(size_t row:{size_t(0),size_t(1),size_t(3)})for(size_t i=0;i<columns;++i)Need(after[row*columns+i]==before[row*columns+i],"capture changed survivor row");
 Reject([&]{gpu.ReadRowPrefixBlocks(matrix,4,g,cap,w,t,0,{});},"native direct row bounds accepted");
 Reject([&]{gpu.ReadRowPrefixBlocks(MetalMatrix(),0,g,cap,w,t,0,{});},"empty native matrix accepted");
 auto unready=gpu.Allocate(4,columns);Reject([&]{gpu.ReadRowPrefixBlocks(unready,0,g,cap,w,t,0,{});},"uninitialized native matrix accepted");
 gpu.Begin();Reject([&]{gpu.ReadRowPrefixBlocks(matrix,0,g,cap,w,t,0,{});},"open command capture accepted");gpu.Abort();
 auto badRows=before;badRows[columns+1]=std::numeric_limits<float>::infinity();gpu.Update(matrix,badRows);
 Reject([&]{gpu.ReadRowPrefixBlocks(matrix,1,g,cap,w,t,0,{});},"native nonfinite data accepted");
 auto badAfter=gpu.Download(matrix);Need(std::memcmp(badRows.data(),badAfter.data(),badRows.size()*4)==0,"failed native capture changed source");
 std::cout<<"direct completed-row finite lineage, strided byte equality, copy counters, changed/tail blocks and owning isolation PASS\n";
}
#endif
int main(){try{Accounting();Validation();Words();
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
DirectCapture();
#endif
std::cout<<"shared prefix unique-byte accounting, exact maintenance hints, independent recurrent/hidden boundaries, owning eviction/clear and FP32 block assembly PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
