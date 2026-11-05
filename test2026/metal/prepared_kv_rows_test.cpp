// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace Garnet {
struct MetalPreparedKVRowsTestAccess {
    static MetalMatrix View(MetalContext& gpu,const MetalMatrix& owner,size_t first,size_t rows,size_t h,size_t d,size_t c) {
        return gpu.PrepareKVRowSpan(owner,first,rows,h,d,c);
    }
};
}
namespace {
void Need(bool good,const char* message){if(!good)throw std::runtime_error(message);}
template<class F>void Reject(F&& f,const char* message){bool failed=false;try{f();}catch(const std::exception&){failed=true;}Need(failed,message);}
MetalMatrix View(MetalContext& gpu,const MetalMatrix& owner,size_t first,size_t rows,size_t h,size_t d,size_t c){return MetalPreparedKVRowsTestAccess::View(gpu,owner,first,rows,h,d,c);}
void Equal(const std::vector<float>& a,const std::vector<float>& b,const char* message){Need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),message);}
void Cell(MetalContext& gpu,size_t first,size_t rows,size_t h,size_t d,size_t c,size_t chunk,bool gapped) {
    const size_t width=h*d*c,tokenWidth=h*d;
    std::vector<float> original(4*width),tokens(rows*chunk*tokenWidth),slots(rows*chunk),flags(rows*chunk);
    for(size_t i=0;i<original.size();++i)original[i]=float(int((i*131+71)%65521)-32760)*0x1p-16f;
    for(size_t i=0;i<tokens.size();++i)tokens[i]=float(int((i*37+97)%65521)-32760)*0x1p-15f;
    for(size_t row=0;row<rows;++row)for(size_t t=0;t<chunk;++t){slots[row*chunk+t]=float(gapped?t%3:c-chunk+t);flags[row*chunk+t]=float(!gapped||t%3!=1);}
    auto cache=gpu.Upload(original,4,width);const auto allocated=MetalContext::MemoryStats().allocated;
    auto view=View(gpu,cache,first,rows,h,d,c);
    Need(view.SharesStorage(cache)&&MetalContext::MatrixMetadataMatchesLive(view),"Prepared row span lost owner/metadata");
    Need(MetalContext::MemoryStats().allocated==allocated,"Preparing cache view allocated GPU memory");
    auto token=gpu.Upload(tokens,rows*chunk,tokenWidth),positions=gpu.Upload(slots,rows,chunk),active=gpu.Upload(flags,rows,chunk);
    auto expected=original;
    for(size_t row=0;row<rows;++row)for(size_t t=0;t<chunk;++t)if(flags[row*chunk+t])for(size_t head=0;head<h;++head)
        std::copy_n(tokens.data()+(row*chunk+t)*tokenWidth+head*d,d,expected.data()+(first+row)*width+(head*c+size_t(slots[row*chunk+t]))*d);
    gpu.Begin();gpu.EncodeKVChunk(token,view,positions,active,view,h,d,c,chunk,!gapped);gpu.SubmitAndWait();
    Equal(gpu.Download(cache),expected,"Span update changed complete owner cache");
    Equal(gpu.Download(view),std::vector<float>(expected.begin()+first*width,expected.begin()+(first+rows)*width),"Span download ignored byte offset");
    // Host updates and zeros must change only the selected physical rows.
    gpu.Update(view,std::vector<float>(rows*width,7.f));std::fill(expected.begin()+first*width,expected.begin()+(first+rows)*width,7.f);Equal(gpu.Download(cache),expected,"Span host update crossed rows");
    gpu.Zero(view);std::fill(expected.begin()+first*width,expected.begin()+(first+rows)*width,0.f);Equal(gpu.Download(cache),expected,"Span zero crossed rows");
}
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
void LastHiddenRows(MetalContext& gpu) {
    for(size_t first=0;first<4;++first)for(size_t rows=1;rows<=std::min(size_t(2),4-first);++rows) {
        constexpr size_t width=16,chunk=8;std::vector<float> original(4*width,7.f),values(rows*chunk*width),flags(rows*chunk,1.f);
        for(size_t i=0;i<values.size();++i)values[i]=float(i+1);
        for(size_t row=0;row<rows;++row)flags[row*chunk+chunk-1]=0;
        auto owner=gpu.Upload(original,4,width),view=View(gpu,owner,first,rows,1,width,1);
        auto input=gpu.Upload(values,rows,chunk*width),active=gpu.Upload(flags,rows,chunk);
        gpu.Begin();gpu.EncodeLastActiveHidden(input,active,view,chunk);gpu.SubmitAndWait();
        for(size_t row=0;row<rows;++row)std::copy_n(values.data()+(row*chunk+chunk-2)*width,width,original.data()+(first+row)*width);
        Equal(gpu.Download(owner),original,"Last-hidden row span changed physical row routing");
        gpu.Update(active,std::vector<float>(rows*chunk,0.f));gpu.Begin();gpu.EncodeLastActiveHidden(input,active,view,chunk);gpu.SubmitAndWait();
        Equal(gpu.Download(owner),original,"Paused compact owner changed its last-hidden bank");
    }
}
#endif
void Readiness(MetalContext& gpu) {
    constexpr size_t h=2,d=8,c=16,width=h*d*c;
    std::vector<float> original(4*width,3.f);
    auto owner=gpu.Upload(original,4,width),a=View(gpu,owner,2,1,h,d,c),b=View(gpu,owner,1,2,h,d,c),other=View(gpu,owner,0,1,h,d,c);
    auto token=gpu.Upload(std::vector<float>(h*d,8.f),1,h*d),slot=gpu.Upload({2.f},1,1),active=gpu.Upload({1.f},1,1);
    auto blank=gpu.Allocate(4,width);Reject([&]{View(gpu,blank,0,1,h,d,c);},"Prepared an uninitialized backing owner");
    for(auto pair:std::vector<std::pair<size_t,size_t>>{{4,1},{3,2},{0,0},{0,5}})Reject([&]{View(gpu,owner,pair.first,pair.second,h,d,c);},"Invalid row span accepted");
    Reject([&]{View(gpu,owner,0,1,h,d+1,c);},"Mismatched cache layout accepted");
    auto nested=View(gpu,b,1,1,h,d,c);Equal(gpu.Download(nested),gpu.Download(a),"Nested span offset differs");
    gpu.Begin();Reject([&]{View(gpu,owner,0,1,h,d,c);},"Created row descriptor during execution");
    gpu.EncodeKVChunk(token,a,slot,active,a,h,d,c,1,true);
    Reject([&]{gpu.Download(owner);},"Parent readable during span write");Reject([&]{gpu.Download(other);},"Sibling readable during span write");
    MetalContext second;second.Begin();Reject([&]{second.EncodeKVChunk(token,other,slot,active,other,h,d,c,1,true);},"Foreign command consumed pending backing storage");second.Abort();
    // Same-command aliases share one pending identity, including nested views.
    gpu.EncodeKVChunk(token,nested,slot,active,nested,h,d,c,1,true);gpu.SubmitAndWait();
    auto expected=original;for(size_t head=0;head<h;++head)std::fill_n(expected.begin()+2*width+(head*c+2)*d,d,8.f);Equal(gpu.Download(owner),expected,"Same-command span dependency changed cache");
    gpu.Begin();gpu.EncodeKVChunk(token,a,slot,active,a,h,d,c,1,true);gpu.Abort();
    for(const auto* value:{&owner,&a,&b,&other})Reject([&]{gpu.Download(*value);},"Abort left backing alias readable");
    Reject([&]{gpu.Update(a,std::vector<float>(width,0.f));},"Partial update recovered aborted whole owner");
    Reject([&]{gpu.Zero(a);},"Partial zero recovered aborted whole owner");
    auto healthy=gpu.Upload(original,4,width),healthySpan=View(gpu,healthy,2,1,h,d,c);
    gpu.Begin();Reject([&]{gpu.EncodeKVChunk(token,healthySpan,slot,active,a,h,d,c,1,true);},"Partial GPU output recovered aborted whole owner");gpu.Abort();
    Reject([&]{gpu.Download(owner);},"Rejected partial GPU recovery made parent readable");
    gpu.Update(owner,original);Equal(gpu.Download(a),std::vector<float>(width,3.f),"Whole owner recovery did not restore alias readiness");
    gpu.Begin();Reject([&]{gpu.EncodeKVUpdate(token,a,slot,active,other,h,d,c);},"Different-offset in-place cache alias accepted");gpu.Abort();
    owner=MetalMatrix();b=MetalMatrix();other=MetalMatrix();a=MetalMatrix();Equal(gpu.Download(nested),std::vector<float>(width,3.f),"Nested view did not retain backing owner");
}
#ifdef GARNET_METAL_CHUNK_ATTENTION_MPS
void Attention(MetalContext& gpu,size_t first,size_t rows,bool gated) {
    constexpr size_t h=4,kv=2,d=8,c=128,chunk=16,cacheWidth=kv*d*c;
    auto values=[](size_t n,float phase){std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=float(.2*std::sin(i*.131+phase));return v;};
    auto kc=values(4*cacheWidth,2.1f),vc=values(4*cacheWidth,3.2f),q=values(rows*chunk*h*d,.1f),k=values(rows*chunk*kv*d,.7f),v=values(k.size(),1.1f),gate=values(q.size(),4.3f);
    auto fullK=gpu.Upload(kc,4,cacheWidth),fullV=gpu.Upload(vc,4,cacheWidth);
    auto key=View(gpu,fullK,first,rows,kv,d,c),value=View(gpu,fullV,first,rows,kv,d,c);
    auto copyK=gpu.Upload(std::vector<float>(kc.begin()+first*cacheWidth,kc.begin()+(first+rows)*cacheWidth),rows,cacheWidth);
    auto copyV=gpu.Upload(std::vector<float>(vc.begin()+first*cacheWidth,vc.begin()+(first+rows)*cacheWidth),rows,cacheWidth);
    std::vector<float> slots(rows*chunk),flags(rows*chunk,1.f);for(size_t row=0;row<rows;++row)for(size_t t=0;t<chunk;++t){slots[row*chunk+t]=float(96+t);if(row+1==rows&&t+1==chunk)flags[row*chunk+t]=0;}
    std::vector<MetalMatrix> args={gpu.Upload(q,rows*chunk,h*d),gpu.Upload(k,rows*chunk,kv*d),gpu.Upload(v,rows*chunk,kv*d),key,value,gpu.Upload(slots,rows,chunk),gpu.Upload(flags,rows,chunk),gpu.Upload(gate,rows*chunk,h*d)};
    auto output=gpu.Allocate(rows*chunk,h*d),control=gpu.Allocate(rows*chunk,h*d);
    gpu.PrepareAttentionChunk(args,h,kv,d,c,chunk);const auto plans=gpu.ChunkAttentionMPSPlanCount(),scratch=gpu.ChunkAttentionMPSScratchBytes();
    gpu.Begin();gpu.EncodeAttentionChunk(args,output,h,kv,d,c,chunk,gated);gpu.SubmitAndWait();
    args[3]=copyK;args[4]=copyV;gpu.Begin();gpu.EncodeAttentionChunk(args,control,h,kv,d,c,chunk,gated);gpu.SubmitAndWait();
    auto actual=gpu.Download(output);Equal(actual,gpu.Download(control),"Nonzero-offset MPS cache view changed attention words");
    Equal(gpu.Download(fullK),kc,"Attention modified full key owner");Equal(gpu.Download(fullV),vc,"Attention modified full value owner");
    Need(gpu.ChunkAttentionMPSPlanCount()==plans&&gpu.ChunkAttentionMPSScratchBytes()==scratch,"Row span attention changed prepared plans/scratch");
    for(size_t row=0;row<rows;++row)for(size_t t:{size_t(0),size_t(8),size_t(14)})for(size_t head:{size_t(0),h-1}) {
        const size_t visible=97+t,kvh=head/(h/kv);std::vector<double> scores(visible);double maximum=-INFINITY;
        for(size_t s=0;s<visible;++s){double sum=0;for(size_t x=0;x<d;++x) {
            float keyValue=s<96?kc[((first+row)*kv*c+kvh*c+s)*d+x]:k[((row*chunk+s-96)*kv+kvh)*d+x];
            sum+=double(q[((row*chunk+t)*h+head)*d+x])*keyValue;
        }scores[s]=sum/std::sqrt(double(d));maximum=std::max(maximum,scores[s]);}
        double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
        for(size_t x:{size_t(0),d-1}){double expected=0;for(size_t s=0;s<visible;++s){
            float vv=s<96?vc[((first+row)*kv*c+kvh*c+s)*d+x]:v[((row*chunk+s-96)*kv+kvh)*d+x];expected+=scores[s]/denominator*vv;
        }const size_t index=((row*chunk+t)*h+head)*d+x;if(gated)expected/=1+std::exp(-double(gate[index]));
        Need(std::abs(double(actual[index])-expected)<=1e-5+2e-4*std::abs(expected),"Row span attention exceeds FP64 oracle");}
    }
}
#endif
}
int main(){try{
    size_t cases=0;
    {MetalContext gpu;
     for(size_t first=0;first<4;++first)for(size_t rows=1;rows<=4-first;++rows)for(bool gaps:{false,true}) {
        Cell(gpu,first,rows,2,3,16,4,gaps);++cases;
        Cell(gpu,first,rows,2,8,32,8,gaps);++cases;
     }
     Cell(gpu,3,1,8,128,4096,128,false);++cases;
     Readiness(gpu);
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
     LastHiddenRows(gpu);
#endif
#ifdef GARNET_METAL_CHUNK_ATTENTION_MPS
     for(size_t first=0;first<4;++first)for(size_t rows=1;rows<=4-first;++rows)for(bool gated:{false,true}){Attention(gpu,first,rows,gated);++cases;}
#endif
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Row-span test retained native owners");
    std::cout<<"prepared_kv_rows PASS cases="<<cases<<" complete_CPU_cache_parity=true readiness_and_abort=true owners=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
