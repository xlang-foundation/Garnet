// SPDX-License-Identifier: Apache-2.0
// Prove complete attention writes before sharing their output allocations.
#include "metal_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;
namespace {
void Need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
void Cell(const char* label,size_t batch,size_t chunk,size_t heads,size_t kv,size_t dim,size_t capacity,
          size_t prefix,std::array<size_t,4> counts,int mutation,bool gated) {
    MetalContext gpu;const size_t rows=batch*chunk,width=heads*dim,cache=kv*capacity*dim;
    auto values=[](size_t n,double phase){std::vector<float> x(n);for(size_t i=0;i<n;++i)x[i]=float(.23*std::sin(i*.137+phase)+.17*std::cos(i*.071+phase));return x;};
    auto q=values(rows*width,.1),k=values(rows*kv*dim,.7),v=values(k.size(),1.3),kc=values(batch*cache,2.1),vc=values(kc.size(),3.4),gate=values(q.size(),4.7);
    std::vector<float> slots(rows),active(rows);
    for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t){active[b*chunk+t]=float(t<counts[b]);slots[b*chunk+t]=active[b*chunk+t]?float(prefix+t):0;}
    if(mutation==1)active[(batch-1)*chunk+2]=0;
    if(mutation==2){slots[8]=slots[6];slots[10]=slots[7];}
    std::vector<MetalMatrix> args={gpu.Upload(q,rows,width),gpu.Upload(k,rows,kv*dim),gpu.Upload(v,rows,kv*dim),
        gpu.Upload(kc,batch,cache),gpu.Upload(vc,batch,cache),gpu.Upload(slots,batch,chunk),gpu.Upload(active,batch,chunk),gpu.Upload(gate,rows,width)};
    auto output=gpu.Upload(std::vector<float>(q.size(),std::numeric_limits<float>::quiet_NaN()),rows,width);
    auto copied=gpu.Allocate(batch,chunk),zeros=gpu.Upload(std::vector<float>(rows,0),batch,chunk);
    gpu.PrepareAttentionChunk(args,heads,kv,dim,capacity,chunk);
    const auto plans=gpu.ChunkAttentionMPSPlanCount(),scratch=gpu.ChunkAttentionMPSScratchBytes();
    const auto owners=MetalContext::ProfileOwnerStats();
    std::vector<float> previous;size_t checked=0,inactive=0;double maximumError=0;
    for(size_t repeat=0;repeat<2;++repeat) {
        gpu.Update(output,std::vector<float>(q.size(),repeat?73.5f:std::numeric_limits<float>::quiet_NaN()));
        gpu.Begin();auto inputs=args;if(mutation==3){gpu.EncodeAdd(args[5],zeros,copied);inputs[5]=copied;}
        gpu.EncodeAttentionChunk(inputs,output,heads,kv,dim,capacity,chunk,gated);gpu.SubmitAndWait();
        auto actual=gpu.Download(output);Need(actual.size()==q.size(),"Attention output extent changed");
        if(repeat)Need(std::memcmp(actual.data(),previous.data(),actual.size()*sizeof(float))==0,"Attention retained poison or prior output");
        previous=actual;
        for(size_t i=0;i<actual.size();++i){Need(std::isfinite(actual[i]),"Attention did not overwrite every poisoned word");if(!active[i/width]){Need(actual[i]==0,"Inactive attention output retained scratch");++inactive;}}
        // Independent FP64 samples include old-cache, live overlay, GQA,
        // changed token positions, gating and zero-masked physical owners.
        for(size_t sample=0;sample<128;++sample) {
            const size_t index=(sample*104729+19)%actual.size(),flat=index/width,b=flat/chunk,t=flat%chunk,h=(index%width)/dim,d=index%dim,kh=h/(heads/kv);
            if(!active[flat])continue;
            const size_t visible=size_t(slots[flat])+1;std::vector<double> scores(visible),valuesAt(visible);double maximum=-INFINITY;
            for(size_t pos=0;pos<visible;++pos) {
                int latest=-1;for(size_t token=0;token<=t;++token)if(active[b*chunk+token]&&slots[b*chunk+token]==float(pos))latest=int(token);
                double score=0;for(size_t col=0;col<dim;++col) {
                    const double key=latest<0?kc[b*cache+(kh*capacity+pos)*dim+col]:k[(b*chunk+size_t(latest))*kv*dim+kh*dim+col];
                    score+=double(q[flat*width+h*dim+col])*key;
                }
                scores[pos]=score/std::sqrt(double(dim));maximum=std::max(maximum,scores[pos]);
                valuesAt[pos]=latest<0?vc[b*cache+(kh*capacity+pos)*dim+d]:v[(b*chunk+size_t(latest))*kv*dim+kh*dim+d];
            }
            double denominator=0,sum=0;for(size_t pos=0;pos<visible;++pos){double probability=std::exp(scores[pos]-maximum);denominator+=probability;sum+=probability*valuesAt[pos];}
            double expected=sum/denominator;if(gated)expected/=1+std::exp(-double(gate[index]));
            const double fraction=std::abs(double(actual[index])-expected)/(1e-5+2e-4*std::abs(expected));Need(fraction<=1,"Attention full-write output exceeds independent FP64 bound");maximumError=std::max(maximumError,fraction);++checked;
        }
    }
    const auto after=MetalContext::ProfileOwnerStats();Need(owners.matrices==after.matrices&&owners.affineWeights==after.affineWeights&&owners.pipelineSets==after.pipelineSets&&plans==gpu.ChunkAttentionMPSPlanCount()&&scratch==gpu.ChunkAttentionMPSScratchBytes(),"Attention dispatch changed prepared native owners");
    auto oldK=gpu.Download(args[3]),oldV=gpu.Download(args[4]);Need(oldK==kc&&oldV==vc,"Read-only attention changed KV");
    std::cout<<"attention_full_write "<<label<<" words="<<2*q.size()<<" inactive_zero_words="<<inactive<<" fp64_samples="<<checked<<" maximum_error_fraction="<<maximumError<<" PASS\n";
}
}
int main(){try{
    Cell("owner512-initial",1,512,32,8,128,4096,0,{512,0,0,0},0,false);
    Cell("owner512-cached",1,512,32,8,128,4096,3072,{512,0,0,0},0,false);
    Cell("pair256-ragged",2,256,32,8,128,4096,3072,{256,173,0,0},0,false);
    Cell("four128-gapped",4,128,32,8,128,4096,1920,{128,0,7,91},1,false);
    Cell("scalar-tail",4,8,4,2,8,4096,3000,{1,0,2,0},0,true);
    Cell("all-inactive",4,16,4,2,8,512,0,{0,0,0,0},0,true);
    Cell("overwrite",2,16,4,2,8,512,384,{16,13,0,0},2,true);
    Cell("pending-controls",2,16,4,2,8,512,384,{16,13,0,0},3,true);
    Cell("small-gated",2,16,4,2,8,512,384,{16,9,0,0},0,true);
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Full-write test retained owners");
    std::cout<<"attention_full_write PASS cases=9 poison_variants=2 all_owners_closed=true\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
