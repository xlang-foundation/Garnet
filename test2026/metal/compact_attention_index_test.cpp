// SPDX-License-Identifier: Apache-2.0
// Independent FP64 softmax/contraction, including old-cache -1 and latest writes.
#include "metal_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
static bool strictBitwise=false;
static void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void Case(MetalContext& gpu,size_t batch,size_t chunk,size_t capacity,bool gated,bool contiguous=false,size_t heads=4,size_t kvHeads=2,size_t dim=8) {
    const size_t rows=batch*chunk,qWidth=heads*dim,kvWidth=kvHeads*dim;
    std::vector<float> q(rows*qWidth),k(rows*kvWidth),v(k.size()),kc(batch*kvHeads*capacity*dim),vc(kc.size()),gate(q.size()),slots(rows),active(rows,1);
    auto fill=[](std::vector<float>& values,double phase){for(size_t i=0;i<values.size();++i)values[i]=float(std::sin(i*.131+phase)*.2);};
    fill(q,.1);fill(k,.7);fill(v,1.1);fill(kc,2.1);fill(vc,3.2);fill(gate,4.3);
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t) {
        // Duplicate/backward slots exercise latest-token selection; high slots
        // ensure the compact index is not accidentally storing a KV position.
        slots[row*chunk+t]=float(capacity-chunk-2+t-(!contiguous&&t>=3?2:0));
        if(!contiguous&&t==1&&row==batch-1)active[row*chunk+t]=0;
    }
    std::vector<MetalMatrix> inputs={gpu.Upload(q,rows,qWidth),gpu.Upload(k,rows,kvWidth),gpu.Upload(v,rows,kvWidth),
        gpu.Upload(kc,batch,kvHeads*capacity*dim),gpu.Upload(vc,batch,kvHeads*capacity*dim),
        gpu.Upload(slots,batch,chunk),gpu.Upload(active,batch,chunk),gpu.Upload(gate,rows,qWidth)};
    auto output=gpu.Allocate(rows,qWidth);
    gpu.PrepareAttentionChunk(inputs,heads,kvHeads,dim,capacity,chunk);
    gpu.Begin();gpu.EncodeAttentionChunk(inputs,output,heads,kvHeads,dim,capacity,chunk,gated);gpu.SubmitAndWait();
    const auto actual=gpu.Download(output);
    for(float value:actual)Need(std::isfinite(value),"Nonfinite chunk attention output");
#if defined(GARNET_METAL_CHUNK_PROBABILITY_REUSE)||defined(GARNET_METAL_CHUNK_ATTENTION_INDEX32)
    Need(gpu.LastChunkAttentionOptimizedDispatches()==1,"Attention candidate route was not exercised");
    auto reference=gpu.Allocate(rows,qWidth);
    gpu.Begin();gpu.EncodeAttentionChunk(inputs,reference,heads,kvHeads,dim,capacity,chunk,gated,true);gpu.SubmitAndWait();
    const auto original=gpu.Download(reference);
    Need(gpu.LastChunkAttentionOptimizedDispatches()==0,"Reference unexpectedly used the candidate attention route");
    if(std::memcmp(actual.data(),original.data(),actual.size()*sizeof(float))!=0) {
        size_t mismatches=0;double maximum=0;
        for(size_t i=0;i<actual.size();++i)if(actual[i]!=original[i]) {
            if(mismatches<8)std::cerr<<"attention parity index="<<i<<" candidate="<<std::hexfloat<<actual[i]<<" control="<<original[i]<<std::defaultfloat<<"\n";
            ++mismatches;maximum=std::max(maximum,std::abs(double(actual[i])-original[i]));
        }
        std::cerr<<"attention parity batch="<<batch<<" chunk="<<chunk<<" dim="<<dim<<" capacity="<<capacity<<" gated="<<gated<<" mismatches="<<mismatches<<" max_abs="<<maximum<<"\n";
    }
    // The original main library uses default fast math. Retain the strict
    // experiment explicitly; admission still requires the original FP64 bound
    // and the frozen model's full output-token/functional quality gates.
    for(size_t i=0;i<actual.size();++i)Need(std::isfinite(original[i])&&
        std::abs(double(actual[i])-original[i])<=1e-5+2e-4*std::abs(double(original[i])),
        "Attention candidate exceeds the original numerical error bound versus control");
    if(strictBitwise)Need(std::memcmp(actual.data(),original.data(),actual.size()*sizeof(float))==0,"Attention candidate differs bitwise from the original kernel");
#endif
    double maximumError=0;
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t)for(size_t head=0;head<heads;++head) {
        const size_t flat=row*chunk+t,kv=head/(heads/kvHeads);
        if(!active[flat]) {
            for(size_t d=0;d<dim;++d)Need(actual[flat*qWidth+head*dim+d]==0,"Inactive query row was not zero");
            continue;
        }
        // Retain the exhaustive small oracle; sample nine query/head pairs per
        // row at real27B dimensions while comparing every output to control.
        if(dim>8&&((t!=0&&t!=chunk/2&&t!=chunk-1)||(head!=0&&head!=heads/2&&head!=heads-1)))continue;
        const size_t visible=size_t(slots[flat])+1;
        std::vector<int> selected(visible,-1);std::vector<double> scores(visible);
        for(size_t s=0;s<visible;++s) {
            for(size_t prior=0;prior<=t;++prior)if(active[row*chunk+prior]&&slots[row*chunk+prior]==float(s))selected[s]=int(prior);
            double dot=0;
            for(size_t d=0;d<dim;++d) {
                const auto key=selected[s]<0?kc[((row*kvHeads+kv)*capacity+s)*dim+d]:k[(row*chunk+size_t(selected[s]))*kvWidth+kv*dim+d];
                dot+=double(q[flat*qWidth+head*dim+d])*key;
            }
            scores[s]=dot/std::sqrt(double(dim));
        }
        const double largest=*std::max_element(scores.begin(),scores.end());double sum=0;
        for(auto& value:scores){value=std::exp(value-largest);sum+=value;}
        for(size_t d=0;d<dim;++d) {
            double expected=0;
            for(size_t s=0;s<visible;++s) {
                const auto value=selected[s]<0?vc[((row*kvHeads+kv)*capacity+s)*dim+d]:v[(row*chunk+size_t(selected[s]))*kvWidth+kv*dim+d];
                expected+=scores[s]/sum*value;
            }
            const size_t index=flat*qWidth+head*dim+d;
            if(gated)expected/=1+std::exp(-double(gate[index]));
            const double error=std::abs(double(actual[index])-expected);maximumError=std::max(maximumError,error);
            Need(std::isfinite(actual[index])&&error<=1e-5+2e-4*std::abs(expected),"Compact index changed attention semantics");
        }
    }
    gpu.Begin();bool rejected=false;
    try{gpu.EncodeAttentionChunk(inputs,output,heads,kvHeads,dim,capacity,kMetalPrefillChunkLimit+1,gated);}
    catch(const std::invalid_argument&){rejected=true;}
    gpu.Abort();Need(rejected&&gpu.Download(output)==actual,"Chunk limit rejection changed attention output");
    std::cout<<"compact_attention capacity="<<capacity<<" batch="<<batch<<" chunk="<<chunk<<" gated="<<gated<<" contiguous="<<contiguous<<" heads="<<heads<<" dim="<<dim<<" index_bytes="<<kMetalAttentionIndexBytes<<" max_abs="<<maximumError<<"\n";
}
int main(int argc,char** argv){try{
    Need(argc==1||(argc==2&&std::string(argv[1])=="--strict-bitwise"),"Unknown attention test mode");
    strictBitwise=argc==2;
    MetalContext gpu;
    Case(gpu,2,8,64,true);Case(gpu,2,8,4096,true);Case(gpu,1,64,4096,false);
#ifdef GARNET_METAL_QWEN38_LARGE_PREFILL
    static_assert(kMetalAttentionIndexBytes==2&&kMetalPrefillChunkLimit==256,"Large chunks require the signed 16-bit profile");
    Case(gpu,1,128,4096,true,true);Case(gpu,1,256,4096,false);Case(gpu,2,256,512,true);
    Case(gpu,2,128,4096,true,true,24,4,256);Case(gpu,2,8,4096,false,false,24,4,256);
#endif
    bool rejected=false;auto out=gpu.Allocate(1,1);gpu.Begin();
    try{gpu.EncodeAttentionChunk({},out,4,2,8,4097,8,true);}catch(const std::invalid_argument&){rejected=true;}
    gpu.Abort();Need(rejected,"Unqualified attention capacity accepted");
    std::cout<<"compact signed index, causal overwrite, inactive rows and 4K FP64 reference gates passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
