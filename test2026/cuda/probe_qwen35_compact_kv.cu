// SPDX-License-Identifier: Apache-2.0
// Unwired whole-buffer copy-only corpus. No model or speed admission.
#include "src/cuda/qwen35_compact_kv.cuh"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
using namespace Garnet;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    void* owner=nullptr;void* p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){check(cudaMalloc(&owner,n+512));p=static_cast<char*>(owner)+256;check(cudaMemset(owner,0x6d,n+512));}
    ~Buffer(){if(owner)cudaFree(owner);}
    template<class T>void put(const std::vector<T>& v){if(v.size()*sizeof(T)!=bytes)throw std::runtime_error("Upload size");check(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(){std::vector<T> v(bytes/sizeof(T));check(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));return v;}
    void guard(){std::array<unsigned char,256>a{},b{};check(cudaMemcpy(a.data(),owner,256,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),static_cast<char*>(p)+bytes,256,cudaMemcpyDeviceToHost));for(auto v:a)if(v!=0x6d)throw std::runtime_error("Leading guard changed");for(auto v:b)if(v!=0x6d)throw std::runtime_error("Trailing guard changed");}
};
template<class T>void exact(Buffer& b,const std::vector<T>& expected){auto actual=b.get<T>();if(actual.size()!=expected.size()||std::memcmp(actual.data(),expected.data(),b.bytes))throw std::runtime_error("Whole-buffer parity failed");}
}
int main(){try{
    check(cudaSetDevice(0));cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));int cases=0,negatives=0;
    std::cout<<json{{"event","compact_kv_metadata"},{"scope","Unwired copy-only pack/scatter. Whole buffers, guard poisoning, mixed prefixes, original FP32 bytes. No TensorRT ownership/rollback/serving integration, model quality, allocations or timing admission."}}.dump()<<std::endl;
    for(int kv:{2,4})for(int batch:{1,2,4,8})for(int capacity:{384,512})for(int tokens:{4,64}){
        Qwen35CompactKVConfig c{batch,kv,capacity,tokens,256};size_t full=size_t(batch)*kv*capacity*256,small=size_t(batch)*kv*tokens*256;
        Buffer key(full*4),value(full*4),outK(full*4),outV(full*4),dk(small*4),dv(small*4),slots(size_t(batch)*tokens*4),active(size_t(batch)*tokens*4),error(4);
        std::vector<float> oldK(full),oldV(full);for(size_t i=0;i<full;++i){oldK[i]=float(int(i%1024)-512)/64;oldV[i]=-oldK[i]-16;}
        for(int pattern=0;pattern<6;++pattern){
            std::vector<float> currentK=oldK,currentV=oldV,expectedK=oldK,expectedV=oldV,compactK(small,0),compactV(small,0),s(size_t(batch)*tokens),a(s.size());
            Qwen35CompactKVSpans spans{};std::array<int,8> submitted{};
            for(int b=0;b<batch;++b){int start=pattern==4?(capacity-tokens)/2:pattern==5?17:pattern%2?capacity-tokens:0;submitted[b]=pattern==3?(b*7+1)%(tokens+1):tokens;int count=pattern==0?0:pattern==1?submitted[b]:pattern==5?(b%2?tokens:0):((b+1)*3)%(submitted[b]+1);spans.start[b]=start;spans.count[b]=count;
                for(int t=0;t<tokens;++t){s[b*tokens+t]=float(start+(t<submitted[b]?t:submitted[b]));a[b*tokens+t]=float(t<submitted[b]);}
                for(int h=0;h<kv;++h)for(int t=0;t<submitted[b];++t)for(int d=0;d<256;++d){size_t j=((size_t(b)*kv+h)*capacity+start+t)*256+d;size_t q=((size_t(b)*kv+h)*tokens+t)*256+d;currentK[j]=128+float((t*256+d)%1024)/64;currentV[j]=-currentK[j]-32;compactK[q]=currentK[j];compactV[q]=currentV[j];if(t<count){expectedK[j]=currentK[j];expectedV[j]=currentV[j];}}
            }
            key.put(currentK);value.put(currentV);outK.put(oldK);outV.put(oldV);slots.put(s);active.put(a);error.put(std::vector<int>{0});
            check(cudaMemsetAsync(dk.p,0x3c,dk.bytes,stream));check(cudaMemsetAsync(dv.p,0x3c,dv.bytes,stream));
            check(PackQwen35CompactKV(c,(float*)key.p,(float*)value.p,(float*)slots.p,(float*)active.p,(float*)dk.p,(float*)dv.p,(int*)error.p,stream));check(cudaStreamSynchronize(stream));
            exact(error,std::vector<int>{0});exact(dk,compactK);exact(dv,compactV);
            check(CommitQwen35CompactKV(c,spans,(float*)dk.p,(float*)dv.p,(float*)outK.p,(float*)outV.p,stream));check(cudaStreamSynchronize(stream));
            exact(outK,expectedK);exact(outV,expectedV);exact(key,currentK);exact(value,currentV);exact(slots,s);exact(active,a);exact(dk,compactK);exact(dv,compactV);
            for(auto* b:{&key,&value,&outK,&outV,&dk,&dv,&slots,&active,&error})b->guard();++cases;
            std::cout<<json{{"event","case"},{"batch",batch},{"kv",kv},{"capacity",capacity},{"tokens",tokens},{"pattern",pattern},{"full_words_per_cache",full},{"compact_words_per_cache",small},{"all_bytes_and_guards_exact",true}}.dump()<<std::endl;
        }
        Qwen35CompactKVSpans bad{};bad.start[0]=-1;if(CommitQwen35CompactKV(c,bad,(float*)dk.p,(float*)dv.p,(float*)outK.p,(float*)outV.p,stream)!=cudaErrorInvalidValue)throw std::runtime_error("Negative span accepted");++negatives;
        bad.start[0]=capacity;bad.count[0]=1;if(CommitQwen35CompactKV(c,bad,(float*)dk.p,(float*)dv.p,(float*)outK.p,(float*)outV.p,stream)!=cudaErrorInvalidValue)throw std::runtime_error("Past-capacity span accepted");++negatives;
        bad.start[0]=0;bad.count[0]=tokens+1;if(CommitQwen35CompactKV(c,bad,(float*)dk.p,(float*)dv.p,(float*)outK.p,(float*)outV.p,stream)!=cudaErrorInvalidValue)throw std::runtime_error("Past-chunk span accepted");++negatives;
        auto invalid=c;invalid.batch=3;if(PackQwen35CompactKV(invalid,(float*)key.p,(float*)value.p,(float*)slots.p,(float*)active.p,(float*)dk.p,(float*)dv.p,(int*)error.p,stream)!=cudaErrorInvalidValue)throw std::runtime_error("Invalid batch accepted");++negatives;
        // GPU metadata errors must remain bounded and visible to the caller.
        std::vector<float> s(size_t(batch)*tokens,0),a(s.size(),1);s[0]=-1;slots.put(s);active.put(a);error.put(std::vector<int>{0});
        check(PackQwen35CompactKV(c,(float*)key.p,(float*)value.p,(float*)slots.p,(float*)active.p,(float*)dk.p,(float*)dv.p,(int*)error.p,stream));check(cudaStreamSynchronize(stream));exact(error,std::vector<int>{1});++negatives;
        for(auto* b:{&key,&value,&outK,&outV,&dk,&dv,&slots,&active,&error})b->guard();
    }
    std::cout<<json{{"event","complete"},{"cases",cases},{"negative_checks",negatives},{"all_passed",true},{"primary_performance_eligible",false},{"serving_integrated",false}}.dump()<<std::endl;
    check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
