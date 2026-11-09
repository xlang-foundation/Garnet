// SPDX-License-Identifier: Apache-2.0
#include "mistral_n4_layout.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
void Need(bool value){if(!value)throw std::runtime_error("N4 layout policy check failed");}
template<class T>void PackCheck(size_t n,size_t width){
    std::vector<T> source(n*width),packed(n*width+32,T(0x735a));
    for(size_t i=0;i<source.size();i++)source[i]=T(i*2654435761u+37);
    MistralN4Pack(source.data(),packed.data()+16,n,width);
    for(size_t col=0;col<n;col++)for(size_t word=0;word<width;word++)Need(packed[16+((col/4)*width+word)*4+col%4]==source[col*width+word]);
    for(size_t i=0;i<16;i++)Need(packed[i]==T(0x735a)&&packed[16+n*width+i]==T(0x735a));
}
int main(){try{
 for(auto shape:std::array<std::array<size_t,2>,2>{{{4096,14336},{14336,4096}}}){
    auto k=shape[0],n=shape[1];for(size_t rows=0;rows<17;rows++)Need(MistralN4Profile(k,n,rows)==(rows==4));
    Need(MistralN4PayloadBytes(k,n)==33030144);PackCheck<uint32_t>(n,k/8);PackCheck<uint16_t>(n,k/64);
 }
 for(size_t k:{0ul,64ul,4096ul,14336ul,SIZE_MAX})for(size_t n:{0ul,1ul,1024ul,4096ul,14336ul,32768ul,SIZE_MAX}){
    bool expected=(k==4096&&n==14336)||(k==14336&&n==4096);Need(MistralN4Shape(k,n)==expected);
 }
 constexpr uint64_t bytes=33030144,limit=20ull*1024*1024*1024,reserve=256ull*1024*1024;
 Need(MistralN4MemoryFits(0,bytes,0,limit));Need(!MistralN4MemoryFits(kMistralN4CacheBudget,bytes,0,limit));
 Need(!MistralN4MemoryFits(0,UINT64_MAX,0,limit));Need(!MistralN4MemoryFits(0,bytes,limit-reserve,limit));
 Need(!MistralN4MemoryFits(0,bytes,0,reserve));Need(!MistralN4MemoryFits(UINT64_MAX,1,0,limit));
 Need(uint64_t(96)*bytes==3170893824ull&&uint64_t(96)*bytes<kMistralN4CacheBudget);
 bool rejected=false;std::array<uint32_t,8> v{};try{MistralN4Pack(v.data(),v.data()+4,1,1);}catch(const std::invalid_argument&){rejected=true;}Need(rejected);
 std::cout<<"N4 layout/memory policy PASS exact roundtrips and occupancy4 guard\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
