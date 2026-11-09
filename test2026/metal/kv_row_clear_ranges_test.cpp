// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/metal_context.h"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace Garnet;
#ifdef GARNET_METAL_PREPARED_KV_ROW_SPANS
namespace Garnet {
struct MetalPreparedKVRowsTestAccess {
    static MetalMatrix Span(MetalContext& gpu,const MetalMatrix& matrix) {
        return gpu.PrepareKVRowSpan(matrix,1,1,3,5,9);
    }
};
}
#endif
int main() {
    try {
        size_t checks=0,rejections=0;
        auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("GPU row-range clear witness differs");};
        MetalContext gpu;
        std::vector<float> av(3*3*9*5),bv(2*2*11*3);
        for(size_t i=0;i<av.size();++i)av[i]=float(i)+.25f;
        for(size_t i=0;i<bv.size();++i)bv[i]=float(i)+.75f;
        auto a=gpu.Upload(av,3,135),b=gpu.Upload(bv,2,66);
        std::vector<MetalRowClearRange> ranges{{a,1,3,9,5,2,6},{b,0,2,11,3,0,3}};
        check(gpu.ZeroRowRanges(ranges)==312);
        for(size_t g=0;g<3;++g)for(size_t t=2;t<6;++t)for(size_t d=0;d<5;++d)av[135+(g*9+t)*5+d]=0;
        for(size_t g=0;g<2;++g)for(size_t t=0;t<3;++t)for(size_t d=0;d<3;++d)bv[(g*11+t)*3+d]=0;
        check(gpu.Download(a)==av);check(gpu.Download(b)==bv);
        auto reject=[&](const std::vector<MetalRowClearRange>& bad) {
            bool failed=false;try{gpu.ZeroRowRanges(bad);}catch(const std::exception&){failed=true;}
            check(failed);check(gpu.Download(a)==av);check(gpu.Download(b)==bv);++rejections;
        };
        for(const auto& bad:std::vector<MetalRowClearRange>{
            {a,3,3,9,5,0,1},{a,0,0,9,5,0,1},{a,0,3,0,5,0,0},{a,0,3,9,0,0,1},
            {a,0,3,9,5,5,4},{a,0,3,9,5,0,10},{a,0,2,9,5,0,1},
            {a,0,std::numeric_limits<size_t>::max(),9,5,0,1},
            {a,0,1,std::numeric_limits<size_t>::max(),2,0,1},{MetalMatrix(),0,1,1,1,0,1}})
            reject({{a,0,3,9,5,0,1},bad});
        auto uninitialized=gpu.Allocate(1,4);
        reject({{a,0,3,9,5,0,1},{uninitialized,0,1,1,4,0,1}});
        gpu.Begin();bool failed=false;
        try{gpu.ZeroRowRanges(ranges);}catch(const std::exception&){failed=true;}
        gpu.Abort();check(failed);++rejections;check(gpu.Download(a)==av);
        check(gpu.ZeroRowRanges({})==0);check(gpu.ZeroRowRanges({{a,1,3,9,5,4,4}})==0);
        size_t cleared=312;
#ifdef GARNET_METAL_PREPARED_KV_ROW_SPANS
        auto span=MetalPreparedKVRowsTestAccess::Span(gpu,a);
        check(gpu.ZeroRowRanges({{span,0,3,9,5,8,9}})==60);
        for(size_t g=0;g<3;++g)for(size_t d=0;d<5;++d)av[135+(g*9+8)*5+d]=0;
        check(gpu.Download(a)==av);check(gpu.Download(b)==bv);
        cleared+=60;
#endif
        std::cout<<"GPU row-range clear PASS checks="<<checks<<" invalid_cases="<<rejections<<" logical_zero_bytes="<<cleared<<'\n';
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
