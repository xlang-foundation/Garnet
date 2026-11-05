// SPDX-License-Identifier: Apache-2.0
// Primitive row isolation; native session/serving lifecycle gates are separate.
#include "metal_context.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Garnet;
static void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void Rejected(F action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}
    Need(caught,"Invalid request-row operation was accepted");}
int main(){try{
    MetalContext gpu;std::vector<float> values(4*37);
    for(size_t row=0;row<4;++row)for(size_t col=0;col<37;++col)values[row*37+col]=float(100*row+col+1);
    auto matrix=gpu.Upload(values,4,37);
    const auto owned=gpu.ReadRow(matrix,1);
    Need(owned==std::vector<float>(values.begin()+37,values.begin()+74),"Row capture used wrong stride");
    gpu.ZeroRow(matrix,1);gpu.ZeroRow(matrix,3);
    auto actual=gpu.Download(matrix);
    for(size_t row=0;row<4;++row)for(size_t col=0;col<37;++col)
        Need(actual[row*37+col]==((row==1||row==3)?0:values[row*37+col]),"Row reset modified another request");
    auto saved=actual;
    // Restore an immutable snapshot to a different row after its source reset.
    gpu.WriteRow(matrix,3,owned);actual=gpu.Download(matrix);
    for(size_t row=0;row<4;++row)for(size_t col=0;col<37;++col)
        Need(actual[row*37+col]==(row==3?owned[col]:saved[row*37+col]),"Row restore modified another request");
    saved=actual;
    Rejected([&]{gpu.ReadRow(matrix,4);});
    Rejected([&]{gpu.WriteRow(matrix,4,owned);});
    Rejected([&]{gpu.WriteRow(matrix,0,std::vector<float>(36,1));});
    auto nonfinite=owned;nonfinite[36]=std::numeric_limits<float>::infinity();
    Rejected([&]{gpu.WriteRow(matrix,0,nonfinite);});
    Need(gpu.Download(matrix)==saved,"Invalid restore partially mutated state");
    Rejected([&]{gpu.ZeroRow(matrix,4);});Need(gpu.Download(matrix)==saved,"Invalid row reset mutated state");
    auto uninitialized=gpu.Allocate(4,37);
    Rejected([&]{gpu.ZeroRow(uninitialized,0);});
    Rejected([&]{gpu.ReadRow(uninitialized,0);});
    Rejected([&]{gpu.WriteRow(uninitialized,0,owned);});
    gpu.Begin();Rejected([&]{gpu.ZeroRow(matrix,0);});
    Rejected([&]{gpu.ReadRow(matrix,0);});Rejected([&]{gpu.WriteRow(matrix,0,owned);});gpu.Abort();
    Need(gpu.Download(matrix)==saved,"Rejected active-command reset mutated state");
    gpu.Update(matrix,values);gpu.ZeroRow(matrix,0);actual=gpu.Download(matrix);
    for(size_t i=0;i<actual.size();++i)Need(actual[i]==(i<37?0:values[i]),"Row reuse/recovery leaked state");
    std::vector<float> kv(4*2*5*3);
    for(size_t i=0;i<kv.size();++i)kv[i]=float(i+1);
    auto cache=gpu.Upload(kv,4,30);
    const auto prefix=gpu.ReadRowPrefixGroups(cache,1,2,5,3,3);
    Need(prefix.size()==18,"Prefix retained full-capacity state");
    for(size_t group=0;group<2;++group)for(size_t i=0;i<9;++i)
        Need(prefix[group*9+i]==kv[30+group*15+i],"Prefix capture mixed requests/groups");
    gpu.ZeroRow(cache,3);gpu.WriteRowPrefixGroups(cache,3,2,5,3,3,prefix);
    actual=gpu.Download(cache);
    for(size_t row=0;row<4;++row)for(size_t group=0;group<2;++group)for(size_t i=0;i<15;++i)
        Need(actual[row*30+group*15+i]==(row==3?(i<9?prefix[group*9+i]:0):kv[row*30+group*15+i]),
             "Prefix restore leaked state or modified another request");
    const auto retained=actual;
    Rejected([&]{gpu.ReadRowPrefixGroups(cache,4,2,5,3,3);});
    Rejected([&]{gpu.ReadRowPrefixGroups(cache,0,2,5,3,6);});
    Rejected([&]{gpu.ReadRowPrefixGroups(cache,0,SIZE_MAX,5,3,3);});
    Rejected([&]{gpu.WriteRowPrefixGroups(cache,0,2,5,3,3,std::vector<float>(17,1));});
    auto badPrefix=prefix;badPrefix.back()=std::numeric_limits<float>::quiet_NaN();
    Rejected([&]{gpu.WriteRowPrefixGroups(cache,0,2,5,3,3,badPrefix);});
    gpu.Begin();Rejected([&]{gpu.ReadRowPrefixGroups(cache,0,2,5,3,3);});
    Rejected([&]{gpu.WriteRowPrefixGroups(cache,0,2,5,3,3,prefix);});gpu.Abort();
    Need(gpu.Download(cache)==retained,"Rejected prefix operation mutated state");
    const auto beforeRelease=MetalContext::MemoryStats().allocated;
    {
        auto temporary=gpu.Upload(std::vector<float>(4*1024*1024,1.0f),4,1024*1024);
        // Session row reset validates every owner before its pooled write.
        // Standalone validation must not retain autoreleased buffer properties.
        for(size_t row=0;row<4;++row)gpu.ValidateCompletedRow(temporary,row);
    }
    const auto afterRelease=MetalContext::MemoryStats().allocated;
    std::cout<<"standalone_row_validation_release before="<<beforeRelease<<" after="<<afterRelease<<"\n";
    Need(afterRelease<=beforeRelease+1024*1024,"Standalone row validation retained released Metal storage");
    std::cout<<"request-row padded-stride isolation, bounds, readiness, active-command rejection, reuse and buffer release gates passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
