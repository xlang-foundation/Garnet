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
    std::cout<<"request-row padded-stride isolation, bounds, readiness, active-command rejection and reuse gates passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}}
