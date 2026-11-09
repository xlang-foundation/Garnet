// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <vector>
#include <limits>
#include <iostream>
using namespace Garnet;
void Need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class F>void Reject(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}Need(yes,"Expected dense view guard");}
int main(){try{{MetalContext gpu;std::vector<float> original(1024);for(size_t i=0;i<1024;++i)original[i]=float(i);auto owner=gpu.Upload(original,16,64);auto one=gpu.PrepareDiagnosticDenseView(owner,0,4,128),two=gpu.PrepareDiagnosticDenseView(owner,512,4,128);Need(one.SharesStorage(owner)&&two.SharesStorage(owner),"Lost parent ownership");
Reject([&]{gpu.PrepareDiagnosticDenseView(owner,1024,1,4);});Reject([&]{gpu.PrepareDiagnosticDenseView(owner,1,1,4);});Reject([&]{gpu.PrepareDiagnosticDenseView(owner,0,0,4);});Reject([&]{gpu.PrepareDiagnosticDenseView(owner,0,8193,4);});Reject([&]{gpu.PrepareDiagnosticDenseView(owner,0,1,std::numeric_limits<size_t>::max());});
auto a=gpu.Upload(std::vector<float>(512,1.f),4,128),b=gpu.Upload(std::vector<float>(512,2.f),4,128);gpu.Begin();Reject([&]{gpu.PrepareDiagnosticDenseView(owner,0,4,128);});gpu.EncodeAdd(a,b,one);gpu.EncodeAdd(a,a,two);gpu.SubmitAndWait();auto values=gpu.Download(owner);for(size_t i=0;i<1024;++i)Need(values[i]==(i<512?3.f:2.f),"Dense span writer or offset wrong");
gpu.Begin();gpu.EncodeAdd(a,b,one);gpu.Abort();Reject([&]{gpu.Download(owner);});Reject([&]{gpu.Update(two,std::vector<float>(512,1.f));});gpu.Update(owner,original);Need(gpu.Download(one)==std::vector<float>(original.begin(),original.begin()+512),"Parent restore did not restore views");
gpu.Begin();gpu.EncodeAdd(a,b,one);gpu.Flush();Reject([&]{gpu.PrepareDiagnosticDenseView(owner,0,4,128);});gpu.SubmitAndWait();}
auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.pipelineSets,"Dense spans retain owners");std::cout<<"Dense bulk views PASS; complete1024 words, shared readiness, abort/full-parent restore, bounds and pending preparation guards\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
