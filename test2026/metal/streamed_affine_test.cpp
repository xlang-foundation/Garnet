// SPDX-License-Identifier: Apache-2.0
// Bounded streaming operator qualification. Full saved outputs require the
// independent offline FP64 audit; this executable also samples FP64 directly.
#include "../../src/backends/metal/metal_context.h"
#include "../../src/model/affine_int4_codec.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
using namespace Garnet;using JSON=nlohmann::json;
void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class F> void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Need(rejected,"Expected guard rejection");}
template<class T> void Save(const std::filesystem::path& p,const std::vector<T>& data){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));Need(bool(f),"Cannot save payload");}
struct Fixture {
    MetalContext gpu;size_t m,n,k;AffineInt4Layout layout;
    std::vector<float> input,poison;std::vector<uint32_t> packed;std::vector<uint16_t> scales,biases;
    MetalAffineInt4Weight weight;MetalMatrix a,out;bool eligible;std::filesystem::path root;
    Fixture(size_t rows,size_t outputs,size_t inner,std::filesystem::path directory):m(rows),n(outputs),k(inner),layout(AffineInt4Layout::Create(n,k,AffineInt4CoefficientType::FP16)),root(directory) {
        input.resize(m*k);packed.resize(layout.weightBytes/4);scales.resize(layout.coefficientBytes/2);biases.resize(scales.size());
        uint32_t seed=37;for(auto& x:input){seed=1664525u*seed+1013904223u;x=float(int((seed>>8)%65521)-32760)*0x1p-18f;}
        for(size_t i=0;i<packed.size();++i){seed=1664525u*seed+1013904223u;packed[i]=seed;}
        for(size_t i=0;i<scales.size();++i){scales[i]=Float32ToFloat16RNE(float(int(i%13)-6)*.00171f);biases[i]=Float32ToFloat16RNE(float(int(i%17)-8)*.00313f);}
        weight=gpu.UploadAffineInt4(packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes,n,k,true);
        poison.assign(m*n,std::numeric_limits<float>::quiet_NaN());a=gpu.Upload(input,m,k);out=gpu.Upload(poison,m,n);
        Need(gpu.PrepareAffineInt4Profile(a,weight,out,m),"Cannot prepare affine controls");
        eligible=gpu.PrepareStreamedAffineProfile(a,weight,out,m);
        Need(eligible==(m==512&&((k==4096&&(n==4096||n==14336))||(k==14336&&n==4096))),"Wrong streamed preparation selection");
        const auto plans=gpu.StreamedAffinePlanCount();const auto owners=MetalContext::ProfileOwnerStats();
        Need(gpu.PrepareStreamedAffineProfile(a,weight,out,m)==eligible&&gpu.StreamedAffinePlanCount()==plans&&MetalContext::ProfileOwnerStats().matrices==owners.matrices,"Repeated preparation allocates");
        size_t bytes=0;std::set<uintptr_t> unique;for(const auto& scratch:gpu.StreamedAffineScratch()){auto info=MetalContext::MatrixBufferInfo(scratch);if(info.identity){bytes+=info.lengthBytes;unique.insert(info.identity);}}
        Need(bytes==(eligible?161480704u:0u)&&unique.size()==(eligible?3u:0u),"Scratch is not one bounded set");
        if(!root.empty()){Need(std::filesystem::create_directory(root),"Fresh case directory required");Save(root/"input.f32",input);Save(root/"weight.u32",packed);Save(root/"scales.f16",scales);Save(root/"biases.f16",biases);}
    }
    JSON Cell(size_t count,bool reversed,bool dense,bool benchmark) {
        std::vector<size_t> ids(count);std::vector<bool> active(m);for(size_t i=0;i<count;++i){ids[i]=dense?i:(i*127+5)%m;active[ids[i]]=true;}if(reversed)std::reverse(ids.begin(),ids.end());
        std::array<std::vector<float>,2> first;std::array<std::vector<double>,2> times;const size_t plans=gpu.StreamedAffinePlanCount(),pipelines=gpu.AffineProfilePipelineCount();auto owners=MetalContext::ProfileOwnerStats();size_t checks=0;double maximum=0;
        const int reps=benchmark?9:1,warmups=benchmark?3:0;
        for(int rep=-warmups;rep<reps;++rep)for(size_t step=0;step<2;++step){size_t v=(step+size_t(rep+warmups))%2;
            gpu.SetStreamedAffineEnabled(v!=0);gpu.Update(out,poison);gpu.Begin();
            if(dense)gpu.EncodeAffineInt4MatMul(a,weight,out);else gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,ids);
            gpu.SubmitAndWait();Need(gpu.LastStreamedAffineDispatches()==size_t(v&&eligible&&count==512),"Wrong actual streamed route");
            Need(gpu.LastAffineProfileMisses()==0&&gpu.LastAffineProfileDispatches()==size_t(count>0),"Native profile accounting differs");
            Need(gpu.LastIndexedAffineDispatches()==size_t(!dense&&count>0),"Indexed dispatch accounting differs");
            Need(gpu.StreamedAffinePlanCount()==plans&&gpu.AffineProfilePipelineCount()==pipelines,"Request-time plan creation");
            auto values=gpu.Download(out);for(size_t i=0;i<values.size();++i){Need(std::isfinite(values[i]),"Missing/nonfinite output after poison");if(!active[i/n])Need(values[i]==0,"Inactive output changed");}
            if(first[v].empty())first[v]=values;else Need(std::memcmp(first[v].data(),values.data(),values.size()*4)==0,"Nondeterministic projection");
            if(rep>=0)times[v].push_back(gpu.LastGPUMs());
        }
        AffineInt4View view(layout,packed.data(),layout.weightBytes,scales.data(),layout.coefficientBytes,biases.data(),layout.coefficientBytes);std::vector<float> decoded(k);
        for(size_t col:std::set<size_t>{0,1,15,16,31,32,63,64,n/2,n-1}){view.DecodeRow(col,decoded.data(),k);if(count)for(size_t row:std::set<size_t>{0,count/2,count-1}) {
            double sum=0,absolute=0;for(size_t j=0;j<k;++j){double product=double(Float16ToFloat32(Float32ToFloat16RNE(input[ids[row]*k+j])))*Float16ToFloat32(Float32ToFloat16RNE(decoded[j]));sum+=product;absolute+=std::abs(product);}
            for(size_t v=0;v<2;++v){double fraction=std::abs(double(first[v][ids[row]*n+col])-sum)/(2e-6+4e-6*absolute);Need(fraction<=1,"Original FP64 bound exceeded");maximum=std::max(maximum,fraction);++checks;}
        }}
        auto after=MetalContext::ProfileOwnerStats();Need(owners.matrices==after.matrices&&owners.affineWeights==after.affineWeights&&owners.pipelineSets==after.pipelineSets,"Runtime owner allocation");
        std::string label=std::to_string(count)+(reversed?"-reverse":"-perm")+(dense?"-dense":"-indexed");if(!root.empty()){Save(root/(label+"-control.f32"),first[0]);Save(root/(label+"-candidate.f32"),first[1]);}
        return JSON{{"rows",m},{"outputs",n},{"inner",k},{"active",ids},{"label",label},{"eligible",eligible},{"streamed",eligible&&count==512},{"dense",dense},{"fp64_samples",checks},{"max_fp64_fraction",maximum},{"control_gpu_ms",times[0]},{"candidate_gpu_ms",times[1]},{"warmups",warmups},{"repetitions",reps},{"checked_words",2*m*n*size_t(warmups+reps)},{"plans",plans}};
    }
    void Guards(){
        std::vector<size_t> ids(m);std::iota(ids.begin(),ids.end(),0);auto bad=ids;bad[1]=bad[0];
        gpu.SetStreamedAffineEnabled(true);gpu.Update(out,std::vector<float>(m*n,17.f));auto before=gpu.Download(out);
        gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,bad);});Reject([&]{gpu.SetStreamedAffineEnabled(false);});Reject([&]{gpu.PrepareStreamedAffineProfile(a,weight,out,m);});gpu.Abort();Need(gpu.Download(out)==before,"Invalid call wrote output");
        bad=ids;bad[1]=m;gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,bad);});gpu.Abort();Need(gpu.Download(out)==before,"Invalid index wrote output");
        auto unready=gpu.Allocate(m,k);gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(unready,weight,out,ids);});gpu.Abort();
        gpu.Begin();gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,ids);gpu.Abort();Reject([&]{gpu.Download(out);});
        gpu.Begin();gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,ids);gpu.SubmitAndWait();Need(gpu.LastStreamedAffineDispatches()==size_t(eligible),"Recovery route differs");
        if(eligible&&n==14336){auto privateOutput=gpu.StreamedAffineScratch()[2];gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(a,weight,privateOutput,ids);});gpu.Abort();}
        if(eligible&&k==14336){auto privateInput=gpu.StreamedAffineScratch()[2];gpu.Update(privateInput,input);gpu.Begin();Reject([&]{gpu.EncodeAffineInt4IndexedMatMul(privateInput,weight,out,ids);});gpu.Abort();}
        if(eligible){gpu.Begin();gpu.EncodeAffineInt4IndexedMatMulControl(a,weight,out,ids);gpu.SubmitAndWait();Need(gpu.LastStreamedAffineDispatches()==0&&gpu.LastReuse64AffineDispatches()==1,"Control API selected streamed candidate");}
        gpu.Begin();gpu.EncodeAffineInt4IndexedMatMul(a,weight,out,ids);gpu.Flush();Reject([&]{gpu.SetStreamedAffineEnabled(false);});Reject([&]{gpu.PrepareStreamedAffineProfile(a,weight,out,m);});gpu.SubmitAndWait();
    }
};
int main(int argc,char** argv){try{
    bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";Need(argc==1||benchmark,"Usage [--benchmark NEW_DIRECTORY]");std::filesystem::path root;if(benchmark){root=argv[2];Need(std::filesystem::create_directory(root),"Fresh benchmark directory required");}
    JSON report={{"scope","Native streamed FP16 prefill operator; no model/serving or quality qualification"},{"cases",JSON::array()}};
    size_t index=0;
    for(auto shape:std::vector<std::array<size_t,2>>{{14336,4096},{4096,14336},{4096,4096},{1024,4096}}){
        Fixture f(512,shape[0],shape[1],benchmark?root/("case-"+std::to_string(index)):std::filesystem::path{});
        for(size_t count:std::vector<size_t>{0,128,256,511,512})report["cases"].push_back(f.Cell(count,false,false,benchmark&&count==512));
        report["cases"].push_back(f.Cell(512,true,false,false));report["cases"].push_back(f.Cell(512,false,true,benchmark));f.Guards();
        std::cout<<"streamed_shape "<<index++<<" PASS\n"<<std::flush;
    }
    {Fixture f(256,4096,4096,benchmark?root/"case-4":std::filesystem::path{});report["cases"].push_back(f.Cell(256,false,true,false));}
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Retained native owners after close");report["passed"]=true;
    if(benchmark){std::ofstream f(root/"results.json");f<<report.dump(2)<<'\n';Need(bool(f),"Cannot save report");}else std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
