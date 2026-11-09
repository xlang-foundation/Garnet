// Standalone arithmetic-policy pilot. No runtime or serving admission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nlohmann/json.hpp"
#include "src/model/float16_codec.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
using J=nlohmann::json;namespace fs=std::filesystem;
void Need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class T>std::vector<T> Read(fs::path p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);Need(bool(f)&&size_t(f.tellg())==n*sizeof(T),"Wrong trained input length");f.seekg(0);std::vector<T> v(n);f.read((char*)v.data(),n*sizeof(T));Need(bool(f),"Trained input read failed");return v;}
std::string Sha(const void* p,size_t n){Need(n<=UINT32_MAX,"Hash length overflow");unsigned char d[32];CC_SHA256(p,CC_LONG(n),d);std::ostringstream s;for(auto c:d)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return s.str();}
void Save(fs::path p,const J& j){auto tmp=p;tmp+=".tmp";std::ofstream f(tmp);f<<j.dump(2)<<'\n';f.close();fs::rename(tmp,p);}
struct Params{uint64_t rows,inner,columns,aStride,oStride;};
int main(int argc,char** argv){fs::path out;J report={{"completed",false},{"cells",J::array()},{"timing",J::array()},{"resources",J::array()}};
try{@autoreleasepool{
    Need(argc==5,"STUDY TRAINED-ASSETS TRAINED-CAPTURES FRESH-OUTPUT");fs::path root=argv[1],assets=argv[2],captures=argv[3];out=argv[4];Need(!fs::exists(out),"Never overwrite pilot output");fs::create_directory(out);
    auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"Metal device absent");const size_t baseline=device.currentAllocatedSize;report["device"]=device.name.UTF8String;report["baseline_allocated_bytes"]=baseline;
    {@autoreleasepool{
    auto queue=[device newCommandQueue];auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.fastMathEnabled=NO;NSError* error=nil;
    std::array<id<MTLLibrary>,2> libraries;
    for(int idx=0;idx<2;++idx){auto path=root/(idx?"indexed2.metal":"ordinary2.metal");auto text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()] encoding:NSUTF8StringEncoding error:&error];Need(text!=nil,"Missing composed MSL source");libraries[idx]=[device newLibraryWithSource:text options:options error:&error];if(!libraries[idx])throw std::runtime_error(error.localizedDescription.UTF8String);}
    using Key=std::tuple<bool,size_t,size_t,size_t,int>;std::map<Key,id<MTLComputePipelineState>> pipelines;
    for(auto shape:std::array<std::pair<size_t,size_t>,2>{{{4096,14336},{14336,4096}}})for(bool idx:{false,true})for(size_t rows:{1ul,4ul})for(int method:{0,1,4,99}){
        auto [k,n]=shape;auto c=[MTLFunctionConstantValues new];uint32_t count=uint32_t(rows),cols=uint32_t(method==4?4:1);uint32_t dims[]{uint32_t(k),uint32_t(n),uint32_t(k),uint32_t(n)};
        [c setConstantValue:&count type:MTLDataTypeUInt atIndex:0];for(NSUInteger i=0;i<4;++i)[c setConstantValue:dims+i type:MTLDataTypeUInt atIndex:10+i];if(method)[c setConstantValue:&cols type:MTLDataTypeUInt atIndex:21];
        std::string name=method==99?"centered_input_sums":method?"centered_q4_group64":rows==4?"affine_q4_blocked":idx?"affine_q4_indexed_decode":"affine_q4_decode";
        auto f=[libraries[idx] newFunctionWithName:[NSString stringWithUTF8String:name.c_str()] constantValues:c error:&error];if(!f)throw std::runtime_error(error.localizedDescription.UTF8String);auto p=[device newComputePipelineStateWithFunction:f error:&error];if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);
        Need(p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength,"Pipeline exceeds qualified resources");pipelines[{idx,k,n,rows,method}]=p;
        report["resources"].push_back({{"indexed",idx},{"inner",k},{"columns",n},{"rows",rows},{"method",method},{"max_threads",p.maxTotalThreadsPerThreadgroup},{"threadgroup_bytes",p.staticThreadgroupMemoryLength}});
    }
    report["all_pipelines_prepared_before_collection"]=true;size_t numericFailures=0,coefficientFailures=0,batchFailures=0,parentChecks=0,oracleWords=0;constexpr uint32_t guard=0x1937abef;
    for(bool down:{false,true}){@autoreleasepool{
        const size_t k=down?14336:4096,n=down?4096:14336,groups=k/64;const std::string name=down?"down":"gate";
        auto asset=assets/"layer-15",capture=captures/"layer-15";
        auto weight=Read<uint32_t>(asset/(name+"-weight.u32"),n*k/8);auto scales=Read<uint16_t>(asset/(name+"-scales.f16"),n*groups),biases=Read<uint16_t>(asset/(name+"-biases.f16"),n*groups);auto captured=Read<float>(capture/(down?"silu.f32":"norm-input.f32"),512*k);
        std::vector<uint32_t> xWords(32+4*k,guard);for(size_t row=0;row<4;++row)std::memcpy(xWords.data()+16+row*k,captured.data()+std::array<size_t,4>{0,127,256,511}[row]*k,k*4);
        const float* x=reinterpret_cast<const float*>(xWords.data()+16);std::vector<float> sums(4*groups);
        for(size_t r=0;r<4;++r)for(size_t g=0;g<groups;++g){float sum=0;for(size_t j=0;j<64;++j)sum+=x[r*k+g*64+j];sums[r*groups+g]=sum;}
        for(size_t c=0;c<n;++c)for(size_t g=0;g<groups;++g){float s=Garnet::Float16ToFloat32(scales[c*groups+g]),b=Garnet::Float16ToFloat32(biases[c*groups+g]);float zf=s!=0?-b/s:-1;int z=zf>=0&&zf<=15?int(std::nearbyint(zf)):0;float residual=std::fma(float(z),s,b);for(int q=0;q<16;++q){float a=std::fma(float(q),s,b),v=std::fma(float(q-z),s,residual);if(std::memcmp(&a,&v,4))++coefficientFailures;}}
        // Full independent original-weight FP64 oracle, shared by all variants.
        std::vector<double> oracle(4*n),absolute(4*n);
        for(size_t c=0;c<n;++c){std::array<double,4> value{},sumAbs{};for(size_t j=0;j<k;++j){const size_t ci=c*groups+j/64;float s=Garnet::Float16ToFloat32(scales[ci]),b=Garnet::Float16ToFloat32(biases[ci]);float w=std::fma(float((weight[c*(k/8)+j/8]>>((j%8)*4))&15),s,b);for(size_t r=0;r<4;++r){double p=double(x[r*k+j])*double(w);value[r]+=p;sumAbs[r]+=std::abs(p);}}for(size_t r=0;r<4;++r){oracle[r*n+c]=value[r];absolute[r*n+c]=sumAbs[r];}}
        auto upload=[&](const auto& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*sizeof(v[0]) options:MTLResourceStorageModeShared];Need(b!=nil,"Buffer upload failed");return b;};auto xb=upload(xWords),wb=upload(weight),sb=upload(scales),bb=upload(biases);auto y=[device newBufferWithLength:(32+4*n)*4 options:MTLResourceStorageModeShared];auto sumsBuffer=[device newBufferWithLength:(32+4*groups)*4 options:MTLResourceStorageModeShared];Need(y&&sumsBuffer,"Output allocation failed");
        const auto inputSha=Sha(xb.contents,xb.length),weightSha=Sha(wb.contents,wb.length),scaleSha=Sha(sb.contents,sb.length),biasSha=Sha(bb.contents,bb.length);
        auto encode=[&](bool idx,size_t rows,int method,std::array<uint32_t,4> selected){
            Params params{rows,k,n,k,n};auto cb=[queue commandBuffer];
            auto bind=[&](id<MTLComputeCommandEncoder> e){[e setBuffer:xb offset:64 atIndex:0];[e setBuffer:wb offset:0 atIndex:1];[e setBuffer:sb offset:0 atIndex:2];[e setBuffer:bb offset:0 atIndex:3];[e setBuffer:y offset:64 atIndex:4];[e setBytes:&params length:sizeof(params) atIndex:5];if(idx)[e setBytes:selected.data() length:16 atIndex:6];if(method)[e setBuffer:sumsBuffer offset:64 atIndex:7];};
            if(method){auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipelines.at({idx,k,n,rows,99})];bind(e);[e dispatchThreads:MTLSizeMake(rows*groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
            auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipelines.at({idx,k,n,rows,method})];bind(e);size_t width=method?4*size_t(method):rows==4?16:4;[e dispatchThreadgroups:MTLSizeMake((n+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];Need(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");return cb;
        };
        auto clear=[&](){std::fill_n(static_cast<uint32_t*>(y.contents),y.length/4,guard);std::fill_n(static_cast<uint32_t*>(sumsBuffer.contents),sumsBuffer.length/4,guard);};
        std::map<std::pair<size_t,int>,std::vector<uint32_t>> known;std::map<int,std::vector<uint32_t>> full;
        for(size_t rows:{4ul,1ul})for(int method:{0,1,4}){
            clear();encode(false,rows,method,{0,1,2,3});std::vector<uint32_t> bits(y.length/4);std::memcpy(bits.data(),y.contents,y.length);size_t mismatch=0;double maxFraction=0;
            for(size_t i=0;i<bits.size();++i){bool active=i>=16&&i<16+rows*n;if(!active)Need(bits[i]==guard,"Projection parent guard changed");else{const size_t offset=i-16;float v;std::memcpy(&v,bits.data()+i,4);double fraction=std::abs(double(v)-oracle[offset])/(2e-6+4e-6*absolute[offset]);maxFraction=std::max(maxFraction,fraction);if(!std::isfinite(v)||fraction>1)++numericFailures;++oracleWords;if(method&&bits[i]!=known.at({rows,0})[i])++mismatch;}}
            if(method){const auto* raw=static_cast<const uint32_t*>(sumsBuffer.contents);for(size_t i=0;i<sumsBuffer.length/4;++i){bool active=i>=16&&i<16+rows*groups;if(active)Need(!std::memcmp(raw+i,sums.data()+i-16,4),"Shared group sum differs from sequential FP32 oracle");else Need(raw[i]==guard,"Shared-sum parent guard changed");}}
            if(rows==4)full[method]=bits;else if(method)for(size_t c=0;c<n;++c)if(bits[16+c]!=full[method][16+c])++batchFailures;
            if(method){std::ifstream previous(root/"pilot-output1/result1.json");J old;previous>>old;bool found=false;for(const auto& cell:old["cells"])if(cell["shape"]==name&&cell["rows"]==rows&&cell["method"]==method){Need(cell["output_sha256"]==Sha(bits.data(),bits.size()*4),"Vector loads changed arithmetic-policy output bits");found=true;}Need(found,"Missing prototype output anchor");}
            known[{rows,method}]=bits;report["cells"].push_back({{"shape",name},{"rows",rows},{"method",method},{"old_fp32_nonidentical_words",mismatch},{"maximum_original_fp64_fraction",maxFraction},{"output_sha256",Sha(bits.data(),bits.size()*4)}});parentChecks+=bits.size();
        }
        // Independently select each physical row at occupancy1, preserving new
        // arithmetic-policy fixed1-vs-batch identity at every output column.
        for(int method:{1,4})for(uint32_t row=0;row<4;++row){clear();encode(true,1,method,{row,0,0,0});auto* bits=static_cast<const uint32_t*>(y.contents);for(size_t i=0;i<y.length/4;++i){bool active=i>=16+row*n&&i<16+(row+1)*n;if(active){if(bits[i]!=full[method][i])++batchFailures;}else Need(bits[i]==guard,"Selected-row inactive parent changed");}parentChecks+=y.length/4;}
        Need(Sha(xb.contents,xb.length)==inputSha&&Sha(wb.contents,wb.length)==weightSha&&Sha(sb.contents,sb.length)==scaleSha&&Sha(bb.contents,bb.length)==biasSha,"Read-only source changed");
        report["numerical_failures"]=numericFailures;report["coefficient_reconstruction_failures"]=coefficientFailures;report["fixed1_batch_word_failures"]=batchFailures;Save(out/"result1.json",report);
        // Pilot command time includes shared input-sum preparation. Every
        // measured output must remain bit-identical to its checked candidate.
        if(numericFailures==0&&coefficientFailures==0&&batchFailures==0)for(size_t rows:{1ul,4ul})for(int round=-5;round<21;++round)for(int order=0;order<3;++order){int at=(order+round+5)%3;if((round+5)&1)at=2-at;int method=std::array<int,3>{0,1,4}[at];clear();auto cb=encode(false,rows,method,{0,1,2,3});const double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000;Need(std::isfinite(ms)&&ms>0,"Invalid GPU duration");Need(!std::memcmp(y.contents,known.at({rows,method}).data(),y.length),"Timed projection drifted");if(round>=0)report["timing"].push_back({{"shape",name},{"rows",rows},{"method",method},{"round",round},{"position",order},{"gpu_ms",ms}});}
        std::cout<<J{{"shape",name},{"numeric_failures",numericFailures},{"coefficient_failures",coefficientFailures},{"batch_word_failures",batchFailures},{"timing_samples",report["timing"].size()}}.dump()<<'\n'<<std::flush;
        report["oracle_words"]=oracleWords;report["parent_words_checked"]=parentChecks;Save(out/"result1.json",report);
    }}
    report["numerically_eligible_for_further_screening"]=numericFailures==0&&coefficientFailures==0&&batchFailures==0;
    }}
    report["close_allocated_bytes"]=device.currentAllocatedSize;Need(device.currentAllocatedSize<=baseline+4194304,"Standalone allocations retained");report["completed"]=true;report["no_runtime_integration"]=true;report["serving_speed_claim"]=false;report["goal_complete"]=false;Save(out/"result1.json",report);return 0;
}}catch(const std::exception& e){report["error"]=e.what();if(!out.empty()&&fs::exists(out))Save(out/"result1.json",report);std::cerr<<e.what()<<'\n';return 1;}}
