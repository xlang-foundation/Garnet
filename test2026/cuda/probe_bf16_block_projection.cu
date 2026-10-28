// SPDX-License-Identifier: Apache-2.0
// Unwired component: exact BF16-bit decoding, then the original production GEMM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunBF16Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::vector<unsigned char> bytes(const std::string& path){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);require(bool(stream),"Missing fixture");
    size_t size=size_t(stream.tellg());stream.seekg(0);std::vector<unsigned char> result(size);
    stream.read(reinterpret_cast<char*>(result.data()),size);require(bool(stream),"Fixture read failed");return result;
}
struct Buffer {
    unsigned char* owner=nullptr;unsigned char* data=nullptr;size_t size=0;
    explicit Buffer(size_t n):size(n){check(cudaMalloc(&owner,n+512));data=owner+256;check(cudaMemset(owner,0xa5,n+512));check(cudaDeviceSynchronize());}
    ~Buffer(){if(owner)cudaFree(owner);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    bool guards()const{
        std::array<unsigned char,512> host{};check(cudaMemcpy(host.data(),owner,256,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host.data()+256,data+size,256,cudaMemcpyDeviceToHost));
        return std::all_of(host.begin(),host.end(),[](unsigned char v){return v==0xa5;});
    }
};
template<bool Streaming> __device__ unsigned char loadByte(const unsigned char* p){
    if constexpr(Streaming)return static_cast<unsigned char>(__ldcs(reinterpret_cast<const char*>(p)));
    else return *p;
}
template<bool Streaming> __device__ unsigned short loadWord(const unsigned char* p){return unsigned(loadByte<Streaming>(p))|(unsigned(loadByte<Streaming>(p+1))<<8);}
template<bool Streaming> __device__ unsigned short unpack(unsigned code,unsigned base,const unsigned char* payload){
    unsigned delta=code&15;
    if(delta==15)return loadWord<Streaming>(payload+384+2*(code>>4));
    return ((code&0x800)<<4)|((base-delta)<<7)|((code>>4)&127);
}
template<bool Streaming> __global__ void decodeBlocks(const unsigned char* blob,unsigned short* output,size_t words){
    size_t block=blockIdx.x,first=block*256,pair=threadIdx.x*2;
    size_t blocks=(words+255)/256;const auto* header=blob+block*8;
    unsigned offset=*reinterpret_cast<const unsigned*>(header),base=header[4],mode=header[5];
    unsigned count=unsigned(header[6])|(unsigned(header[7])<<8);
    if(pair>=count||first+pair>=words)return;
    const auto* payload=blob+blocks*8+offset;unsigned short low,high=0;
    if(mode==0){low=loadWord<Streaming>(payload+pair*2);if(pair+1<count)high=loadWord<Streaming>(payload+(pair+1)*2);}
    else {
        const auto* p=payload+(pair/2)*3;
        unsigned packed=unsigned(loadByte<Streaming>(p))|(unsigned(loadByte<Streaming>(p+1))<<8)|(unsigned(loadByte<Streaming>(p+2))<<16);
        low=unpack<Streaming>(packed&4095,base,payload);high=unpack<Streaming>(packed>>12,base,payload);
    }
    if(pair+1<count&&first+pair+1<words)
        reinterpret_cast<unsigned*>(output)[(first+pair)/2]=unsigned(low)|(unsigned(high)<<16);
    else output[first+pair]=low;
}
void decode(Buffer& encoded,Buffer& stage,size_t words,cudaStream_t stream,bool streaming){
    if(streaming)decodeBlocks<true><<<unsigned((words+255)/256),128,0,stream>>>(encoded.data,reinterpret_cast<unsigned short*>(stage.data),words);
    else decodeBlocks<false><<<unsigned((words+255)/256),128,0,stream>>>(encoded.data,reinterpret_cast<unsigned short*>(stage.data),words);
    check(cudaGetLastError());
}
__global__ void initialize(__nv_bfloat16* p,size_t count){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=count)return;
    unsigned h=unsigned(index)*747796405u+2891336453u;h=((h>>((h>>28)+4))^h)*277803737u;h=(h>>22)^h;
    p[index]=__float2bfloat16_rn((float(h%257)-128.f)/1024.f);
}
__global__ void evict(unsigned* p,size_t count,unsigned round){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index<count)p[index]=p[index]+round+1;
}
struct Captured {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~Captured(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
std::unique_ptr<Captured> capture(cudaStream_t stream,const std::function<void()>& body){
    auto result=std::make_unique<Captured>();check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    try{body();}catch(...){cudaStreamEndCapture(stream,&result->graph);throw;}
    check(cudaStreamEndCapture(stream,&result->graph));check(cudaGraphInstantiateWithFlags(&result->executable,result->graph,0));return result;
}
double median(std::vector<double> values){require(!values.empty(),"Missing timing samples");std::sort(values.begin(),values.end());size_t n=values.size();return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;}
struct Timer {
    cudaEvent_t begin=nullptr,ready=nullptr,end=nullptr;
    Timer(){check(cudaEventCreate(&begin));check(cudaEventCreate(&ready));check(cudaEventCreate(&end));}
    ~Timer(){cudaEventDestroy(begin);cudaEventDestroy(ready);cudaEventDestroy(end);}
    std::array<double,3> run(Captured& graph,cudaStream_t stream,Buffer& eviction,bool cold,unsigned round){
        check(cudaEventRecord(begin,stream));
        if(cold)evict<<<unsigned((eviction.size/4+255)/256),256,0,stream>>>(reinterpret_cast<unsigned*>(eviction.data),eviction.size/4,round);
        check(cudaEventRecord(ready,stream));check(cudaGraphLaunch(graph.executable,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float evictionMs=0,graphMs=0,totalMs=0;check(cudaEventElapsedTime(&evictionMs,begin,ready));
        check(cudaEventElapsedTime(&graphMs,ready,end));check(cudaEventElapsedTime(&totalMs,begin,end));
        return {graphMs*1000.,evictionMs*1000.,totalMs*1000.};
    }
};
std::uint64_t differentWords(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b){
    require(a.size()==b.size()&&a.size()%2==0,"Word comparison size mismatch");std::uint64_t count=0;
    for(size_t i=0;i<a.size();i+=2)count+=(a[i]!=b[i]||a[i+1]!=b[i+1]);return count;
}
std::vector<unsigned char> download(const Buffer& value){std::vector<unsigned char> result(value.size);check(cudaMemcpy(result.data(),value.data,value.size,cudaMemcpyDeviceToHost));return result;}
int main(int argc,char** argv){try{
    require(argc>=3,"Require fixture manifest and unused outputJSONL path");
    bool quick=false,codecOnly=false;for(int i=3;i<argc;++i){std::string flag=argv[i];if(flag=="--quick")quick=true;else if(flag=="--codec-only")codecOnly=true;else throw std::runtime_error("Unknown probe flag");}
    require(!std::ifstream(argv[2]).good(),"Preserve component output");
    std::ifstream manifestStream(argv[1]);json manifest;manifestStream>>manifest;
    std::ofstream log(argv[2]);require(bool(log),"Component output unavailable");
    check(cudaSetDevice(0));cudaDeviceProp property{};check(cudaGetDeviceProperties(&property,0));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(property.uuid.bytes[i]));}
    log<<json{{"event","metadata"},{"device",property.name},{"gpu_uuid",uuid.str()},{"uploads_ordered_on_test_stream",true},
        {"total_global_bytes",property.totalGlobalMem},{"l2_cache_bytes",property.l2CacheSize},
        {"multiprocessors",property.multiProcessorCount},{"codec_only",codecOnly},{"quick",quick},
        {"scope","Unwired canonical prevalidated byte fixtures only. Same production BF16 GEMM source; no model/VRAM/inference claim. Cold event timing excludes deliberate cache eviction; eviction/total times also retained. No significance or exact runtime-cache analogy."}}.dump()<<std::endl;
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=codecOnly?std::shared_ptr<void>{}:Garnet::CreateBF16ProjectionContext();require(codecOnly||bool(context),"Production GEMM context failed");
    size_t cases=0,codecCases=0;
    for(const auto& fixture:manifest.at("fixtures")){
        std::string name=fixture.at("name");auto raw=bytes(fixture.at("raw_file"));auto packed=bytes(fixture.at("packed_file"));
        size_t words=fixture.at("words");require(raw.size()==words*2&&packed.size()==fixture.at("packed_bytes").get<size_t>(),"Fixture length changed");
        Buffer weight(raw.size()),encoded(packed.size()),stage(raw.size());
        check(cudaMemcpyAsync(weight.data,raw.data(),raw.size(),cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(encoded.data,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
        for(bool streaming:{false,true}){
            decode(encoded,stage,words,stream,streaming);check(cudaStreamSynchronize(stream));
            auto result=download(stage);auto mismatches=differentWords(raw,result);
            bool guarded=stage.guards()&&encoded.guards()&&weight.guards();json first=json::array();
            if(mismatches)for(size_t j=0;j<raw.size()&&first.size()<8;j+=2)if(raw[j]!=result[j]||raw[j+1]!=result[j+1])
                first.push_back({{"word",j/2},{"expected",unsigned(raw[j])|(unsigned(raw[j+1])<<8)},{"actual",unsigned(result[j])|(unsigned(result[j+1])<<8)}});
            log<<json{{"event","codec"},{"fixture",name},{"streaming_reads",streaming},{"words",words},{"different_words",mismatches},{"first_mismatches",first},{"guards_passed",guarded}}.dump()<<std::endl;
            require(mismatches==0&&guarded,"GPU bit decoder/guard mismatch");
            ++codecCases;
        }
        int inner=fixture.at("inner"),columns=fixture.at("columns");
        if(codecOnly||!inner)continue;
        require(size_t(inner)*columns==words,"Learned matrix shape mismatch");
        size_t evictionBytes=std::max(size_t(256)<<20,size_t(property.l2CacheSize)*4);
        Buffer input(size_t(512)*inner*2),eviction(evictionBytes);check(cudaMemsetAsync(eviction.data,0,eviction.size,stream));
        initialize<<<unsigned((input.size/2+255)/256),256,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(input.data),input.size/2);check(cudaGetLastError());check(cudaStreamSynchronize(stream));
        std::vector<int> shapes=quick?std::vector<int>{4,32,512}:std::vector<int>{4,8,16,32,64,128,256,512};
        for(int rows:shapes){
            Buffer ordinary(size_t(rows)*columns*2),candidate(size_t(rows)*columns*2);
            auto gemm=[&](const void* w,void* out){require(Garnet::RunBF16Projection(context,input.data,w,out,rows,inner,columns,stream,nullptr,0)==0,"Production GEMM failed");};
            gemm(weight.data,ordinary.data);decode(encoded,stage,words,stream,false);gemm(stage.data,candidate.data);check(cudaStreamSynchronize(stream));
            auto expected=download(ordinary);require(differentWords(expected,download(candidate))==0,"Initial stage/GEMM output differs");
            std::array<std::unique_ptr<Captured>,3> graphs;
            graphs[0]=capture(stream,[&]{gemm(weight.data,ordinary.data);});
            graphs[1]=capture(stream,[&]{decode(encoded,stage,words,stream,false);gemm(stage.data,candidate.data);});
            graphs[2]=capture(stream,[&]{decode(encoded,stage,words,stream,true);gemm(stage.data,candidate.data);});
            int repeats=quick?3:31;Timer timer;
            std::array<std::vector<double>,3> cold,warm,evictions,total;
            for(int round=0;round<repeats;++round){
                for(int pass=0;pass<3;++pass){int mode=round%2?2-pass:pass;auto t=timer.run(*graphs[mode],stream,eviction,true,unsigned(round));cold[mode].push_back(t[0]);evictions[mode].push_back(t[1]);total[mode].push_back(t[2]);}
            }
            for(int mode=0;mode<3;++mode){
                for(int i=0;i<5;++i)check(cudaGraphLaunch(graphs[mode]->executable,stream));check(cudaStreamSynchronize(stream));
                for(int i=0;i<repeats;++i)warm[mode].push_back(timer.run(*graphs[mode],stream,eviction,false,unsigned(i))[0]);
                auto current=download(mode?candidate:ordinary);require(differentWords(expected,current)==0,"Captured stage/GEMM output differs");
                require(ordinary.guards()&&candidate.guards()&&stage.guards()&&encoded.guards()&&weight.guards()&&input.guards()&&eviction.guards(),"Projection canary changed");
                log<<json{{"event","projection"},{"fixture",name},{"rows",rows},{"inner",inner},{"columns",columns},
                    {"variant",mode==0?"ordinary":mode==1?"stage":"stage_streaming"},{"source_weight_bytes",raw.size()},
                    {"encoded_weight_bytes",packed.size()},{"temporary_stage_bytes",mode?raw.size():0},
                    {"cold_median_us",median(cold[mode])},{"warm_median_us",median(warm[mode])},
                    {"eviction_median_us",median(evictions[mode])},{"cold_total_with_eviction_median_us",median(total[mode])},
                    {"cold_samples_us",cold[mode]},{"warm_samples_us",warm[mode]},
                    {"all_output_words_bit_exact",true},{"guards_passed",true},{"timing_eligible",!quick}}.dump()<<std::endl;
            }
            require(differentWords(raw,download(stage))==0,"Staging weight bits changed after GEMMs");++cases;
        }
        std::cout<<"BF16_BLOCK_MATRIX_COMPLETE "<<name<<std::endl;
    }
    check(cudaStreamSynchronize(stream));context.reset();check(cudaStreamDestroy(stream));
    log<<json{{"event","complete"},{"codec_cases",codecCases},{"projection_profiles",cases},{"all_passed",true}}.dump()<<std::endl;
    std::cout<<"BF16_BLOCK_PROJECTION_COMPLETE "<<codecCases<<' '<<cases<<std::endl;return 0;
}catch(const std::exception& error){std::cerr<<"BF16_BLOCK_PROJECTION_FAILED "<<error.what()<<std::endl;return 1;}}
