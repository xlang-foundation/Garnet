// SPDX-License-Identifier: Apache-2.0
// One engine/process at a time. Compare fresh OFF/ON processes with identical
// buffers, calls and shape order; initialization/uploads/readback are untimed.
#include "metal_context.h"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
using Clock=std::chrono::steady_clock;
namespace {
void Need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
double Ms(Clock::time_point start,Clock::time_point end){return std::chrono::duration<double,std::milli>(end-start).count();}
std::vector<float> Values(size_t count,uint32_t seed,float scale) {
    std::vector<float> values(count);
    for(auto& value:values){seed=1664525u*seed+1013904223u;value=float(int((seed>>8)%65521)-32760)*scale;}
    return values;
}
std::string Digest(const std::vector<float>& values) {
    Need(values.size()*sizeof(float)<=UINT32_MAX,"Digest input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(values.data(),CC_LONG(values.size()*sizeof(float)),digest);
    std::ostringstream text;
    for(auto byte:digest)text<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(byte);
    return text.str();
}
const char* Names[]={"dense","single79","pair79_145","tail7","holes","zero"};
std::vector<float> Mask(size_t chunk,size_t mode) {
    std::vector<float> mask(2*chunk);
    for(size_t row=0;row<2;++row)for(size_t token=0;token<chunk;++token) {
        bool enabled=false;
        switch(mode) {
        case 0:enabled=true;break;
        case 1:enabled=row==0&&token<79;break;
        case 2:enabled=token<(row?145:79);break;
        case 3:enabled=row==0&&token<7;break;
        case 4:enabled=(row+token)%3!=1;break;
        case 5:break;
        }
        mask[row*chunk+token]=float(enabled);
    }
    return mask;
}
}
int main(int argc,char** argv){try{
    Need(argc==3,"Expected on/off and cohort number");
#ifdef GARNET_METAL_LINEAR_CHUNK_ACTIVE_ONLY
    constexpr bool activeOnly=true;
#else
    constexpr bool activeOnly=false;
#endif
    Need(std::string(argv[1])==(activeOnly?"on":"off"),"Wrong benchmark policy build");
    size_t cohort=std::stoul(argv[2]);Need(cohort>=1&&cohort<=3,"Expected cohort1..3");
    std::cout<<std::setprecision(17);
    const auto began=Clock::now();
    {
        MetalContext gpu;
        const auto memory=MetalContext::MemoryStats();
        std::cout<<"{\"event\":\"ready\",\"active_only\":"<<(activeOnly?"true":"false")<<",\"cohort\":"<<cohort
                 <<",\"device\":"<<std::quoted(gpu.DeviceName())<<",\"registry\":"<<memory.registry<<",\"init_ms\":"<<Ms(began,Clock::now())<<"}\n";
        size_t cells=0,measured=0;
        for(size_t chunk:{128,256})for(size_t mode=0;mode<6;++mode)for(bool recurrent:{false,true}) {
            const size_t batch=2,heads=48,width=128,qw=heads*128,vw=heads*width,pw=2*qw+vw+2*heads,channels=10240,kernel=4;
            const size_t inputWidth=recurrent?pw:channels,stateWidth=recurrent?qw*width:channels*kernel,outputWidth=recurrent?vw:channels;
            auto input=Values(batch*chunk*inputWidth,393,.00001f),initial=Values(batch*stateWidth,153,.000001f),mask=Mask(chunk,mode);
            if(recurrent)for(size_t row=0;row<batch;++row)for(size_t t=0;t<chunk;++t)for(size_t h=0;h<heads;++h){input[(row*chunk+t)*pw+2*qw+vw+h]=-.1f-float((h+t)%11)*.03f;input[(row*chunk+t)*pw+2*qw+vw+heads+h]=.2f+float((h+t)%7)*.1f;}
            auto data=gpu.Upload(input,batch*chunk,inputWidth),state=gpu.Upload(initial,batch,stateWidth),active=gpu.Upload(mask,batch,chunk);
            auto weights=gpu.Upload(Values(channels*kernel,49,.00001f),channels,kernel);
            auto output=gpu.Allocate(batch*chunk,outputWidth),next=gpu.Allocate(batch,stateWidth);
            const std::string cell=std::string(recurrent?"delta":"conv")+"-b2-t"+std::to_string(chunk)+"-"+Names[mode];
            for(size_t sample=0;sample<12;++sample) {
                const auto start=Clock::now();gpu.Begin();
                if(recurrent)gpu.EncodeGatedDeltaChunk(data,state,active,output,next,chunk,heads,128,width);
                else gpu.EncodeCausalConvChunk(data,weights,state,active,output,next,chunk,kernel);
                const auto encoded=Clock::now();gpu.SubmitAndWait();const auto completed=Clock::now();
                double gpuMs=gpu.LastGPUMs();Need(std::isfinite(gpuMs)&&gpuMs>0,"Missing GPU elapsed time");
                std::cout<<"{\"event\":\"sample\",\"cell\":"<<std::quoted(cell)<<",\"phase\":\""<<(sample<3?"warmup":"measured")
                         <<"\",\"sample\":"<<(sample<3?sample:sample-3)<<",\"gpu_ms\":"<<gpuMs<<",\"wall_ms\":"<<Ms(start,completed)<<",\"encode_ms\":"<<Ms(start,encoded)<<"}\n";
                if(sample>=3)++measured;
            }
            auto values=gpu.Download(output),finalState=gpu.Download(next);
            for(float value:values)Need(std::isfinite(value),"Nonfinite benchmark output");
            for(float value:finalState)Need(std::isfinite(value),"Nonfinite benchmark state");
            if(mode==5){for(float value:values)Need(value==0&&!std::signbit(value),"Inactive output not positive zero");Need(finalState==initial,"Inactive state changed");}
            std::cout<<"{\"event\":\"cell_complete\",\"cell\":"<<std::quoted(cell)<<",\"output_words\":"<<values.size()<<",\"state_words\":"<<finalState.size()
                     <<",\"output_sha256\":"<<std::quoted(Digest(values))<<",\"state_sha256\":"<<std::quoted(Digest(finalState))<<"}\n";
            ++cells;
        }
        Need(cells==24&&measured==216,"Benchmark coverage differs");
        std::cout<<"{\"event\":\"complete\",\"cells\":"<<cells<<",\"measured_samples\":"<<measured<<",\"scope\":\"Operator timings only, not model token speed\"}\n";
    }
    const auto owners=MetalContext::ProfileOwnerStats();
    Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Benchmark owners remain after close");
    std::cout<<"{\"event\":\"closed\",\"owners\":0}\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"linear chunk benchmark FAILED: "<<error.what()<<"\n";return 1;}}
