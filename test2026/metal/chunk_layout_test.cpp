// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/metal/metal_context.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
namespace {
void Need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void Array(const std::vector<double>& xs){std::cout<<'[';for(size_t i=0;i<xs.size();++i){if(i)std::cout<<',';std::cout<<xs[i];}std::cout<<']';}
void Cell(MetalContext& gpu,size_t index,size_t rows,size_t columns,size_t outRows,size_t tokenWidth,
          unsigned scenario,bool gpuFlags,const std::filesystem::path& folder,int repetitions) {
    const size_t count=rows*columns,outColumns=count/outRows,flagColumns=tokenWidth?columns/tokenWidth:1;
    Need(count%outRows==0,"Invalid layout fixture");
    const uint32_t patterns[]={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x3f800000,0xbf800000};
    std::vector<float> values(count),flags(rows*flagColumns),expected(count);
    for(size_t i=0;i<count;++i){uint32_t bits=patterns[(i*7+3)%10];std::memcpy(&values[i],&bits,4);}
    for(size_t row=0;row<rows;++row)for(size_t t=0;t<flagColumns;++t) {
        const bool live=scenario==0?true:scenario==1?row==0:scenario==2?row+1==rows:scenario==3?(row+t)%3!=1:false;
        flags[row*flagColumns+t]=live?((row+t)%2?-2.f:1.f):((row+t)%2?-0.f:0.f);
    }
    expected=values;
    if(tokenWidth)for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)
        if(flags[row*flagColumns+col/tokenWidth]==0.f)expected[row*columns+col]=0.f;
    auto input=gpu.Upload(values,rows,columns),mask=gpu.Upload(flags,rows,flagColumns),output=gpu.Allocate(outRows,outColumns);
    auto pending=gpuFlags?gpu.Allocate(rows,flagColumns):MetalMatrix{};
    std::vector<double> device,wall;size_t dispatches=0;const int warmups=repetitions>1?3:0;
    for(int repeat=-warmups;repeat<repetitions;++repeat) {
        const auto start=std::chrono::steady_clock::now();gpu.Begin();
        if(gpuFlags)gpu.EncodeLayout(mask,mask,pending);
        gpu.EncodeLayout(input,gpuFlags?pending:mask,output,tokenWidth);gpu.SubmitAndWait();
        const auto finish=std::chrono::steady_clock::now();dispatches=gpu.LastLayoutVectorDispatches();
        size_t expectedDispatches=0;
#ifdef GARNET_METAL_LAYOUT_VECTOR4
        expectedDispatches=size_t(columns%4==0&&outColumns%4==0&&tokenWidth%4==0)+size_t(gpuFlags&&flagColumns%4==0);
#endif
        Need(dispatches==expectedDispatches,"Wrong vector/scalar layout route");
        if(repeat>=0){device.push_back(gpu.LastGPUMs());wall.push_back(std::chrono::duration<double,std::milli>(finish-start).count());}
    }
    auto actual=gpu.Download(output);
    Need(actual.size()==count&&!std::memcmp(actual.data(),expected.data(),count*4),"Layout differs from complete CPU bit oracle");
    auto original=gpu.Download(input);Need(!std::memcmp(original.data(),values.data(),count*4),"Layout changed input storage");
    if(!folder.empty()) {
        std::ofstream file(folder/("case-"+std::to_string(index)+".f32"),std::ios::binary);
        file.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(count*4));file.close();Need(bool(file),"Cannot retain complete layout output");
    }
    std::cout<<"{\"kind\":\"cell\",\"index\":"<<index<<",\"rows\":"<<rows<<",\"columns\":"<<columns<<",\"output_rows\":"<<outRows<<",\"output_columns\":"<<outColumns
        <<",\"token_width\":"<<tokenWidth<<",\"scenario\":"<<scenario<<",\"gpu_flags\":"<<(gpuFlags?"true":"false")<<",\"vector_dispatches\":"<<dispatches
        <<",\"cpu_reference_words\":"<<count<<",\"warmups\":"<<warmups<<",\"gpu_ms\":";Array(device);std::cout<<",\"wall_ms\":";Array(wall);std::cout<<"}\n"<<std::flush;
}
void Guards(MetalContext& gpu) {
    auto a=gpu.Upload(std::vector<float>(16,1.f),2,8),flags=gpu.Upload(std::vector<float>(4,1.f),2,2);
    gpu.Begin();bool alias=false;try{gpu.EncodeLayout(a,flags,a,4);}catch(const std::invalid_argument&){alias=true;}gpu.Abort();Need(alias,"Layout accepted input/output alias");
    auto blank=gpu.Allocate(2,8),out=gpu.Allocate(2,8);
    gpu.Begin();bool unreadable=false;try{gpu.EncodeLayout(blank,flags,out,4);}catch(const std::logic_error&){unreadable=true;}gpu.Abort();Need(unreadable,"Layout accepted uninitialized input");
    gpu.Begin();bool shape=false;try{gpu.EncodeLayout(a,flags,out,3);}catch(const std::invalid_argument&){shape=true;}gpu.Abort();Need(shape,"Layout accepted mismatched mask shape");
    gpu.Begin();gpu.EncodeLayout(a,flags,out,4);gpu.Abort();
    Need(gpu.LastLayoutVectorDispatches()==0,"Abort retained vector counter");
    bool aborted=false;try{gpu.Download(out);}catch(const std::logic_error&){aborted=true;}Need(aborted,"Aborted output remained readable");
}
}
int main(int argc,char** argv){try{
    const bool benchmark=argc==3&&std::string(argv[1])=="--benchmark";Need(argc==1||benchmark,"Usage: test [--benchmark NEW_DIRECTORY]");
    std::filesystem::path folder;if(benchmark){folder=argv[2];Need(std::filesystem::create_directory(folder),"Output folder must be new");}
    size_t index=0;
    {
        MetalContext gpu;bool enabled=false;
#ifdef GARNET_METAL_LAYOUT_VECTOR4
        enabled=true;
#endif
        std::cout<<std::setprecision(12)<<"{\"kind\":\"metadata\",\"device\":"<<std::quoted(gpu.DeviceName())<<",\"vector_compiled\":"<<(enabled?"true":"false")<<"}\n";
        for(unsigned scenario=0;scenario<5;++scenario) {
            if(benchmark) {
                Cell(gpu,index++,4,128*4096,512,4096,scenario,false,folder,9);
                Cell(gpu,index++,512,4096,4,0,scenario,false,folder,9);
                Cell(gpu,index++,4,128*1024,512,1024,scenario,false,folder,9);
            } else {
                Cell(gpu,index++,4,8*12,32,12,scenario,false,{},1);
                Cell(gpu,index++,32,12,4,0,scenario,false,{},1);
                Cell(gpu,index++,4,7*3,28,3,scenario,false,{},1);
                Cell(gpu,index++,4,12,16,0,scenario,false,{},1);
                Cell(gpu,index++,4,8*12,32,12,scenario,true,{},1);
            }
        }
        Cell(gpu,index++,benchmark?512:4,4100,benchmark?512:4,4100,3,false,folder,benchmark?9:1);
        Cell(gpu,index++,benchmark?512:4,3,benchmark?512:4,3,3,false,folder,benchmark?9:1);
        Cell(gpu,index++,4,128*4096,512,4096,3,true,folder,benchmark?9:1);
        Cell(gpu,index++,4,5*12,20,12,3,true,folder,benchmark?9:1); // Padded GPU-produced flag rows.
        if(!benchmark)Guards(gpu);
    }
    const auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"Layout test retained native owners");
    std::cout<<"{\"kind\":\"closed\",\"cases\":"<<index<<",\"owners\":0}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
