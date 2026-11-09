// SPDX-License-Identifier: Apache-2.0
// Tests the actual generic provider; no model runner and no CPython.
#include "streamed_bf16_projection.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
static void Check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
static uint16_t B(float v){uint32_t u;std::memcpy(&u,&v,4);return uint16_t((u+32767+((u>>16)&1))>>16);}
static float F(uint16_t v){uint32_t u=uint32_t(v)<<16;float f;std::memcpy(&f,&u,4);return f;}
struct Buffers {
    void* x=nullptr;void* y=nullptr;void* workspace=nullptr;cudaStream_t stream=nullptr;
    Buffers(){Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));Check(cudaMalloc(&x,16*16*2));Check(cudaMalloc(&y,16*16*2));Check(cudaMalloc(&workspace,32ULL<<20));}
    ~Buffers(){if(stream)cudaStreamSynchronize(stream);if(x)cudaFree(x);if(y)cudaFree(y);if(workspace)cudaFree(workspace);if(stream)cudaStreamDestroy(stream);}
};
int main(int argc,char** argv){try {
    if(argc<2||argc>5)throw std::runtime_error("streamed_projection OUTPUT_DIR [GPU] [--check-second-device] [--registered]");
    const int device=argc>2?std::stoi(argv[2]):0;Check(cudaSetDevice(device));
    bool registered=false,checkSecond=false,budgetFailure=false;
    for(int argument=3;argument<argc;++argument){
        const std::string option(argv[argument]);if(option=="--registered")registered=true;
        else if(option=="--require-budget-failure"){registered=true;budgetFailure=true;}
        else if(option=="--check-second-device")checkSecond=true;else throw std::runtime_error("unknown test option");
    }
    const auto directory=std::filesystem::absolute(argv[1]);std::filesystem::create_directories(directory);
    const auto file=directory/"weights.safetensors";nlohmann::json header;
    std::vector<uint16_t> weights(512);
    for(int layer=0;layer<2;++layer){
        for(int i=0;i<256;++i)weights[layer*256+i]=B(float((i*7+layer*3)%17-8)/32);
        header["layer."+std::to_string(layer)+".weight"]={{"dtype","BF16"},{"shape",{16,16}},{"data_offsets",{layer*512,(layer+1)*512}}};
    }
    std::string encoded=header.dump();encoded.append((8-encoded.size()%8)%8,' ');const uint64_t bytes=encoded.size();
    {std::ofstream output(file,std::ios::binary);output.write(reinterpret_cast<const char*>(&bytes),8);output.write(encoded.data(),encoded.size());output.write(reinterpret_cast<const char*>(weights.data()),weights.size()*2);output.close();if(!output)throw std::runtime_error("fixture write failed");}
    Garnet::SafeTensorsIndex index;std::string error;if(!index.Open(file,error))throw std::runtime_error(error);
    auto a=Garnet::StreamedBF16ProjectionSpec::FromMetadata(*index.Find("layer.0.weight"));
    auto b=Garnet::StreamedBF16ProjectionSpec::Parse(Garnet::StreamedBF16ProjectionSpec::FromMetadata(*index.Find("layer.1.weight")).Serialize());
    a.registeredSource=b.registeredSource=registered;
    if(budgetFailure){
        std::string message;
        {Garnet::StreamedBF16Projection projection(a);Buffers buffers;
            try{projection.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);}
            catch(const std::exception& exception){message=exception.what();}
        }
        const auto final=nlohmann::json::parse(Garnet::StreamedBF16ProjectionStats());
        if(message.find("budget")==std::string::npos||final.at("uploaded_weight_bytes")!=0||
            final.at("active_device_pools")!=0||final.at("mapped_files")!=0||final.at("registered_checkpoint_bytes")!=0)
            throw std::runtime_error("registered host-memory budget failed to reject before upload/allocation");
        std::cout<<nlohmann::json{{"passed",true},{"budget_rejected_before_upload",true},{"cleanup",final}}.dump(2)<<std::endl;return 0;
    }
    {
        Garnet::StreamedBF16Projection first(a),second(b);
        Garnet::SafeTensorsMappedFile mapping;if(!mapping.Open(index.Find("layer.0.weight")->filePath,error))throw std::runtime_error(error);
        std::atomic<int> failures{0};
        auto worker=[&](int rank,Garnet::StreamedBF16Projection& projection,const Garnet::StreamedBF16ProjectionSpec& spec){try {
            Check(cudaSetDevice(device));Buffers buffers;
            const auto* weights=static_cast<const uint16_t*>(mapping.DataAt(spec.offset,spec.bytes));
            for(int iteration=0;iteration<128;++iteration){
                const int rows=4+iteration%9;std::vector<uint16_t> x(rows*16),y(rows*16);
                for(size_t i=0;i<x.size();++i)x[i]=B(float(int((i*7+iteration+rank)%31)-15)/128);
                Check(cudaMemcpyAsync(buffers.x,x.data(),x.size()*2,cudaMemcpyHostToDevice,buffers.stream));
                projection.Run(buffers.x,buffers.y,rows,buffers.stream,buffers.workspace,32ULL<<20);
                Check(cudaMemcpyAsync(y.data(),buffers.y,y.size()*2,cudaMemcpyDeviceToHost,buffers.stream));Check(cudaStreamSynchronize(buffers.stream));
                for(int r=0;r<rows;++r)for(int c=0;c<16;++c){float sum=0;for(int k=0;k<16;++k)sum+=F(x[r*16+k])*F(weights[c*16+k]);if(y[r*16+c]!=B(sum))throw std::runtime_error("concurrent scalar mismatch");}
            }
        }catch(const std::exception& exception){std::cerr<<exception.what()<<std::endl;++failures;}};
        std::thread one(worker,0,std::ref(first),std::cref(a)),two(worker,1,std::ref(second),std::cref(b));one.join();two.join();
        if(failures)throw std::runtime_error("concurrent provider test failed");
        Buffers buffers;
        const auto before=nlohmann::json::parse(Garnet::StreamedBF16ProjectionStats()).at("uploaded_weight_bytes");
        Check(cudaStreamBeginCapture(buffers.stream,cudaStreamCaptureModeThreadLocal));bool rejected=false;
        try{first.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);}catch(...){rejected=true;}
        cudaGraph_t graph=nullptr;Check(cudaStreamEndCapture(buffers.stream,&graph));if(graph)Check(cudaGraphDestroy(graph));
        if(!rejected||before!=nlohmann::json::parse(Garnet::StreamedBF16ProjectionStats()).at("uploaded_weight_bytes"))throw std::runtime_error("capture mutated provider state");
        auto stale=a;++stale.modified;Garnet::StreamedBF16Projection changed(stale);rejected=false;
        try{changed.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);}catch(...){rejected=true;}
        if(!rejected)throw std::runtime_error("stale checkpoint identity accepted");
        auto invalid=a;invalid.offset=invalid.fileBytes;rejected=false;try{invalid.Validate();}catch(...){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid checkpoint range accepted");
        int devices=0;if(checkSecond)Check(cudaGetDeviceCount(&devices));if(devices>1){
            Check(cudaSetDevice((device+1)%devices));rejected=false;
            try{first.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);}catch(...){rejected=true;}
            Check(cudaSetDevice(device));if(!rejected)throw std::runtime_error("device migration accepted");
        }
        const auto active=nlohmann::json::parse(Garnet::StreamedBF16ProjectionStats());
        if(active.at("active_device_pools")!=1||active.at("mapped_files")!=1)throw std::runtime_error("provider ownership accounting mismatch");
        if(registered&&active.at("registered_checkpoint_bytes")!=std::filesystem::file_size(file))throw std::runtime_error("registered checkpoint accounting mismatch");
    }
    {
        // Release one source while its DMA is queued and another source still
        // owns the same GPU pool. The mapping must outlive its own upload.
        const auto other=directory/"weights-other.safetensors";
        std::filesystem::copy_file(file,other,std::filesystem::copy_options::overwrite_existing);
        Garnet::SafeTensorsIndex otherIndex;if(!otherIndex.Open(other,error))throw std::runtime_error(error);
        auto spec=Garnet::StreamedBF16ProjectionSpec::FromMetadata(*otherIndex.Find("layer.0.weight"));spec.registeredSource=registered;
        Garnet::StreamedBF16Projection keepalive(spec);Buffers buffers;
        std::vector<uint16_t> input(64,B(0.25f)),actual(64);
        Check(cudaMemcpyAsync(buffers.x,input.data(),128,cudaMemcpyHostToDevice,buffers.stream));
        keepalive.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);
        {Garnet::StreamedBF16Projection temporary(a);temporary.Run(buffers.x,buffers.y,4,buffers.stream,buffers.workspace,32ULL<<20);}
        Check(cudaMemcpyAsync(actual.data(),buffers.y,128,cudaMemcpyDeviceToHost,buffers.stream));Check(cudaStreamSynchronize(buffers.stream));
        for(int r=0;r<4;++r)for(int c=0;c<16;++c){float sum=0;for(int k=0;k<16;++k)sum+=0.25f*F(weights[c*16+k]);if(actual[r*16+c]!=B(sum))throw std::runtime_error("source retirement corrupted asynchronous result");}
    }
    const auto final=nlohmann::json::parse(Garnet::StreamedBF16ProjectionStats());
    if(final.at("active_device_pools")!=0||final.at("mapped_files")!=0||final.at("registered_checkpoint_bytes")!=0)throw std::runtime_error("provider resources leaked");
    std::cout<<nlohmann::json{{"passed",true},{"independent_scalar_projections",256},{"concurrent_streams",2},
        {"capture_rejected_before_upload",true},{"source_identity_rejected",true},{"range_rejected",true},{"cleanup",final}}.dump(2)<<std::endl;
    return 0;
}catch(const std::exception& exception){std::cerr<<exception.what()<<std::endl;return 1;}}
