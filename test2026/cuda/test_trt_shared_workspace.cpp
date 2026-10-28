// SPDX-License-Identifier: Apache-2.0
// Standalone context-scratch ordering probe; no Garnet model integration.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace nvinfer1;
using json=nlohmann::json;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void cu(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Logger final:ILogger{
    void log(Severity severity,const char* message) noexcept override{
        if(severity<=Severity::kWARNING)std::cerr<<message<<'\n';
    }
};
static Logger logger;
struct Buffer{
    void* data=nullptr;size_t bytes=0;bool pinned=false;
    Buffer(size_t n,bool host=false):bytes(n),pinned(host){
        if(host)cu(cudaMallocHost(&data,n));else{cu(cudaMalloc(&data,n));cu(cudaMemset(data,0,n));cu(cudaDeviceSynchronize());}
    }
    ~Buffer(){if(data){if(pinned)cudaFreeHost(data);else cudaFree(data);}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
struct Stream{
    cudaStream_t value=nullptr;
    Stream(){cu(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
struct Engine{
    int m,k,n,o;std::vector<float> first,second;
    std::unique_ptr<IRuntime> runtime;std::unique_ptr<ICudaEngine> engine;
    Engine(int rows,int inner,int hidden,int outputs):m(rows),k(inner),n(hidden),o(outputs),first(k*n),second(n*o){
        for(size_t i=0;i<first.size();++i)first[i]=float(int((i*17)%31)-15)/128;
        for(size_t i=0;i<second.size();++i)second[i]=float(int((i*29)%23)-11)/128;
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));require(bool(builder),"builder");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        require(network&&config,"network/config");config->clearFlag(BuilderFlag::kTF32);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);config->setMaxAuxStreams(0);
        auto* input=network->addInput("input",DataType::kFLOAT,Dims2(m,k));require(input,"input");
        auto* w=network->addConstant(Dims2(k,n),Weights{DataType::kFLOAT,first.data(),int64_t(first.size())});
        require(w,"first weights");
        auto* product=network->addMatrixMultiply(*input,MatrixOperation::kNONE,*w->getOutput(0),MatrixOperation::kNONE);
        require(product,"first product");
        auto* relu=network->addActivation(*product->getOutput(0),ActivationType::kRELU);require(relu,"relu");
        auto* v=network->addConstant(Dims2(n,o),Weights{DataType::kFLOAT,second.data(),int64_t(second.size())});
        require(v,"second weights");
        auto* output=network->addMatrixMultiply(*relu->getOutput(0),MatrixOperation::kNONE,*v->getOutput(0),MatrixOperation::kNONE);
        require(output,"second product");output->getOutput(0)->setName("output");network->markOutput(*output->getOutput(0));
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));require(bool(plan),"plan");
        runtime.reset(createInferRuntime(logger));require(bool(runtime),"runtime");
        engine.reset(runtime->deserializeCudaEngine(plan->data(),plan->size()));require(bool(engine),"engine");
        require(engine->getDeviceMemorySizeV2()>0,"Need a network with nonzero context scratch");
    }
    std::vector<float> input(int seed)const{
        std::vector<float> x(size_t(m)*k);
        for(size_t i=0;i<x.size();++i)x[i]=float(int((i*13+seed*19)%127)-63)/64;
        return x;
    }
    double checkReference(const std::vector<float>& x,const float* output)const{
        // gamma(K) for the first dot, ReLU's 1-Lipschitz bound, then gamma(N)
        // for the second dot. All source/input values are exact binary FP32.
        const double u=std::ldexp(1.,-24),gk=k*u/(1-k*u),gn=n*u/(1-n*u);
        double worst=0;
        for(int row=0;row<m;++row){
            std::vector<double> hidden(n),errors(n);
            for(int j=0;j<n;++j){
                double dot=0,absolute=0;
                for(int q=0;q<k;++q){double term=double(x[row*k+q])*first[q*n+j];dot+=term;absolute+=std::abs(term);}
                hidden[j]=std::max(0.,dot);errors[j]=gk*absolute;
            }
            for(int col=0;col<o;++col){
                double reference=0,absolute=0,propagated=0;
                for(int j=0;j<n;++j){double weight=second[j*o+col];reference+=hidden[j]*weight;
                    absolute+=std::abs(hidden[j]*weight);propagated+=errors[j]*std::abs(weight);}
                const double bound=gn*absolute+(1+gn)*propagated+1e-7;
                const double error=std::abs(double(output[row*o+col])-reference);
                require(std::isfinite(output[row*o+col])&&error<=bound,"FP64 forward-error bound");
                worst=std::max(worst,error/bound);
            }
        }
        return worst;
    }
};
struct Arena{
    Buffer raw;void* data=nullptr;int64_t capacity;size_t offset=0;
    explicit Arena(int64_t bytes):raw(size_t(bytes)+1536),capacity(bytes){
        require(bytes>0,"arena size");cu(cudaMemset(raw.data,0xa5,raw.bytes));cu(cudaDeviceSynchronize());
        auto address=(reinterpret_cast<uintptr_t>(raw.data)+511)&~uintptr_t(511);
        data=reinterpret_cast<void*>(address+512);offset=static_cast<char*>(data)-static_cast<char*>(raw.data);
        require(reinterpret_cast<uintptr_t>(data)%512==0,"arena alignment");
    }
    void guards()const{
        std::vector<unsigned char> host(raw.bytes);cu(cudaMemcpy(host.data(),raw.data,raw.bytes,cudaMemcpyDeviceToHost));
        require(std::all_of(host.begin(),host.begin()+offset,[](auto b){return b==0xa5;}),"prefix guard");
        require(std::all_of(host.begin()+offset+size_t(capacity),host.end(),[](auto b){return b==0xa5;}),"suffix guard");
    }
};
struct Route{
    std::shared_ptr<Arena> arena;std::mutex mutex;cudaEvent_t retired=nullptr;cudaStream_t last=nullptr;bool used=false;
    size_t handoffs=0,submissions=0;
    explicit Route(std::shared_ptr<Arena> value):arena(std::move(value)){cu(cudaEventCreateWithFlags(&retired,cudaEventDisableTiming));}
    ~Route(){if(used)cudaEventSynchronize(retired);cudaEventDestroy(retired);}
    void wait(cudaStream_t stream){if(used&&last!=stream){cu(cudaStreamWaitEvent(stream,retired,0));++handoffs;}}
    void record(cudaStream_t stream){cu(cudaEventRecord(retired,stream));last=stream;used=true;++submissions;}
};
struct Slot{
    Engine& net;std::shared_ptr<Route> route;Buffer input,output;Stream stream;
    std::unique_ptr<IExecutionContext> context;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    Slot(Engine& value,std::shared_ptr<Route> shared={}):net(value),route(std::move(shared)),
        input(size_t(net.m)*net.k*sizeof(float)),output(size_t(net.m)*net.o*sizeof(float)){
        if(route)require(route->arena->capacity>=net.engine->getDeviceMemorySizeV2(),"insufficient shared workspace");
        context.reset(net.engine->createExecutionContext(route?ExecutionContextAllocationStrategy::kUSER_MANAGED:ExecutionContextAllocationStrategy::kSTATIC));
        require(bool(context),"context");
        if(route)context->setDeviceMemoryV2(route->arena->data,route->arena->capacity);
        require(context->setTensorAddress("input",input.data)&&context->setTensorAddress("output",output.data),"bindings");
    }
    ~Slot(){cudaStreamSynchronize(stream.value);if(executable)cudaGraphExecDestroy(executable);
        if(graph)cudaGraphDestroy(graph);context.reset();}
    void capture(){
        std::unique_lock<std::mutex> lease;if(route){lease=std::unique_lock<std::mutex>(route->mutex);route->wait(stream.value);}
        require(context->enqueueV3(stream.value),"warm enqueue");cu(cudaStreamSynchronize(stream.value));
        cu(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        bool ok=context->enqueueV3(stream.value);auto status=cudaStreamEndCapture(stream.value,&graph);
        cu(status);require(ok&&graph,"capture enqueue");cu(cudaGraphInstantiateWithFlags(&executable,graph,0));
        if(route)route->record(stream.value);
    }
    void run(const std::vector<float>& values,float* destination,bool captured){
        std::unique_lock<std::mutex> lease;if(route){lease=std::unique_lock<std::mutex>(route->mutex);route->wait(stream.value);}
        cu(cudaMemcpyAsync(input.data,values.data(),input.bytes,cudaMemcpyHostToDevice,stream.value));
        if(captured){require(executable,"missing graph");cu(cudaGraphLaunch(executable,stream.value));}
        else require(context->enqueueV3(stream.value),"ordinary enqueue");
        cu(cudaMemcpyAsync(destination,output.data,output.bytes,cudaMemcpyDeviceToHost,stream.value));
        if(route)route->record(stream.value);
    }
    void sync(){cu(cudaStreamSynchronize(stream.value));}
};
int main(int argc,char** argv){
    if(argc!=2)return 2;
    json result={{"all_passed",false},{"scope","Standalone TensorRT scratch ownership/order/graph probe; no Garnet integration, model memory or speed acceptance"}};
    try{
        Engine a(32,128,512,64),b(17,96,768,80);constexpr int cases=16;
        std::vector<std::vector<float>> expectedA(cases),expectedB(cases),inputsA(cases),inputsB(cases);double worst=0;
        for(int i=0;i<cases;++i){inputsA[i]=a.input(i+1);inputsB[i]=b.input(i+1);}
        {
            Buffer outA(size_t(a.m)*a.o*4,true),outB(size_t(b.m)*b.o*4,true);Slot ca(a),cb(b);
            for(int i=0;i<cases;++i){
                const auto& xa=inputsA[i];const auto& xb=inputsB[i];ca.run(xa,static_cast<float*>(outA.data),false);cb.run(xb,static_cast<float*>(outB.data),false);
                ca.sync();cb.sync();worst=std::max({worst,a.checkReference(xa,static_cast<float*>(outA.data)),b.checkReference(xb,static_cast<float*>(outB.data))});
                expectedA[i].assign(static_cast<float*>(outA.data),static_cast<float*>(outA.data)+a.m*a.o);
                expectedB[i].assign(static_cast<float*>(outB.data),static_cast<float*>(outB.data)+b.m*b.o);
            }
        }
        auto arena=std::make_shared<Arena>(std::max(a.engine->getDeviceMemorySizeV2(),b.engine->getDeviceMemorySizeV2()));
        std::weak_ptr<Arena> weak=arena;auto route=std::make_shared<Route>(arena);
        {
            auto tooSmall=std::make_shared<Route>(std::make_shared<Arena>(a.engine->getDeviceMemorySizeV2()-1));
            bool rejected=false;try{Slot invalid(a,tooSmall);}catch(const std::exception&){rejected=true;}
            require(rejected,"undersized arena accepted");
        }
        size_t handoffs=0,submissions=0;
        {
            Buffer outputsA(size_t(cases)*a.m*a.o*4,true),outputsB(size_t(cases)*b.m*b.o*4,true);
            Slot sa(a,route),sb(b,route);
            auto outA=[&](int i){return static_cast<float*>(outputsA.data)+size_t(i)*a.m*a.o;};
            auto outB=[&](int i){return static_cast<float*>(outputsB.data)+size_t(i)*b.m*b.o;};
            auto verify=[&](){
                sa.sync();sb.sync();
                for(int i=0;i<cases;++i){
                    require(!std::memcmp(outA(i),expectedA[i].data(),sa.output.bytes),"A differs from independent context");
                    require(!std::memcmp(outB(i),expectedB[i].data(),sb.output.bytes),"B differs from independent context");
                    worst=std::max({worst,a.checkReference(inputsA[i],outA(i)),b.checkReference(inputsB[i],outB(i))});
                }
            };
            for(int i=0;i<cases;++i){sa.run(inputsA[i],outA(i),false);sb.run(inputsB[i],outB(i),false);}verify();
            sa.capture();sb.capture();arena.reset();route.reset();require(!weak.expired(),"contexts lost retained workspace owner");
            for(int i=0;i<cases;++i){sa.run(inputsA[i],outA(i),true);sb.run(inputsB[i],outB(i),true);}verify();
            std::exception_ptr failureA,failureB;
            std::thread ta([&](){try{for(int i=0;i<cases;++i)sa.run(inputsA[i],outA(i),true);}catch(...){failureA=std::current_exception();}});
            std::thread tb([&](){try{for(int i=0;i<cases;++i)sb.run(inputsB[i],outB(i),true);}catch(...){failureB=std::current_exception();}});
            ta.join();tb.join();if(failureA)std::rethrow_exception(failureA);if(failureB)std::rethrow_exception(failureB);verify();
            sa.route->arena->guards();handoffs=sa.route->handoffs;submissions=sa.route->submissions;
            require(handoffs>=cases*2,"cross-stream ordering was not exercised");
        }
        require(weak.expired(),"workspace owner retained after all contexts/graphs retired");
        result.update({{"all_passed",true},{"engine_context_upper_bounds_bytes",{a.engine->getDeviceMemorySizeV2(),b.engine->getDeviceMemorySizeV2()}},
            {"source_cases",cases*2},{"shared_output_checks",cases*2*3},{"maximum_fp64_bound_fraction",worst},
            {"cross_stream_handoffs",handoffs},{"ordered_submissions",submissions},
            {"ordinary_alternating_and_captured_alternating_and_two_host_threads",true},
            {"insufficient_workspace_rejected",true},{"public_owner_drop_and_final_retirement",true}});
    }catch(const std::exception& error){result["error"]=error.what();}
    std::ofstream(argv[1])<<result.dump(2)<<'\n';std::cout<<result.dump()<<std::endl;
    return result["all_passed"].get<bool>()?0:1;
}
