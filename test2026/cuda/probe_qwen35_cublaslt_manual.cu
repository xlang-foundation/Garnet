// SPDX-License-Identifier: Apache-2.0
// Unwired explicit-algorithm screen. Never compile/run alongside a model.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunBF16Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
}
namespace {
void cudaCheck(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void blas(cublasStatus_t value){if(value!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(value)));}
struct Buffer {
    void* owner=nullptr;void* p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){cudaCheck(cudaMalloc(&owner,n+512));p=static_cast<char*>(owner)+256;cudaCheck(cudaMemset(owner,0x6d,n+512));}
    ~Buffer(){if(owner)cudaFree(owner);}
    void guard(){std::array<unsigned char,256>a{},b{};cudaCheck(cudaMemcpy(a.data(),owner,256,cudaMemcpyDeviceToHost));cudaCheck(cudaMemcpy(b.data(),static_cast<char*>(p)+bytes,256,cudaMemcpyDeviceToHost));for(auto v:a)if(v!=0x6d)throw std::runtime_error("Leading guard changed");for(auto v:b)if(v!=0x6d)throw std::runtime_error("Trailing guard changed");}
    template<class T>void put(const std::vector<T>& v){if(v.size()*sizeof(T)!=bytes)throw std::runtime_error("Upload size");cudaCheck(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(){std::vector<T> v(bytes/sizeof(T));cudaCheck(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));return v;}
};
__global__ void evict(uint32_t* p,size_t n){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=uint32_t(i*2654435761ULL);}
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    Graph(const std::function<void()>& run,cudaStream_t stream){run();cudaCheck(cudaStreamSynchronize(stream));cudaCheck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();cudaCheck(cudaStreamEndCapture(stream,&graph));cudaCheck(cudaGraphInstantiate(&executable,graph,0));}
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    void run(cudaStream_t stream){cudaCheck(cudaGraphLaunch(executable,stream));}
};
double median(std::vector<float> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
double selectionTime(const std::function<void()>& run,cudaStream_t stream){
    run();cudaCheck(cudaStreamSynchronize(stream));cudaEvent_t a,b;cudaCheck(cudaEventCreate(&a));cudaCheck(cudaEventCreate(&b));std::vector<float> samples;
    for(int i=0;i<3;++i){cudaCheck(cudaEventRecord(a,stream));run();cudaCheck(cudaEventRecord(b,stream));cudaCheck(cudaEventSynchronize(b));float ms;cudaCheck(cudaEventElapsedTime(&ms,a,b));samples.push_back(ms*1000);}
    cudaCheck(cudaEventDestroy(a));cudaCheck(cudaEventDestroy(b));return median(samples);
}
std::array<std::vector<float>,2> paired(Graph& a,Graph& b,Buffer& sweep,cudaStream_t stream){
    std::array<Graph*,2> graphs{&a,&b};std::array<std::vector<float>,2> samples;
    for(auto* graph:graphs)for(int i=0;i<3;++i)graph->run(stream);cudaCheck(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;cudaCheck(cudaEventCreate(&begin));cudaCheck(cudaEventCreate(&end));
    for(int repeat=0;repeat<9;++repeat)for(int step=0;step<2;++step){int index=(repeat+step)%2;
        evict<<<2048,256,0,stream>>>((uint32_t*)sweep.p,sweep.bytes/4);cudaCheck(cudaGetLastError());
        cudaCheck(cudaEventRecord(begin,stream));graphs[index]->run(stream);cudaCheck(cudaEventRecord(end,stream));cudaCheck(cudaEventSynchronize(end));float ms;cudaCheck(cudaEventElapsedTime(&ms,begin,end));samples[index].push_back(ms*1000);
    }
    cudaCheck(cudaEventDestroy(begin));cudaCheck(cudaEventDestroy(end));return samples;
}
template<class T>T cap(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoCapAttributes_t attr){T value{};size_t size=0;blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,&value,sizeof(value),&size));return value;}
std::vector<int> caps(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoCapAttributes_t attr){size_t n=0;blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,nullptr,0,&n));std::vector<int> values(n/sizeof(int));if(n)blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,values.data(),n,&n));if(values.empty())values.push_back(0);return values;}
template<class T>bool set(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attr,T value){return cublasLtMatmulAlgoConfigSetAttribute(&algo,attr,&value,sizeof(value))==CUBLAS_STATUS_SUCCESS;}
uint16_t bits(__nv_bfloat16 value){uint16_t v;std::memcpy(&v,&value,2);return v;}
struct Candidate{cublasLtMatmulAlgo_t algo;json config;size_t workspace;double selection;};
}
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("Expected M K N for one actual projection shape");
    int m=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]);
    if(m<1||m>512||k<1||k>16384||n<1||n>32768)throw std::runtime_error("Shape outside prospective bounds");
    cudaCheck(cudaSetDevice(0));cudaDeviceProp props{};cudaCheck(cudaGetDeviceProperties(&props,0));cudaStream_t stream;cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();if(!context)throw std::runtime_error("Production projection context");
    cublasLtHandle_t lt;blas(cublasLtCreate(&lt));cublasLtMatmulDesc_t operation;blas(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    cublasLtMatrixLayout_t aw,bx,cy;blas(cublasLtMatrixLayoutCreate(&aw,CUDA_R_16BF,k,n,k));blas(cublasLtMatrixLayoutCreate(&bx,CUDA_R_16BF,k,m,k));blas(cublasLtMatrixLayoutCreate(&cy,CUDA_R_16BF,n,m,n));
    constexpr size_t workspaceLimit=8ULL<<20;Buffer workspace(workspaceLimit),sweep(256ULL<<20),x(size_t(m)*k*2),w(size_t(n)*k*2),y(size_t(m)*n*2);
    std::mt19937 generator(274971);std::normal_distribution<float> normal(0,.08f);std::vector<__nv_bfloat16> input(size_t(m)*k),weight(size_t(n)*k);for(auto& v:input)v=__float2bfloat16_rn(normal(generator));for(auto& v:weight)v=__float2bfloat16_rn(normal(generator));x.put(input);w.put(weight);
    auto production=[&]{if(Garnet::RunBF16Projection(context,x.p,w.p,y.p,m,k,n,stream,nullptr,0))throw std::runtime_error("Production GEMM failed");};production();cudaCheck(cudaStreamSynchronize(stream));auto reference=y.get<__nv_bfloat16>();
    auto verify=[&]{auto output=y.get<__nv_bfloat16>();size_t different=0;for(size_t i=0;i<output.size();++i){if(!std::isfinite(__bfloat162float(output[i])))throw std::runtime_error("Nonfinite output");different+=bits(output[i])!=bits(reference[i]);}
        double maxFraction=0,u=std::ldexp(1.,-24),gamma=k*u/(1-k*u);int samples=0;
        for(int row=0;row<std::min(m,2);++row)for(int col=0;col<std::min(n,64);++col){int index=col*(n-1)/std::max(1,std::min(n,64)-1);double dot=0,sum=0;for(int d=0;d<k;++d){double term=double(__bfloat162float(input[size_t(row)*k+d]))*__bfloat162float(weight[size_t(index)*k+d]);dot+=term;sum+=std::abs(term);}double bound=gamma*sum+(std::abs(dot)+gamma*sum)/256.+std::ldexp(1.,-133);for(double v:{double(__bfloat162float(output[size_t(row)*n+index])),double(__bfloat162float(reference[size_t(row)*n+index]))}){double f=std::abs(v-dot)/bound;if(f>1)throw std::runtime_error("Original fixed FP64 bound failed");maxFraction=std::max(maxFraction,f);}++samples;}
        return json{{"finite_words",output.size()},{"bf16_values_different_from_production",different},{"fp64_samples_per_provider",samples},{"maximum_original_fixed_bound_fraction",maxFraction}};};
    const float alpha=1,beta=0;
    auto run=[&](Candidate& c){blas(cublasLtMatmul(lt,operation,&alpha,w.p,aw,x.p,bx,&beta,y.p,cy,y.p,cy,&c.algo,c.workspace?workspace.p:nullptr,c.workspace,stream));};
    std::cout<<json{{"event","manual_algorithm_metadata"},{"m",m},{"k",k},{"n",n},{"gpu",props.name},{"seed",274971},{"workspace_limit_bytes",workspaceLimit},{"eviction_bytes",sweep.bytes},{"baseline_math",verify()},{"precision","BF16 inputs/weights/output; FP32 accumulator; only NONE or COMPUTE_TYPE reduction"},{"scope","Synthetic unwired single-shape search, custom0/swizzle0 only, not exhaustive. Three warm selection samples, then nine new alternating graph samples for up to8 finalists. No model speed/quality/memory admission."}}.dump()<<std::endl;
    int ids[256],count=0;blas(cublasLtMatmulAlgoGetIds(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,256,ids,&count));std::vector<Candidate> candidates;size_t checked=0,valid=0;
    for(int i=0;i<count;++i){cublasLtMatmulAlgo_t base;if(cublasLtMatmulAlgoInit(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,ids[i],&base)!=CUBLAS_STATUS_SUCCESS)continue;
        uint64_t flags=cap<uint64_t>(base,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
        if((flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F||(flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF)continue;
        auto tiles=caps(base,CUBLASLT_ALGO_CAP_TILE_IDS),stages=caps(base,CUBLASLT_ALGO_CAP_STAGES_IDS);int splitSupport=cap<int>(base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);uint32_t reductions=cap<uint32_t>(base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
        for(int tile:tiles)for(int stage:stages)for(int split:{1,2,4,8,16}){
            uint32_t reduction=split==1?CUBLASLT_REDUCTION_SCHEME_NONE:CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            if(split>1&&(!splitSupport||!(reductions&reduction)))continue;
            auto algo=base;if(!set(algo,CUBLASLT_ALGO_CONFIG_TILE_ID,tile)||!set(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,stage)||!set(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,split)||!set(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,reduction))continue;
            ++checked;cublasLtMatmulHeuristicResult_t check{};if(cublasLtMatmulAlgoCheck(lt,operation,aw,bx,cy,cy,&algo,&check)!=CUBLAS_STATUS_SUCCESS||check.state!=CUBLAS_STATUS_SUCCESS||check.workspaceSize>workspaceLimit)continue;
            Candidate c{algo,{{"id",ids[i]},{"tile",tile},{"stages",stage},{"split_k",split},{"reduction",reduction},{"numerical_flags",flags}},check.workspaceSize,0};
            auto status=cublasLtMatmul(lt,operation,&alpha,w.p,aw,x.p,bx,&beta,y.p,cy,y.p,cy,&c.algo,c.workspace?workspace.p:nullptr,c.workspace,stream);
            if(status==CUBLAS_STATUS_NOT_SUPPORTED)continue;blas(status);cudaCheck(cudaStreamSynchronize(stream));++valid;c.selection=selectionTime([&]{run(c);},stream);
            std::cout<<json{{"event","selection_only"},{"config",c.config},{"required_workspace_bytes",c.workspace},{"warm_median_us",c.selection}}.dump()<<std::endl;candidates.push_back(c);
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate&a,const Candidate&b){return a.selection<b.selection;});if(candidates.size()>8)candidates.resize(8);if(candidates.empty())throw std::runtime_error("No qualifying manual algorithms");
    Graph baseline(production,stream);
    int selectionRank=0;
    for(auto& c:candidates){run(c);cudaCheck(cudaStreamSynchronize(stream));auto math=verify();Graph candidate([&]{run(c);},stream);auto samples=paired(baseline,candidate,sweep,stream);candidate.run(stream);cudaCheck(cudaStreamSynchronize(stream));auto replayMath=verify();
        std::cout<<json{{"event","finalist"},{"selection_rank",selectionRank++},{"config",c.config},{"required_workspace_bytes",c.workspace},{"selection_warm_median_us",c.selection},{"baseline_cold_graph_us",samples[0]},{"candidate_cold_graph_us",samples[1]},{"baseline_median_us",median(samples[0])},{"candidate_median_us",median(samples[1])},{"component_speed_ratio",median(samples[0])/median(samples[1])},{"math",math},{"post_replay_math",replayMath}}.dump()<<std::endl;
    }
    cudaCheck(cudaStreamSynchronize(stream));if(x.get<__nv_bfloat16>().size()!=input.size())throw std::runtime_error("Input size");auto afterX=x.get<__nv_bfloat16>(),afterW=w.get<__nv_bfloat16>();if(std::memcmp(afterX.data(),input.data(),x.bytes)||std::memcmp(afterW.data(),weight.data(),w.bytes))throw std::runtime_error("Input/weight mutation");for(auto* buffer:{&x,&w,&y,&workspace,&sweep})buffer->guard();
    std::cout<<json{{"event","complete"},{"algorithm_ids_returned",count},{"algorithm_id_request_limit",256},{"configurations_checked",checked},{"configurations_run",valid},{"finalists",candidates.size()},{"all_buffer_guards_and_inputs_preserved",true},{"primary_performance_eligible",false},{"full_request_accepted",false}}.dump()<<std::endl;
    cublasLtMatrixLayoutDestroy(aw);cublasLtMatrixLayoutDestroy(bx);cublasLtMatrixLayoutDestroy(cy);cublasLtMatmulDescDestroy(operation);cublasLtDestroy(lt);cudaStreamSynchronize(stream);cudaStreamDestroy(stream);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
