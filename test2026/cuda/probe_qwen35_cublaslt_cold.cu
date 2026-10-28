// SPDX-License-Identifier: Apache-2.0
// Unwired cold-ranked algorithm search. Compile/run only after model jobs close.
#include "bf16_block_projection_utils.cuh"
#include <cublasLt.h>
#include <cmath>
#include <random>

void blas(cublasStatus_t code){if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(code)));}
template<class T>T capability(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoCapAttributes_t attribute){
    T result{};size_t size=0;blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attribute,&result,sizeof(result),&size));return result;
}
std::vector<int> capabilities(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoCapAttributes_t attribute){
    size_t size=0;blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attribute,nullptr,0,&size));
    std::vector<int> result(size/sizeof(int));if(size)blas(cublasLtMatmulAlgoCapGetAttribute(&algo,attribute,result.data(),size,&size));
    if(result.empty())result.push_back(0);return result;
}
template<class T>bool configure(cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attribute,T value){
    return cublasLtMatmulAlgoConfigSetAttribute(&algo,attribute,&value,sizeof(value))==CUBLAS_STATUS_SUCCESS;
}
unsigned short bf16Bits(__nv_bfloat16 value){unsigned short result;std::memcpy(&result,&value,2);return result;}
std::vector<__nv_bfloat16> makeInput(size_t count,unsigned seed){
    std::mt19937 generator(seed);std::normal_distribution<float> normal(0,1);
    std::vector<__nv_bfloat16> result(count);for(auto& value:result)value=__float2bfloat16_rn(normal(generator));return result;
}
void upload(Buffer& destination,const void* source,size_t size,cudaStream_t stream){
    require(size==destination.size,"Upload byte count differs");
    check(cudaMemcpyAsync(destination.data,source,size,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
}
struct Samples {
    std::array<std::vector<double>,2> graph,eviction,total;
    json asJSON()const{return {{"graph_us",graph},{"excluded_eviction_us",eviction},{"including_eviction_total_us",total},
        {"baseline_median_us",median(graph[0])},{"candidate_median_us",median(graph[1])},
        {"ordinary_over_candidate",median(graph[0])/median(graph[1])}};}
};
Samples paired(Captured& ordinary,Captured& candidate,Buffer& eviction,cudaStream_t stream,int repeats,bool cold){
    std::array<Captured*,2> graphs{&ordinary,&candidate};Samples result;Timer timer;
    for(auto* graph:graphs)for(int repeat=0;repeat<3;++repeat)check(cudaGraphLaunch(graph->executable,stream));
    check(cudaStreamSynchronize(stream));
    for(int repeat=0;repeat<repeats;++repeat)for(int step=0;step<2;++step){int index=(repeat+step)%2;
        auto sample=timer.run(*graphs[index],stream,eviction,cold,unsigned(repeat));
        result.graph[index].push_back(sample[0]);result.eviction[index].push_back(sample[1]);result.total[index].push_back(sample[2]);
    }
    return result;
}
struct Candidate {cublasLtMatmulAlgo_t algo;json config;size_t workspace;double coldRatio;};

int main(int argc,char** argv){try{
    require(argc==5,"Require fixture manifest, fixture name, rows and unused output JSONL");
    require(!std::ifstream(argv[4]).good(),"Preserve cold search output");
    std::ifstream source(argv[1]);json manifest;source>>manifest;json fixture;
    for(const auto& item:manifest.at("fixtures"))if(item.at("name")==argv[2])fixture=item;
    require(!fixture.is_null(),"Unknown learned fixture");
    int rows=std::stoi(argv[3]),inner=fixture.at("inner"),columns=fixture.at("columns");
    require((rows==4||rows==8||rows==16||rows==32)&&inner>0&&columns>0,"Shape outside decode screen");
    auto raw=bytes(fixture.at("raw_file"));require(raw.size()==size_t(inner)*columns*2,"Learned weight size differs");
    std::vector<__nv_bfloat16> weights(size_t(inner)*columns);std::memcpy(weights.data(),raw.data(),raw.size());
    for(auto value:weights)require(std::isfinite(__bfloat162float(value)),"Nonfinite learned weight");
    check(cudaSetDevice(0));cudaDeviceProp property{};check(cudaGetDeviceProperties(&property,0));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int index=0;index<16;++index){if(index==4||index==6||index==8||index==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(property.uuid.bytes[index]));}
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();require(bool(context),"Production context failed");
    cublasLtHandle_t lt=nullptr;blas(cublasLtCreate(&lt));cublasLtMatmulDesc_t operation=nullptr;
    blas(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;
    blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    cublasLtMatrixLayout_t aw=nullptr,bx=nullptr,cy=nullptr;
    blas(cublasLtMatrixLayoutCreate(&aw,CUDA_R_16BF,inner,columns,inner));
    blas(cublasLtMatrixLayoutCreate(&bx,CUDA_R_16BF,inner,rows,inner));
    blas(cublasLtMatrixLayoutCreate(&cy,CUDA_R_16BF,columns,rows,columns));
    constexpr size_t workspaceLimit=8ULL<<20;const size_t evictionBytes=std::max(size_t(256)<<20,size_t(property.l2CacheSize)*4);
    Buffer weight(raw.size()),input(size_t(rows)*inner*2),output(size_t(rows)*columns*2),workspace(workspaceLimit),eviction(evictionBytes);
    auto hostInput=makeInput(size_t(rows)*inner,292147);upload(weight,raw.data(),raw.size(),stream);
    upload(input,hostInput.data(),input.size,stream);check(cudaMemsetAsync(eviction.data,0,eviction.size,stream));check(cudaStreamSynchronize(stream));
    std::ofstream log(argv[4]);require(bool(log),"Cold log unavailable");
    auto ordinary=[&]{require(Garnet::RunBF16Projection(context,input.data,weight.data,output.data,rows,inner,columns,stream,nullptr,0)==0,"Production GEMM failed");};
    ordinary();check(cudaStreamSynchronize(stream));auto expected=download(output);
    auto verify=[&]{
        auto actual=download(output);size_t different=differentWords(expected,actual);double maximum=0;
        std::vector<__nv_bfloat16> values(size_t(rows)*columns);std::memcpy(values.data(),actual.data(),actual.size());
        for(auto value:values)require(std::isfinite(__bfloat162float(value)),"Nonfinite candidate projection");
        double unit=std::ldexp(1.,-24),gamma=inner*unit/(1-inner*unit);int samples=0;
        for(int row=0;row<std::min(rows,2);++row)for(int sample=0;sample<64;++sample){
            int column=sample*(columns-1)/63;double dot=0,absolute=0;
            for(int k=0;k<inner;++k){double term=double(__bfloat162float(hostInput[size_t(row)*inner+k]))*__bfloat162float(weights[size_t(column)*inner+k]);dot+=term;absolute+=std::abs(term);}
            double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
            __nv_bfloat16 baseline;std::memcpy(&baseline,expected.data()+2*(size_t(row)*columns+column),2);
            for(double value:{double(__bfloat162float(values[size_t(row)*columns+column])),double(__bfloat162float(baseline))}){
                double fraction=std::abs(value-dot)/bound;require(fraction<=1,"Original fixed FP64 bound failed");maximum=std::max(maximum,fraction);
            }++samples;
        }
        require(input.guards()&&weight.guards()&&output.guards()&&workspace.guards()&&eviction.guards(),"Cold search canary changed");
        return json{{"different_bf16_words",different},{"finite_words",values.size()},{"fp64_samples_per_provider",samples},{"maximum_original_fixed_bound_fraction",maximum}};
    };
    log<<json{{"event","metadata"},{"fixture",fixture.at("name")},{"fixture_raw_sha256",fixture.at("raw_sha256")},
        {"rows",rows},{"inner",inner},{"columns",columns},{"gpu",property.name},{"gpu_uuid",uuid.str()},{"l2_bytes",property.l2CacheSize},
        {"selection_seed",292147},{"holdout_seed",292148},{"workspace_limit_bytes",workspaceLimit},{"eviction_bytes",evictionBytes},
        {"selection_pairs_per_candidate",5},{"holdout_pairs",31},{"uploads_ordered_on_test_stream",true},{"baseline_math",verify()},
        {"scope","Unwired learned-weight component. Cold pair ranking, frozen single winner, fresh-input/cold holdout. No model memory, quality, latency or2x admission."}}.dump()<<std::endl;
    auto baseline=capture(stream,ordinary);const float alpha=1,beta=0;
    auto run=[&](const Candidate& candidate){blas(cublasLtMatmul(lt,operation,&alpha,weight.data,aw,input.data,bx,&beta,output.data,cy,output.data,cy,
        &candidate.algo,candidate.workspace?workspace.data:nullptr,candidate.workspace,stream));};
    int ids[256],idCount=0;blas(cublasLtMatmulAlgoGetIds(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,256,ids,&idCount));
    std::vector<Candidate> eligible;size_t checked=0,executed=0,bitRejected=0;
    for(int i=0;i<idCount;++i){
        cublasLtMatmulAlgo_t base;if(cublasLtMatmulAlgoInit(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,ids[i],&base)!=CUBLAS_STATUS_SUCCESS)continue;
        uint64_t flags=capability<uint64_t>(base,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
        if((flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F||
           (flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF)continue;
        auto tiles=capabilities(base,CUBLASLT_ALGO_CAP_TILE_IDS),stages=capabilities(base,CUBLASLT_ALGO_CAP_STAGES_IDS);
        int supportsSplit=capability<int>(base,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);uint32_t reductions=capability<uint32_t>(base,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
        for(int tile:tiles)for(int stage:stages)for(int split:{1,2,4,8,16}){
            uint32_t reduction=split==1?CUBLASLT_REDUCTION_SCHEME_NONE:CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            if(split>1&&(!supportsSplit||!(reductions&reduction)))continue;
            auto algo=base;if(!configure(algo,CUBLASLT_ALGO_CONFIG_TILE_ID,tile)||!configure(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,stage)||
                !configure(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,split)||!configure(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,reduction))continue;
            ++checked;cublasLtMatmulHeuristicResult_t validity{};
            if(cublasLtMatmulAlgoCheck(lt,operation,aw,bx,cy,cy,&algo,&validity)!=CUBLAS_STATUS_SUCCESS||validity.state!=CUBLAS_STATUS_SUCCESS||validity.workspaceSize>workspaceLimit)continue;
            Candidate candidate{algo,{{"id",ids[i]},{"tile",tile},{"stages",stage},{"split_k",split},{"reduction",reduction},{"numerical_flags",flags}},validity.workspaceSize,0};
            auto status=cublasLtMatmul(lt,operation,&alpha,weight.data,aw,input.data,bx,&beta,output.data,cy,output.data,cy,&algo,
                candidate.workspace?workspace.data:nullptr,candidate.workspace,stream);
            if(status==CUBLAS_STATUS_NOT_SUPPORTED)continue;blas(status);check(cudaStreamSynchronize(stream));++executed;
            auto math=verify();bool exact=math.at("different_bf16_words").get<size_t>()==0;
            if(!exact){++bitRejected;log<<json{{"event","selection_bit_rejected"},{"config",candidate.config},{"math",math}}.dump()<<std::endl;continue;}
            auto graph=capture(stream,[&]{run(candidate);});auto samples=paired(*baseline,*graph,eviction,stream,5,true);
            candidate.coldRatio=median(samples.graph[0])/median(samples.graph[1]);
            check(cudaGraphLaunch(graph->executable,stream));check(cudaStreamSynchronize(stream));auto replay=verify();
            require(replay.at("different_bf16_words").get<size_t>()==0,"Selection replay changed exact outputs");
            log<<json{{"event","selection_only"},{"config",candidate.config},{"workspace_bytes",candidate.workspace},{"cold_samples",samples.asJSON()},{"math",math},{"replay_math",replay}}.dump()<<std::endl;
            eligible.push_back(candidate);
        }
    }
    require(!eligible.empty(),"No exact cold-ranked candidate");
    std::stable_sort(eligible.begin(),eligible.end(),[](const Candidate& a,const Candidate& b){return a.coldRatio>b.coldRatio;});
    const Candidate selected=eligible.front();
    require(download(weight)==raw,"Selection changed learned weights");
    auto selectionInput=download(input);
    require(selectionInput.size()==hostInput.size()*2&&!std::memcmp(selectionInput.data(),hostInput.data(),selectionInput.size()),"Selection changed input");
    log<<json{{"event","selection_frozen"},{"config",selected.config},{"workspace_bytes",selected.workspace},{"selection_cold_ratio",selected.coldRatio},
        {"eligible_candidates",eligible.size()},{"selection_input_and_weight_bytes_preserved",true},{"other_finalists_allowed_for_rescue",false}}.dump()<<std::endl;
    // Freeze before a new input is generated; never select on held-out samples.
    hostInput=makeInput(size_t(rows)*inner,292148);upload(input,hostInput.data(),input.size,stream);
    ordinary();check(cudaStreamSynchronize(stream));expected=download(output);auto baselineHoldout=verify();
    run(selected);check(cudaStreamSynchronize(stream));auto holdoutMath=verify();
    log<<json{{"event","holdout_correctness"},{"baseline",baselineHoldout},{"candidate",holdoutMath}}.dump()<<std::endl;
    require(holdoutMath.at("different_bf16_words").get<size_t>()==0,"Frozen candidate failed fresh-input bit gate");
    auto selectedGraph=capture(stream,[&]{run(selected);});
    auto cold=paired(*baseline,*selectedGraph,eviction,stream,31,true),warm=paired(*baseline,*selectedGraph,eviction,stream,31,false);
    check(cudaGraphLaunch(selectedGraph->executable,stream));check(cudaStreamSynchronize(stream));auto replayMath=verify();
    require(replayMath.at("different_bf16_words").get<size_t>()==0,"Held-out replay changed exact outputs");
    require(download(weight)==raw,"Learned weights changed");
    auto finalInput=download(input);require(finalInput.size()==hostInput.size()*2&&!std::memcmp(finalInput.data(),hostInput.data(),finalInput.size()),"Input changed");
    log<<json{{"event","holdout"},{"config",selected.config},{"workspace_bytes",selected.workspace},{"cold",cold.asJSON()},{"warm",warm.asJSON()},
        {"all_output_words_bit_exact",true},{"replay_math",replayMath}}.dump()<<std::endl;
    log<<json{{"event","complete"},{"algorithm_ids",idCount},{"algorithm_request_limit",256},{"configurations_checked",checked},
        {"configurations_run",executed},{"selection_bit_rejections",bitRejected},{"exact_candidates",eligible.size()},
        {"all_guards_inputs_weights_preserved",true},{"production_model_modified",false},{"full_request_accepted",false}}.dump()<<std::endl;
    selectedGraph.reset();baseline.reset();cublasLtMatrixLayoutDestroy(aw);cublasLtMatrixLayoutDestroy(bx);cublasLtMatrixLayoutDestroy(cy);
    cublasLtMatmulDescDestroy(operation);cublasLtDestroy(lt);context.reset();check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& error){std::cerr<<"CUBLASLT_COLD_FAILED "<<error.what()<<std::endl;return 1;}}
