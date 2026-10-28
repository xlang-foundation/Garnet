// A separate explicit factory. The legacy factory and every existing plugin keep
// their original implementation; no caller is redirected to this candidate.
std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128ReconstructedContext(){
    ModelOptFP8Z128ReconstructTrace trace;
#if CUDART_VERSION >= 12080
    int device=-1;cudaDeviceProp properties{};
    if(trace.api("cudaGetDevice",[&](){return cudaGetDevice(&device);})!=cudaSuccess||
       trace.api("cudaGetDeviceProperties",[&](){return cudaGetDeviceProperties(&properties,device);})!=cudaSuccess||
       properties.major!=12||properties.minor!=0||
       trace.api("cublasLtGetVersion",[&](){return cublasLtGetVersion();})!=130400)
        return trace.empty<ModelOptFP8Z128Context>("device_or_version",device,properties.major,properties.minor);
    // Keep a separate cache so legacy and reconstructed configurations cannot mix.
    static std::unordered_map<int,std::weak_ptr<ModelOptFP8Z128Context>> reconstructed;
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto previous=reconstructed[device].lock()){trace.success(true);return previous;}
    auto context=std::make_shared<ModelOptFP8Z128Context>();context->device=device;
    const auto okay=[](cublasStatus_t value){return value==CUBLAS_STATUS_SUCCESS;};
    if(!okay(trace.api("cublasLtCreate",[&](){return cublasLtCreate(&context->handle);}))||
       !okay(trace.api("cublasLtMatmulDescCreate",[&](){return cublasLtMatmulDescCreate(&context->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F);})))
        return trace.empty<ModelOptFP8Z128Context>("handle_or_operation");
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    if(!okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA));}))||
       !okay(trace.api("cublasLtMatmulDescSetAttribute",[&](){return cublasLtMatmulDescSetAttribute(context->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB));}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->weight,CUDA_R_8F_E4M3,5120,6144,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->input,CUDA_R_8F_E4M3,5120,128,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&context->output,CUDA_R_16BF,6144,128,6144);})))
        return trace.empty<ModelOptFP8Z128Context>("operation_or_actual_layout");
    CanonicalSelection canonical;
    if(!okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.input,CUDA_R_8F_E4M3,5120,32,5120);}))||
       !okay(trace.api("cublasLtMatrixLayoutCreate",[&](){return cublasLtMatrixLayoutCreate(&canonical.output,CUDA_R_16BF,6144,32,6144);})))
        return trace.empty<ModelOptFP8Z128Context>("canonical_layout");
    cublasLtMatmulAlgo_t algorithm{};
    if(!okay(trace.api("cublasLtMatmulAlgoInit",[&](){return cublasLtMatmulAlgoInit(context->handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,
        CUDA_R_8F_E4M3,CUDA_R_8F_E4M3,CUDA_R_16BF,CUDA_R_16BF,67,&algorithm);})))
        return trace.empty<ModelOptFP8Z128Context>("algorithm_init");
    struct Field{const char* name;cublasLtMatmulAlgoConfigAttributes_t attribute;uint32_t expected;size_t bytes;};
    // Public query values independently observed from both retained descriptors.
    const Field fields[]={
        {"ID",CUBLASLT_ALGO_CONFIG_ID,67,4},{"TILE_ID",CUBLASLT_ALGO_CONFIG_TILE_ID,13,4},
        {"STAGES_ID",CUBLASLT_ALGO_CONFIG_STAGES_ID,17,4},{"SPLITK_NUM",CUBLASLT_ALGO_CONFIG_SPLITK_NUM,1,4},
        {"REDUCTION_SCHEME",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,0,4},{"CTA_SWIZZLING",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,1,4},
        {"CUSTOM_OPTION",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,18,4},{"INNER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,0,2},
        {"CLUSTER_SHAPE_ID",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,0,2}};
    for(size_t i=1;i<9;++i){
        const auto& field=fields[i];const uint16_t value16=static_cast<uint16_t>(field.expected);
        const void* value=field.bytes==2?static_cast<const void*>(&value16):static_cast<const void*>(&field.expected);
        if(!okay(trace.api("cublasLtMatmulAlgoConfigSetAttribute",[&](){return cublasLtMatmulAlgoConfigSetAttribute(&algorithm,field.attribute,value,field.bytes);})))
            return trace.empty<ModelOptFP8Z128Context>("configuration_set",i,int(field.attribute),static_cast<long long>(field.bytes));
    }
    for(size_t i=0;i<9;++i){
        const auto& field=fields[i];uint16_t value16=0;uint32_t value32=0;size_t written=0;
        void* value=field.bytes==2?static_cast<void*>(&value16):static_cast<void*>(&value32);
        const auto status=trace.api("cublasLtMatmulAlgoConfigGetAttribute",[&](){return cublasLtMatmulAlgoConfigGetAttribute(&algorithm,field.attribute,value,field.bytes,&written);});
        const uint32_t actual=field.bytes==2?value16:value32;
        trace.note("configuration",field.name,actual,static_cast<long long>(written),int(field.attribute));
        if(!okay(status)||written!=field.bytes||actual!=field.expected)
            return trace.empty<ModelOptFP8Z128Context>("configuration_get",i,actual,static_cast<long long>(written));
    }
    trace.note("algorithm","configured",0,0,0,&algorithm,sizeof(algorithm));
    cublasLtMatmulHeuristicResult_t checked32{},checked128{};
    const auto status32=trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,
        canonical.input,canonical.output,canonical.output,&algorithm,&checked32);});
    trace.note("check","M32",int(checked32.state),static_cast<long long>(checked32.workspaceSize));
    if(!okay(status32)||checked32.state!=CUBLAS_STATUS_SUCCESS||checked32.workspaceSize>ModelOptFP8Z128BlasWorkspace)
        return trace.empty<ModelOptFP8Z128Context>("algorithm_check_M32",int(checked32.state),static_cast<long long>(checked32.workspaceSize));
    const auto status128=trace.api("cublasLtMatmulAlgoCheck",[&](){return cublasLtMatmulAlgoCheck(context->handle,context->operation,context->weight,
        context->input,context->output,context->output,&algorithm,&checked128);});
    trace.note("check","M128",int(checked128.state),static_cast<long long>(checked128.workspaceSize));
    if(!okay(status128)||checked128.state!=CUBLAS_STATUS_SUCCESS||checked128.workspaceSize>ModelOptFP8Z128BlasWorkspace)
        return trace.empty<ModelOptFP8Z128Context>("algorithm_check_M128",int(checked128.state),static_cast<long long>(checked128.workspaceSize));
    context->info.device=device;context->info.algorithm=67;context->info.requiredWorkspace=checked128.workspaceSize;context->info.opaque=algorithm;
    reconstructed[device]=context;trace.success(false);return context;
#else
    return trace.empty<ModelOptFP8Z128Context>("unsupported_CUDART");
#endif
}

