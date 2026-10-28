// SPDX-License-Identifier: Apache-2.0
// Unwired classic cuBLAS workspace study. No new arithmetic or serving path.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include "bf16_block_projection_utils.cuh"
#include <cmath>
#include <random>

std::vector<__nv_bfloat16> wsInput(size_t n,unsigned seed){
    std::mt19937 g(seed);std::normal_distribution<float> normal(0,1);
    std::vector<__nv_bfloat16> out(n);for(auto& x:out)x=__float2bfloat16_rn(normal(g));return out;
}
void wsUpload(Buffer& b,const void* data,size_t n,cudaStream_t stream){
    require(n==b.size,"Upload span differs");check(cudaMemcpyAsync(b.data,data,n,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
}
struct WSPairs{
    std::array<std::vector<double>,2> graph,eviction,total;
    json asJSON()const{return {{"graph_us",graph},{"excluded_eviction_us",eviction},{"including_eviction_total_us",total},
        {"baseline_median_us",median(graph[0])},{"candidate_median_us",median(graph[1])},
        {"baseline_over_candidate",median(graph[0])/median(graph[1])}};}
};
WSPairs wsPairs(Captured& a,Captured& b,Buffer& eviction,cudaStream_t stream,int count,bool cold){
    std::array<Captured*,2> graphs{&a,&b};WSPairs out;Timer timer;
    for(auto* graph:graphs)for(int i=0;i<3;++i)check(cudaGraphLaunch(graph->executable,stream));
    check(cudaStreamSynchronize(stream));
    for(int r=0;r<count;++r)for(int step=0;step<2;++step){
        int which=(r+step)%2;auto sample=timer.run(*graphs[which],stream,eviction,cold,unsigned(r));
        out.graph[which].push_back(sample[0]);out.eviction[which].push_back(sample[1]);out.total[which].push_back(sample[2]);
    }return out;
}
json wsNodes(cudaGraph_t graph,int depth=0){
    require(depth<=8,"Nested graph depth exceeded");
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
    json counts=json::object(),children=json::array();size_t kernels=0;
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));auto key=std::to_string(int(type));
        counts[key]=counts.value(key,0)+1;
        if(type==cudaGraphNodeTypeGraph){cudaGraph_t child=nullptr;check(cudaGraphChildGraphNodeGetGraph(node,&child));auto record=wsNodes(child,depth+1);kernels+=record.at("kernel_nodes_recursive").get<size_t>();children.push_back(record);continue;}
        if(type==cudaGraphNodeTypeKernel)++kernels;
        require(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeMemset||type==cudaGraphNodeTypeEmpty,
                "Unexpected host/allocation/child/control node in direct GEMM capture");}
    if(depth==0)require(kernels>0,"GEMM graph has no kernel nodes");
    return {{"top_level_nodes",n},{"cuda_node_type_counts",counts},{"kernel_nodes_recursive",kernels},{"children",children}};
}
struct WSMatrix{
    std::string name;json descriptor;int count=0;
    std::vector<unsigned char> raw,expected;
    std::vector<__nv_bfloat16> host;
    std::unique_ptr<Buffer> weight;
    std::unique_ptr<Captured> baseline;
};
std::string wsModule(const char* name){
    auto module=GetModuleHandleA(name);if(!module)return {};
    char path[MAX_PATH]{};auto count=GetModuleFileNameA(module,path,MAX_PATH);
    require(count>0&&count<MAX_PATH,"Library module path unavailable/truncated");return path;
}
int main(int argc,char** argv){try{
    require(argc==4,"Require manifest,group,unused JSONL");require(!std::ifstream(argv[3]).good(),"Preserve workspace evidence");
    json manifest,group;std::ifstream in(argv[1]);in>>manifest;
    for(const auto& row:manifest.at("groups"))if(row.at("name")==argv[2])group=row;
    require(!group.is_null(),"Unknown workspace geometry");
    int rows=group.at("rows"),inner=group.at("inner"),columns=group.at("columns");
    require(rows>=4&&rows<=512&&inner>0&&columns>0,"Invalid workspace dimensions");
    require(manifest.at("selection_seed")==571901&&manifest.at("holdout_seed")==571902,"Wrong fresh seeds");
    check(cudaSetDevice(0));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));}
    require(uuid.str()=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Wrong GPU");
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();require(bool(context),"Production projection context absent");
    Buffer input(size_t(rows)*inner*2),output(size_t(rows)*columns*2),scratch(size_t(4)<<20);
    size_t evictionBytes=std::max(size_t(256)<<20,size_t(prop.l2CacheSize)*4);Buffer eviction(evictionBytes);
    std::vector<WSMatrix> matrices;matrices.reserve(group.at("fixtures").size());
    for(size_t i=0;i<group.at("fixtures").size();++i){
        WSMatrix m;m.name=group.at("fixtures")[i];m.count=group.at("counts_per_pass")[i];
        for(const auto& f:manifest.at("fixtures"))if(f.at("name")==m.name)m.descriptor=f;
        require(!m.descriptor.is_null()&&m.descriptor.at("inner")==inner&&m.descriptor.at("columns")==columns,"Fixture geometry differs");
        m.raw=bytes(m.descriptor.at("raw_file"));require(m.raw.size()==size_t(inner)*columns*2,"Weight span differs");
        m.host.resize(size_t(inner)*columns);std::memcpy(m.host.data(),m.raw.data(),m.raw.size());
        for(auto x:m.host)require(std::isfinite(__bfloat162float(x)),"Nonfinite weight");
        m.weight=std::make_unique<Buffer>(m.raw.size());wsUpload(*m.weight,m.raw.data(),m.raw.size(),stream);matrices.push_back(std::move(m));
    }
    unsigned activeSeed=571901;
    auto hostInput=wsInput(size_t(rows)*inner,activeSeed);wsUpload(input,hostInput.data(),input.size,stream);
    auto run=[&](size_t index,size_t workspace){
        require(workspace==0||workspace==16384||workspace==(size_t(4)<<20),"Unregistered workspace");
        require(Garnet::RunBF16Projection(context,input.data,matrices[index].weight->data,output.data,rows,inner,columns,stream,
                    workspace?scratch.data:nullptr,workspace)==0,"Production classic cuBLAS failed");
    };
    auto replay=[&](Captured& graph){
        // A stale output cannot make an empty or incomplete replay pass.
        check(cudaMemsetAsync(output.data,0xff,output.size,stream));
        check(cudaGraphLaunch(graph.executable,stream));check(cudaStreamSynchronize(stream));
    };
    auto guards=[&]{
        require(input.guards()&&output.guards()&&scratch.guards()&&eviction.guards(),"Buffer guard changed");
        auto actual=download(input);require(actual.size()==hostInput.size()*2&&!std::memcmp(actual.data(),hostInput.data(),actual.size()),"Input changed");
        for(const auto& m:matrices)require(m.weight->guards()&&download(*m.weight)==m.raw,"Readonly weight changed");
    };
    auto verify=[&](size_t index){
        auto& m=matrices[index];auto raw=download(output);size_t different=differentWords(m.expected,raw);
        std::vector<__nv_bfloat16> actual(size_t(rows)*columns);std::memcpy(actual.data(),raw.data(),raw.size());
        for(auto x:actual)require(std::isfinite(__bfloat162float(x)),"Nonfinite output");
        double u=std::ldexp(1.,-24),gamma=inner*u/(1-inner*u),maximum=0.;json differences=json::array();
        for(size_t i=0;i<actual.size()&&differences.size()<16;++i){
            __nv_bfloat16 base;std::memcpy(&base,m.expected.data()+2*i,2);
            if(std::memcmp(&base,&actual[i],2))differences.push_back({{"row",i/columns},{"column",i%columns},
                {"baseline",__bfloat162float(base)},{"candidate",__bfloat162float(actual[i])}});
        }
        int sampledRows=std::min(rows,16),samples=0;
        for(int ri=0;ri<sampledRows;++ri)for(int ci=0;ci<32;++ci){
            int row=ri*(rows-1)/(sampledRows-1),column=ci*(columns-1)/31;double dot=0.,absolute=0.;
            for(int k=0;k<inner;++k){double term=double(__bfloat162float(hostInput[size_t(row)*inner+k]))*__bfloat162float(m.host[size_t(column)*inner+k]);dot+=term;absolute+=std::abs(term);}
            double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
            __nv_bfloat16 base;std::memcpy(&base,m.expected.data()+2*(size_t(row)*columns+column),2);
            for(double value:{double(__bfloat162float(base)),double(__bfloat162float(actual[size_t(row)*columns+column]))}){
                double fraction=std::abs(value-dot)/bound;
                if(!(fraction<=1.)){
                    std::string stem=std::string(argv[3])+".math-failure";
                    std::ofstream metadata(stem+".json");metadata<<json{{"group",group},{"fixture",m.name},{"input_seed",activeSeed},
                        {"row",row},{"column",column},{"reference_fp64_dot",dot},{"absolute_product_sum",absolute},{"bound",bound},{"observed_value",value}}.dump(2);
                    std::ofstream actualFile(stem+".actual.u16",std::ios::binary),baseFile(stem+".baseline.u16",std::ios::binary),inputFile(stem+".input.u16",std::ios::binary);
                    actualFile.write(reinterpret_cast<const char*>(raw.data()),raw.size());baseFile.write(reinterpret_cast<const char*>(m.expected.data()),m.expected.size());
                    inputFile.write(reinterpret_cast<const char*>(hostInput.data()),hostInput.size()*2);
                    throw std::runtime_error("Original FP32 accumulation/BF16 round envelope failed; complete raw buffers retained");
                }maximum=std::max(maximum,fraction);
            }++samples;
        }
        require(output.guards()&&scratch.guards(),"Numerical observation guard changed");
        return json{{"finite_words",actual.size()},{"different_bf16_words",different},{"strict_native_bit_parity_passed",different==0},
            {"first16_differences",differences},{"fp64_samples_per_provider",samples},{"maximum_original_fixed_bound_fraction",maximum},
            {"original_fp32_accumulation_bf16_envelope_passed",true}};
    };
    std::ofstream log(argv[3]);require(bool(log),"Output log unavailable");
    auto module=wsModule("cublas64_13.dll");require(!module.empty(),"cuBLAS module path absent");
    int runtime=0,driver=0;check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
    log<<json{{"event","metadata"},{"group",group},{"process_id",GetCurrentProcessId()},{"gpu_uuid",uuid.str()},{"gpu",prop.name},
        {"cublas_dll_path",module},{"cuda_runtime_version",runtime},{"cuda_driver_version",driver},{"l2_bytes",prop.l2CacheSize},
        {"eviction_bytes",evictionBytes},{"selection_seed",571901},{"holdout_seed",571902},{"selection_pairs",5},{"holdout_pairs",31},
        {"workspace_choices_bytes",{0,16384,4194304}},{"maximum_exclusive_scratch_bytes",scratch.size},
        {"same_production_bf16_function_math_policy",true},{"extra_weights_or_models",false},
        {"scope","Classic cuBLAS workspace component only. FP32 envelope is not a waiver of strict parity, original MTP tolerance or independent model-quality gates."}}.dump()<<std::endl;
    Sleep(1250);
    for(size_t i=0;i<matrices.size();++i){for(int n=0;n<3;++n)run(i,0);check(cudaStreamSynchronize(stream));matrices[i].expected=download(output);
        auto math=verify(i);matrices[i].baseline=capture(stream,[&,i]{run(i,0);});
        replay(*matrices[i].baseline);
        require(download(output)==matrices[i].expected,"Baseline capture changed provider output");
        log<<json{{"event","baseline"},{"fixture",matrices[i].name},{"math",math},{"graph",wsNodes(matrices[i].baseline->graph)}}.dump()<<std::endl;}
    size_t selected=0;double bestRatio=1.;bool selectedExact=true;
    for(size_t workspace:{size_t(16384),size_t(4194304)}){
        double a=0,b=0;bool exact=true;json records=json::array();
        for(size_t i=0;i<matrices.size();++i){
            for(int n=0;n<3;++n)run(i,workspace);check(cudaStreamSynchronize(stream));auto plain=download(output);auto math=verify(i);
            auto graph=capture(stream,[&,i,workspace]{run(i,workspace);});
            replay(*graph);require(download(output)==plain,"Capture changed provider output");
            auto pairs=wsPairs(*matrices[i].baseline,*graph,eviction,stream,5,true);
            replay(*graph);require(download(output)==plain,"Selection replay changed provider output");
            exact=exact&&math.at("strict_native_bit_parity_passed").get<bool>();
            a+=matrices[i].count*median(pairs.graph[0]);b+=matrices[i].count*median(pairs.graph[1]);
            records.push_back({{"fixture",matrices[i].name},{"math",math},{"samples",pairs.asJSON()},{"graph",wsNodes(graph->graph)}});
        }
        guards();double ratio=a/b;
        log<<json{{"event","selection_only"},{"workspace_bytes",workspace},{"weighted_cold_ratio",ratio},{"strict_native_bit_parity_passed",exact},
            {"fp32_envelope_probe_passed",true},{"matrices",records}}.dump()<<std::endl;
        if(ratio>bestRatio){bestRatio=ratio;selected=workspace;selectedExact=exact;}
    }
    log<<json{{"event","selection_frozen"},{"workspace_bytes",selected},{"selection_cold_ratio",bestRatio},
        {"strict_selection_bit_parity_passed",selectedExact},{"quality_admission",selectedExact?"exact component only":"independent model-quality qualification REQUIRED"},
        {"no_finalist_rescue",true},{"gate_up_share_one_workspace",true}}.dump()<<std::endl;
    activeSeed=571902;hostInput=wsInput(size_t(rows)*inner,activeSeed);wsUpload(input,hostInput.data(),input.size,stream);
    bool exact=true;
    for(size_t i=0;i<matrices.size();++i){
        run(i,0);check(cudaStreamSynchronize(stream));matrices[i].expected=download(output);auto baseMath=verify(i);
        replay(*matrices[i].baseline);
        require(download(output)==matrices[i].expected,"Held-out baseline graph changed provider output");
        run(i,selected);check(cudaStreamSynchronize(stream));auto plain=download(output);auto candidateMath=verify(i);
        exact=exact&&candidateMath.at("strict_native_bit_parity_passed").get<bool>();
        auto graph=capture(stream,[&,i]{run(i,selected);});
        auto cold=wsPairs(*matrices[i].baseline,*graph,eviction,stream,31,true);
        auto warm=wsPairs(*matrices[i].baseline,*graph,eviction,stream,31,false);
        replay(*graph);require(download(output)==plain,"Held-out replay changed provider output");
        auto replayMath=verify(i);guards();
        log<<json{{"event","holdout_matrix"},{"fixture",matrices[i].name},{"workspace_bytes",selected},
            {"baseline_math",baseMath},{"candidate_math",candidateMath},{"replay_math",replayMath},
            {"cold",cold.asJSON()},{"warm",warm.asJSON()},{"all_guards_inputs_weights_preserved",true}}.dump()<<std::endl;
    }
    guards();log<<json{{"event","complete"},{"workspace_bytes",selected},{"strict_holdout_bit_parity_passed",exact},
        {"all_fp32_envelope_probes_passed",true},{"all_guards_inputs_weights_preserved",true},{"full_model_quality_accepted",false},
        {"actual_cublas_lt_dll_path",wsModule("cublasLt64_13.dll")},
        {"original_strict_mtp_gate_waived",false},{"native_serving_modified",false},{"whole_model_speed_accepted",false}}.dump()<<std::endl;
    check(cudaStreamSynchronize(stream));matrices.clear();context.reset();check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& error){std::cerr<<"PROJECTION_WORKSPACE_COMPONENT_FAILED "<<error.what()<<std::endl;return 2;}}
