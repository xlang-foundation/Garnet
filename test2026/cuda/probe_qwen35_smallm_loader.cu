// Standalone learned-weight screen. No Garnet serving/default integration.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include "bf16_block_projection_utils.cuh"
#include "qwen35_smallm_loader_wmma.cuh"
#include <cmath>
#include <random>

std::vector<__nv_bfloat16> makeSmallInput(size_t count,unsigned seed) {
    std::mt19937 generator(seed);std::normal_distribution<float> normal(0,1);
    std::vector<__nv_bfloat16> result(count);
    for(auto& value:result)value=__float2bfloat16_rn(normal(generator));return result;
}
void uploadSmall(Buffer& buffer,const void* source,size_t size,cudaStream_t stream) {
    require(buffer.size==size,"Input/upload size mismatch");
    check(cudaMemcpyAsync(buffer.data,source,size,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
}
struct Geometry {int tokens,k,warps;};
json configuration(const Geometry& g) {return {{"tokens",g.tokens},{"tile_k",g.k},{"warps",g.warps}};}
cudaError_t launchSmall(const Geometry& g,const Buffer& input,const Buffer& weight,Buffer& output,
                        int rows,int inner,int columns,cudaStream_t stream) {
    const auto* x=reinterpret_cast<const __nv_bfloat16*>(input.data);
    const auto* w=reinterpret_cast<const __nv_bfloat16*>(weight.data);
    auto* y=reinterpret_cast<__nv_bfloat16*>(output.data);
    #define SMALL_CASE(T,K,W) if(g.tokens==T&&g.k==K&&g.warps==W)return Qwen35SmallM::Run<T,K,W>(x,w,y,rows,inner,columns,stream)
    SMALL_CASE(16,32,2);SMALL_CASE(16,32,4);SMALL_CASE(16,32,8);
    SMALL_CASE(16,64,2);SMALL_CASE(16,64,4);SMALL_CASE(16,64,8);
    SMALL_CASE(16,128,2);SMALL_CASE(16,128,4);SMALL_CASE(16,128,8);
    SMALL_CASE(32,32,4);SMALL_CASE(32,32,8);
    SMALL_CASE(32,64,4);SMALL_CASE(32,64,8);
    SMALL_CASE(32,128,4);SMALL_CASE(32,128,8);
    #undef SMALL_CASE
    return cudaErrorInvalidValue;
}
struct PairSamples {
    std::array<std::vector<double>,2> graph,eviction,total;
    json asJSON()const{return {{"graph_us",graph},{"excluded_eviction_us",eviction},
        {"including_eviction_total_us",total},{"baseline_median_us",median(graph[0])},
        {"candidate_median_us",median(graph[1])},{"ordinary_over_candidate",median(graph[0])/median(graph[1])}};}
};
PairSamples pairedSmall(Captured& ordinary,Captured& candidate,Buffer& eviction,cudaStream_t stream,int repeats,bool cold) {
    std::array<Captured*,2> graphs{&ordinary,&candidate};PairSamples result;Timer timer;
    for(auto* graph:graphs)for(int i=0;i<3;++i)check(cudaGraphLaunch(graph->executable,stream));
    check(cudaStreamSynchronize(stream));
    for(int repeat=0;repeat<repeats;++repeat)for(int step=0;step<2;++step) {
        int index=(repeat+step)%2;auto sample=timer.run(*graphs[index],stream,eviction,cold,unsigned(repeat));
        result.graph[index].push_back(sample[0]);result.eviction[index].push_back(sample[1]);result.total[index].push_back(sample[2]);
    }
    return result;
}
struct MatrixFixture {
    std::string name;json descriptor;int count=0;
    std::vector<unsigned char> raw,expected;
    std::vector<__nv_bfloat16> hostWeight;
    std::unique_ptr<Buffer> deviceWeight;
    std::unique_ptr<Captured> baseline;
};
struct RankedGeometry {Geometry geometry;double ratio;};

int main(int argc,char** argv) {try {
    require(argc==4,"Require fixture manifest, group and unused output JSONL");
    require(!std::ifstream(argv[3]).good(),"Preserve component evidence");
    std::ifstream file(argv[1]);json manifest;file>>manifest;json group;
    for(const auto& entry:manifest.at("groups"))if(entry.at("name")==argv[2])group=entry;
    require(!group.is_null(),"Unknown learned shape group");
    int rows=group.at("rows"),inner=group.at("inner"),columns=group.at("columns"),tokens=group.at("tokens");
    require((rows==4&&tokens==16)||(rows==32&&tokens==32),"Only declared M4/M32 screen");
    require(inner>0&&inner%128==0&&columns>0&&columns%16==0,"Unsupported matrix shape");
    check(cudaSetDevice(0));cudaDeviceProp property{};check(cudaGetDeviceProperties(&property,0));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(property.uuid.bytes[i]));}
    require(uuid.str()=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Wrong physical GPU");
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();require(bool(context),"Production reference context unavailable");
    Buffer input(size_t(rows)*inner*2),output(size_t(rows)*columns*2);
    size_t evictionBytes=std::max(size_t(256)<<20,size_t(property.l2CacheSize)*4);Buffer eviction(evictionBytes);
    std::vector<MatrixFixture> matrices;matrices.reserve(group.at("fixtures").size());
    for(size_t i=0;i<group.at("fixtures").size();++i) {
        MatrixFixture matrix;matrix.name=group.at("fixtures")[i];matrix.count=group.at("counts_per_pass")[i];
        for(const auto& entry:manifest.at("fixtures"))if(entry.at("name")==matrix.name)matrix.descriptor=entry;
        require(!matrix.descriptor.is_null(),"Missing matrix descriptor");
        require(matrix.descriptor.at("inner")==inner&&matrix.descriptor.at("columns")==columns,"Group mixes shapes");
        matrix.raw=bytes(matrix.descriptor.at("raw_file"));require(matrix.raw.size()==size_t(inner)*columns*2,"Learned weight bytes differ");
        matrix.hostWeight.resize(size_t(inner)*columns);std::memcpy(matrix.hostWeight.data(),matrix.raw.data(),matrix.raw.size());
        for(auto value:matrix.hostWeight)require(std::isfinite(__bfloat162float(value)),"Nonfinite source weight");
        matrix.deviceWeight=std::make_unique<Buffer>(matrix.raw.size());
        uploadSmall(*matrix.deviceWeight,matrix.raw.data(),matrix.raw.size(),stream);matrices.push_back(std::move(matrix));
    }
    unsigned selectionSeed=manifest.at("selection_seed"),holdoutSeed=manifest.at("holdout_seed");
    require(selectionSeed==384771&&holdoutSeed==384772,"Input seeds differ from design");
    auto hostInput=makeSmallInput(size_t(rows)*inner,selectionSeed);uploadSmall(input,hostInput.data(),input.size,stream);
    check(cudaMemsetAsync(eviction.data,0,eviction.size,stream));check(cudaStreamSynchronize(stream));
    std::ofstream log(argv[3]);require(bool(log),"Component log unavailable");
    auto ordinary=[&](size_t index){require(Garnet::RunBF16Projection(context,input.data,matrices[index].deviceWeight->data,
        output.data,rows,inner,columns,stream,nullptr,0)==0,"Production GEMM failed");};
    auto immutable=[&] {
        require(input.guards()&&output.guards()&&eviction.guards(),"Component buffer guard changed");
        auto actual=download(input);require(actual.size()==hostInput.size()*2&&!std::memcmp(actual.data(),hostInput.data(),actual.size()),"Input mutated");
        for(const auto& matrix:matrices)require(matrix.deviceWeight->guards()&&download(*matrix.deviceWeight)==matrix.raw,"Weight/guard mutated");
    };
    auto verify=[&](size_t index) {
        auto& matrix=matrices[index];auto raw=download(output);size_t different=differentWords(matrix.expected,raw);
        std::vector<__nv_bfloat16> actual(size_t(rows)*columns);std::memcpy(actual.data(),raw.data(),raw.size());
        for(auto value:actual)require(std::isfinite(__bfloat162float(value)),"Nonfinite candidate output");
        double u=std::ldexp(1.,-24),gamma=inner*u/(1-inner*u),maximum=0;int samples=0;
        for(int row=0;row<std::min(rows,2);++row)for(int sample=0;sample<64;++sample) {
            int column=sample*(columns-1)/63;double dot=0,absolute=0;
            for(int k=0;k<inner;++k){double term=double(__bfloat162float(hostInput[size_t(row)*inner+k]))*
                __bfloat162float(matrix.hostWeight[size_t(column)*inner+k]);dot+=term;absolute+=std::abs(term);}
            double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
            __nv_bfloat16 original;std::memcpy(&original,matrix.expected.data()+2*(size_t(row)*columns+column),2);
            for(double value:{double(__bfloat162float(actual[size_t(row)*columns+column])),double(__bfloat162float(original))}) {
                double fraction=std::abs(value-dot)/bound;require(fraction<=1,"Original fixed FP64 envelope failed");maximum=std::max(maximum,fraction);
            }++samples;
        }
        require(input.guards()&&output.guards()&&eviction.guards()&&matrix.deviceWeight->guards(),"Math-check buffer guard changed");
        return json{{"different_bf16_words",different},{"finite_words",actual.size()},{"fp64_samples_per_provider",samples},
            {"maximum_original_fixed_bound_fraction",maximum}};
    };
    json baselineMath=json::array();
    for(size_t i=0;i<matrices.size();++i) {
        ordinary(i);check(cudaStreamSynchronize(stream));matrices[i].expected=download(output);baselineMath.push_back(verify(i));
        matrices[i].baseline=capture(stream,[&,i]{ordinary(i);});
    }
    log<<json{{"event","metadata"},{"group",group},{"fixtures",[&]{json a=json::array();for(const auto& m:matrices)a.push_back(m.descriptor);return a;}()},
        {"gpu",property.name},{"gpu_uuid",uuid.str()},{"process_id",GetCurrentProcessId()},{"l2_bytes",property.l2CacheSize},{"eviction_bytes",evictionBytes},
        {"selection_seed",selectionSeed},{"holdout_seed",holdoutSeed},{"selection_pairs",5},{"holdout_pairs",31},
        {"baseline_math",baselineMath},{"uploads_ordered_on_test_stream",true},{"workspace_bytes",0},
        {"observer_startup_hold_ms",1250},
        {"scope","Unwired uncompressed BF16 loader/compute-warp component, no full-model quality/memory/speed or2x admission."}}.dump()<<std::endl;
    // Guarantee an opportunity for sampled process metadata; excluded from all
    // GPU event timings and not used as a complete-process performance result.
    Sleep(1250);
    std::vector<RankedGeometry> eligible;int bitRejected=0,checked=0;
    for(int tile:{32,64,128})for(int warps:{2,4,8}) {
        if(warps<=tokens/16)continue;Geometry geometry{tokens,tile,warps};++checked;
        json math=json::array();bool exact=true;
        for(size_t i=0;i<matrices.size();++i) {
            check(launchSmall(geometry,input,*matrices[i].deviceWeight,output,rows,inner,columns,stream));check(cudaStreamSynchronize(stream));
            auto record=verify(i);exact=exact&&record.at("different_bf16_words").get<size_t>()==0;math.push_back(record);
        }
        if(!exact){++bitRejected;log<<json{{"event","selection_bit_rejected"},{"config",configuration(geometry)},{"math",math}}.dump()<<std::endl;continue;}
        json samples=json::array(),replays=json::array();double ordinaryCost=0,candidateCost=0;
        for(size_t i=0;i<matrices.size();++i) {
            auto graph=capture(stream,[&,i,geometry]{check(launchSmall(geometry,input,*matrices[i].deviceWeight,output,rows,inner,columns,stream));});
            auto pairs=pairedSmall(*matrices[i].baseline,*graph,eviction,stream,5,true);
            ordinaryCost+=matrices[i].count*median(pairs.graph[0]);candidateCost+=matrices[i].count*median(pairs.graph[1]);
            check(cudaGraphLaunch(graph->executable,stream));check(cudaStreamSynchronize(stream));auto replay=verify(i);
            require(replay.at("different_bf16_words").get<size_t>()==0,"Selection graph replay changed exact words");
            samples.push_back({{"fixture",matrices[i].name},{"samples",pairs.asJSON()}});replays.push_back(replay);
        }
        immutable();double ratio=ordinaryCost/candidateCost;
        log<<json{{"event","selection_only"},{"config",configuration(geometry)},{"math",math},{"cold_samples",samples},
            {"replay_math",replays},{"weighted_cold_ratio",ratio}}.dump()<<std::endl;eligible.push_back({geometry,ratio});
    }
    immutable();
    if(eligible.empty()) {
        log<<json{{"event","complete"},{"screen_status","rejected_no_exact_candidate"},{"configurations_checked",checked},
            {"selection_bit_rejections",bitRejected},{"exact_candidates",0},{"all_guards_inputs_weights_preserved",true},
            {"holdout_performed",false},{"production_model_modified",false},{"full_request_accepted",false}}.dump()<<std::endl;
    } else {
        std::stable_sort(eligible.begin(),eligible.end(),[](const auto& a,const auto& b){return a.ratio>b.ratio;});const auto selected=eligible.front();
        log<<json{{"event","selection_frozen"},{"config",configuration(selected.geometry)},{"selection_cold_ratio",selected.ratio},
            {"exact_candidates",eligible.size()},{"gate_up_share_one_geometry",true},{"other_finalists_allowed_for_rescue",false},
            {"selection_input_and_weight_bytes_preserved",true}}.dump()<<std::endl;
        hostInput=makeSmallInput(size_t(rows)*inner,holdoutSeed);uploadSmall(input,hostInput.data(),input.size,stream);
        json fresh=json::array();bool exact=true;
        for(size_t i=0;i<matrices.size();++i) {
            ordinary(i);check(cudaStreamSynchronize(stream));matrices[i].expected=download(output);auto baseline=verify(i);
            check(launchSmall(selected.geometry,input,*matrices[i].deviceWeight,output,rows,inner,columns,stream));check(cudaStreamSynchronize(stream));
            auto candidate=verify(i);exact=exact&&candidate.at("different_bf16_words").get<size_t>()==0;
            fresh.push_back({{"fixture",matrices[i].name},{"baseline",baseline},{"candidate",candidate}});
        }
        log<<json{{"event","holdout_correctness"},{"matrices",fresh},{"all_output_words_bit_exact",exact}}.dump()<<std::endl;
        if(exact)for(size_t i=0;i<matrices.size();++i) {
            auto graph=capture(stream,[&,i]{check(launchSmall(selected.geometry,input,*matrices[i].deviceWeight,output,rows,inner,columns,stream));});
            auto cold=pairedSmall(*matrices[i].baseline,*graph,eviction,stream,31,true);
            auto warm=pairedSmall(*matrices[i].baseline,*graph,eviction,stream,31,false);
            check(cudaGraphLaunch(graph->executable,stream));check(cudaStreamSynchronize(stream));auto replay=verify(i);
            require(replay.at("different_bf16_words").get<size_t>()==0,"Holdout replay changed exact output");immutable();
            log<<json{{"event","holdout"},{"fixture",matrices[i].name},{"config",configuration(selected.geometry)},
                {"cold",cold.asJSON()},{"warm",warm.asJSON()},{"replay_math",replay},{"all_output_words_bit_exact",true}}.dump()<<std::endl;
        }
        immutable();
        log<<json{{"event","complete"},{"screen_status",exact?"holdout_completed":"rejected_fresh_input_words"},
            {"configurations_checked",checked},{"selection_bit_rejections",bitRejected},{"exact_candidates",eligible.size()},
            {"all_guards_inputs_weights_preserved",true},{"holdout_performed",exact},{"production_model_modified",false},
            {"full_request_accepted",false}}.dump()<<std::endl;
    }
    matrices.clear();context.reset();check(cudaStreamDestroy(stream));return 0;
} catch(const std::exception& error) {std::cerr<<"QWEN35_SMALLM_LOADER_FAILED "<<error.what()<<std::endl;return 1;} }
