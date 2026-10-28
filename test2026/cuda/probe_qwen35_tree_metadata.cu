// SPDX-License-Identifier: Apache-2.0
#include "../../src/cuda/qwen35_tree_metadata.cuh"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <process.h>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Buffer {
    static constexpr int guard=64;uint32_t* raw=nullptr;size_t words;
    explicit Buffer(size_t n):words(n){check(cudaMalloc(&raw,(n+2*guard)*4));std::vector<uint32_t> initial(n+2*guard,0x12345678);check(cudaMemcpy(raw,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(raw);}Buffer(const Buffer&)=delete;
    void* data(){return raw+guard;}
    template<class T>void upload(const std::vector<T>& values){require(values.size()*sizeof(T)==words*4,"Upload size");check(cudaMemcpy(data(),values.data(),words*4,cudaMemcpyHostToDevice));}
    std::vector<uint32_t> read(){std::vector<uint32_t> all(words+2*guard);check(cudaMemcpy(all.data(),raw,all.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<guard;++i)require(all[i]==0x12345678&&all[guard+words+i]==0x12345678,"Buffer guard overwritten");return {all.begin()+guard,all.begin()+guard+words};}
    template<class T>void preserved(const std::vector<T>& expected){auto got=read();require(got.size()*4==expected.size()*sizeof(T)&&!std::memcmp(got.data(),expected.data(),got.size()*4),"Readonly input modified");}
};
int main(int argc,char** argv){try{
    require(argc==4,"Expected fixture,group,output");std::ifstream input(argv[1]);json fixture;input>>fixture;json group;
    for(auto& g:fixture.at("groups"))if(g.at("name")==argv[2])group=g;require(!group.is_null(),"Group not found");
    Garnet::Qwen35TreeMetadataConfig c{group.at("batch"),group.at("nodes"),group.at("capacity")};
    std::ofstream log(argv[3]);require(bool(log),"Output");int device;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
    char uuid[80];auto u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);
    std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
    require(std::string(uuid)=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Wrong physical GPU");
    Buffer parents(size_t(c.batch)*c.nodes),active(size_t(c.batch)*c.nodes),prefix(c.batch),output(size_t(c.batch)*836);cudaStream_t stream{};check(cudaStreamCreate(&stream));
    auto run=[&]{require(Garnet::RunQwen35TreeMetadata(c,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data(),stream)==0,"Metadata launch");};
    auto poison=[&]{check(cudaMemsetAsync(output.data(),0xff,output.words*4,stream));};
    auto first=group.at("scenarios").at(0);auto p=first.at("parents").get<std::vector<float>>(),a=first.at("active").get<std::vector<float>>(),s=first.at("prefix").get<std::vector<float>>();
    parents.upload(p);active.upload(a);prefix.upload(s);run();check(cudaStreamSynchronize(stream));
    cudaGraph_t graph{};cudaGraphExec_t exec{};check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();check(cudaStreamEndCapture(stream,&graph));
    size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));require(count==2,"Metadata graph must clear and build");check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    log<<json({{"event","identity"},{"process_id",_getpid()},{"gpu_uuid",uuid},{"batch",c.batch},{"nodes",c.nodes},{"capacity",c.capacity},
        {"metadata_words_per_row",836},{"graph_kernel_count",count},{"canonical_pointers_accepted",false},{"model_or_speed_accepted",false}}).dump()<<std::endl;
    size_t compared=0;int replays=0;
    for(const auto& scene:group.at("scenarios")){
        p=scene.at("parents").get<std::vector<float>>();a=scene.at("active").get<std::vector<float>>();s=scene.at("prefix").get<std::vector<float>>();
        auto expected=scene.at("expected_int32").get<std::vector<int32_t>>();parents.upload(p);active.upload(a);prefix.upload(s);
        for(int replay=0;replay<3;++replay){poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));auto actual=output.read();
            require(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"GPU metadata differs from independent Python oracle");compared+=actual.size();++replays;}
        parents.preserved(p);active.preserved(a);prefix.preserved(s);
        log<<json({{"event","scenario_complete"},{"scenario",scene.at("scenario")},{"seed",scene.at("seed")},{"all836_words_per_row_exact_python",true},
            {"same_graph_and_storage_with_updated_controls",true},{"poisoned_replays",3}}).dump()<<std::endl;
    }
    p=first.at("parents").get<std::vector<float>>();a=first.at("active").get<std::vector<float>>();s=first.at("prefix").get<std::vector<float>>();
    auto expected=first.at("expected_int32").get<std::vector<int32_t>>();
    int gpu_rejections=0;std::vector<std::string> names;
    auto malformed=[&](std::string name,std::vector<float> pp,std::vector<float> aa,std::vector<float> ss,int code){
        parents.upload(pp);active.upload(aa);prefix.upload(ss);poison();run();check(cudaStreamSynchronize(stream));auto actual=output.read();
        require(actual[0]==0&&int(actual[1])==code,"GPU failed to reject bad row");
        for(int b=1;b<c.batch;++b)require(!std::memcmp(actual.data()+b*836,expected.data()+b*836,836*4),"Malformed row contaminated another row");
        parents.preserved(pp);active.preserved(aa);prefix.preserved(ss);names.push_back(name);++gpu_rejections;
    };
    auto pp=p;pp[0]=0;malformed("root",pp,a,s,1);
    for(float value:{.5f,std::nanf(""),INFINITY}){pp=p;pp[0]=value;malformed("root_noninteger_or_nonfinite",pp,a,s,1);}
    if(c.nodes>1){for(float value:{-2.f,1.f,1.5f,std::nanf(""),INFINITY}){pp=p;pp[1]=value;malformed("parent_range_or_noninteger",pp,a,s,1);}}
    if(c.nodes>=9){pp=p;for(int n=1;n<9;++n)pp[n]=float(n-1);malformed("path_depth_overflow",pp,a,s,4);}
    for(float value:{.5f,std::nanf(""),INFINITY}){auto aa=a;aa[0]=value;malformed("active_nonbinary_or_nonfinite",p,aa,s,2);}
    for(float value:{-1.f,.5f,512.f,513.f,std::nanf(""),INFINITY}){auto ss=s;ss[0]=value;malformed("prefix_range_fractional_nonfinite_or_capacity",p,a,ss,3);}
    parents.upload(p);active.upload(a);prefix.upload(s);poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));auto reset=output.read();
    require(!std::memcmp(reset.data(),expected.data(),reset.size()*4),"Invalid-to-valid graph reset failed");compared+=reset.size();++replays;
    auto before=output.read();int host_rejections=0;
    auto bad=[&](Garnet::Qwen35TreeMetadataConfig cfg,const float* pp,const float* aa,const float* ss,Garnet::Qwen35TreeMetadata* out){require(Garnet::RunQwen35TreeMetadata(cfg,pp,aa,ss,out,stream)!=0,"Bad host call accepted");++host_rejections;};
    auto cfg=c;cfg.batch=3;bad(cfg,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    cfg=c;cfg.nodes=0;bad(cfg,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    cfg=c;cfg.nodes=65;bad(cfg,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    cfg=c;cfg.capacity=8193;bad(cfg,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    bad(c,nullptr,(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    bad(c,(float*)parents.data(),nullptr,(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)output.data());
    bad(c,(float*)parents.data(),(float*)active.data(),nullptr,(Garnet::Qwen35TreeMetadata*)output.data());
    bad(c,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),nullptr);
    bad(c,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)parents.data());
    bad(c,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)active.data());
    bad(c,(float*)parents.data(),(float*)active.data(),(float*)prefix.data(),(Garnet::Qwen35TreeMetadata*)prefix.data());
    check(cudaStreamSynchronize(stream));require(output.read()==before,"Rejected host call advanced output");parents.preserved(p);active.preserved(a);prefix.preserved(s);
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    log<<json({{"event","complete"},{"all_controls_passed",true},{"compared_int32_words",compared},{"poisoned_graph_replays",replays},
        {"gpu_bad_rows_rejected",gpu_rejections},{"gpu_rejection_names",names},{"other_rows_uncontaminated",true},{"host_bad_calls_without_output_advance",host_rejections},
        {"all_inputs_and_guards_preserved",true},{"model_or_speed_or_native_serving_accepted",false}}).dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
