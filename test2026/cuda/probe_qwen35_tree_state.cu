// SPDX-License-Identifier: Apache-2.0
// Unwired bounded tree-state correctness study. No model/timing admission.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <process.h>
#include <memory>
#include <string>
#include "../../src/cuda/qwen35_delta.cu"
#include "qwen35_tree_state.cuh"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{
    static constexpr int guard=64;static constexpr float sentinel=1234567.f;
    float* raw=nullptr;size_t n;
    explicit Buffer(size_t count):n(count){check(cudaMalloc(&raw,(n+2*guard)*4));std::vector<float> initial(n+2*guard,sentinel);check(cudaMemcpy(raw,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(raw);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    float* data(){return raw+guard;}
    void upload(const std::vector<float>& v){if(v.size()!=n)throw std::runtime_error("Upload size");check(cudaMemcpy(data(),v.data(),n*4,cudaMemcpyHostToDevice));}
    std::vector<float> read(){std::vector<float> all(n+2*guard);check(cudaMemcpy(all.data(),raw,all.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<guard;++i)if(all[i]!=sentinel||all[guard+n+i]!=sentinel)throw std::runtime_error("Guard overwritten");
        return {all.begin()+guard,all.begin()+guard+n};}
};
// Independent scalar FP64 equations, fixed bounds shared by both native paths.
std::array<std::vector<double>,3> reference(Garnet::Qwen35DeltaConfig c,int b,int t,const std::vector<std::vector<float>>& x){
    const int ch=2*c.kh*128+c.vh*128;
    std::vector<double> conv(size_t(b)*t*ch),history(x[4].begin(),x[4].end()),state(x[5].begin(),x[5].end()),output(size_t(b)*t*c.vh*128);
    for(int row=0;row<b;++row)for(int channel=0;channel<ch;++channel){
        auto offset=(size_t(row)*ch+channel)*c.kernel;
        for(int token=0;token<t;++token){
            if(x[6][row*t+token]==0)continue;
            for(int k=0;k<c.kernel-1;++k)history[offset+k]=history[offset+k+1];
            history[offset+c.kernel-1]=x[0][(size_t(row)*t+token)*ch+channel];
            double value=0;for(int k=0;k<c.kernel;++k)value+=history[offset+k]*x[7][channel*c.kernel+k];
            conv[(size_t(row)*t+token)*ch+channel]=value/(1+std::exp(-value));
        }
    }
    for(int row=0;row<b;++row)for(int head=0;head<c.vh;++head)for(int token=0;token<t;++token){
        const int bt=row*t+token,kh=head/(c.vh/c.kh);if(x[6][bt]==0)continue;
        std::array<double,128> q,k,core;double qs=0,ks=0;
        for(int d=0;d<128;++d){q[d]=conv[size_t(bt)*ch+kh*128+d];k[d]=conv[size_t(bt)*ch+c.kh*128+kh*128+d];qs+=q[d]*q[d];ks+=k[d]*k[d];}
        for(int d=0;d<128;++d){q[d]/=std::sqrt(qs+1e-6)*std::sqrt(128.0);k[d]/=std::sqrt(ks+1e-6);}
        double av=double(x[3][bt*c.vh+head])+x[8][head];double softplus=std::max(av,0.0)+std::log1p(std::exp(-std::abs(av)));
        double decay=std::exp(-std::exp(double(x[9][head]))*softplus),gate=1/(1+std::exp(-double(x[2][bt*c.vh+head])));
        double sum=0;
        for(int column=0;column<128;++column){
            double memory=0;
            for(int d=0;d<128;++d){auto index=((size_t(row)*c.vh+head)*128+d)*128+column;state[index]*=decay;memory+=state[index]*k[d];}
            double delta=(conv[size_t(bt)*ch+2*c.kh*128+head*128+column]-memory)*gate,result=0;
            for(int d=0;d<128;++d){auto index=((size_t(row)*c.vh+head)*128+d)*128+column;state[index]+=k[d]*delta;result+=state[index]*q[d];}
            core[column]=result;sum+=result*result;
        }
        double factor=1/std::sqrt(sum/128+c.epsilon);
        for(int column=0;column<128;++column){auto index=(size_t(bt)*c.vh+head)*128+column;double z=x[1][index];output[index]=core[column]*factor*x[10][column]*(z/(1+std::exp(-z)));}
    }
    return {std::move(output),std::move(history),std::move(state)};
}

void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
bool same(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
std::vector<int> topology(const std::string& name){
    std::vector<int> p{-1};
    if(name=="chain4"||name=="chain8")for(int n=1;n<(name=="chain4"?4:8);++n)p.push_back(n-1);
    else if(name=="ternary40")for(int n=1;n<40;++n)p.push_back((n-1)/3);
    else if(name=="ragged32"){
        for(int n=1;n<8;++n)p.push_back(n-1);
        for(int n=8;n<32;++n)p.push_back((n-8)%7);
    }else throw std::runtime_error("Tree shape");
    return p;
}
int main(int argc,char** argv){try{
    require(argc==5,"Expected model,batch,topology,output JSONL");
    std::string model=argv[1],tree=argv[3];int b=std::stoi(argv[2]);
    require(model=="2b"||model=="4b","Model");require(b==1||b==2||b==4||b==8,"Batch");
    auto parent=topology(tree);Qwen35Tree::Descriptor d;require(Qwen35Tree::Build(parent.data(),int(parent.size()),d),"Descriptor build");
    std::ofstream log(argv[4]);require(bool(log),"Output file");
    Garnet::Qwen35DeltaConfig c{16,model=="2b"?16:32,128,128,4,1e-6f};int ch=2*c.kh*128+c.vh*128,t=d.count;
    int device;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
    char uuid[80];auto u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);
    std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
    require(std::string(uuid)=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Unexpected physical GPU");
    cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,Qwen35Tree::Recurrent));
    log<<json({{"event","identity"},{"process_id",_getpid()},{"model",model},{"batch",b},{"tree",tree},{"nodes",t},{"leaves",d.leaf_count},
        {"parent",parent},{"gpu",props.name},{"gpu_uuid",uuid},{"state_kernel_registers",attr.numRegs},{"state_kernel_local_bytes",attr.localSizeBytes},
        {"tree_convolution_workspace_bytes",size_t(b)*t*ch*4},{"persistent_per_node_recurrent_state_bytes",0},
        {"canonical_recurrent_bytes",size_t(b)*c.vh*128*128*4},{"timing_or_model_acceptance",false}}).dump()<<std::endl;
    // Host descriptor failures must preserve the previous valid descriptor.
    int malformed=0;auto rejects=[&](std::vector<int> p){auto save=d;require(!Qwen35Tree::Build(p.data(),int(p.size()),save),"Malformed tree accepted");require(!std::memcmp(&save,&d,sizeof(d)),"Failed descriptor modified destination");++malformed;};
    rejects({0});rejects({-1,-1});rejects({-1,1});rejects({-1,2,0});rejects({-1,-2});rejects({-1,0,1,2,3,4,5,6,7});
    auto invalid=d;invalid.owner[0]=d.leaf_count;require(!Qwen35Tree::Valid(invalid),"Invalid owner accepted");++malformed;
    for(int seed:{872191,872192}){
        std::mt19937 rng(seed);std::uniform_real_distribution<float> random(-.2f,.2f);
        std::vector<size_t> sizes={size_t(b)*t*ch,size_t(b)*t*c.vh*128,size_t(b)*t*c.vh,size_t(b)*t*c.vh,size_t(b)*ch*4,
            size_t(b)*c.vh*128*128,size_t(b)*t,size_t(ch)*4,size_t(c.vh),size_t(c.vh),128};
        std::vector<std::vector<float>> x;std::vector<std::unique_ptr<Buffer>> inputs;std::vector<const void*> in;
        for(auto n:sizes){std::vector<float> v(n);for(auto& value:v)value=random(rng);x.push_back(std::move(v));}
        for(auto& value:x[5])value*=.1f;for(auto& value:x[10])value+=1;
        for(int row=0;row<b;++row)for(int node=0;node<t;++node){
            int pattern=seed-872191;
            x[6][row*t+node]=float(!(pattern&&(row%4==3||node%5==2)));
        }
        for(auto& values:x){inputs.push_back(std::make_unique<Buffer>(values.size()));inputs.back()->upload(values);in.push_back(inputs.back()->data());}
        Buffer output(size_t(b)*t*c.vh*128),workspace(size_t(b)*t*ch);cudaStream_t stream{};check(cudaStreamCreate(&stream));
        auto run=[&]{require(Qwen35Tree::Run(c,b,d,in.data(),output.data(),workspace.data(),stream)==0,"Tree launch");};
        auto poison=[&]{check(cudaMemsetAsync(output.data(),0xff,output.n*4,stream));};
        poison();run();check(cudaStreamSynchronize(stream));auto actual=output.read();
        require(std::all_of(actual.begin(),actual.end(),[](float v){return std::isfinite(v);}),"Nonfinite/missing tree output");
        size_t compared=0;double max_abs=0,max_fraction=0;size_t cpu_words=0;
        for(int leaf=0;leaf<d.leaf_count;++leaf){
            std::vector<int> path;for(int n=d.leaf[leaf];n>=0;n=d.parent[n])path.push_back(n);std::reverse(path.begin(),path.end());int length=int(path.size());
            auto y=x;int widths[]={ch,c.vh*128,c.vh,c.vh,0,0,1};
            for(int i:{0,1,2,3,6}){
                y[i].resize(size_t(b)*length*widths[i]);
                for(int row=0;row<b;++row)for(int j=0;j<length;++j)std::copy_n(x[i].data()+(size_t(row)*t+path[j])*widths[i],widths[i],y[i].data()+(size_t(row)*length+j)*widths[i]);
            }
            std::vector<std::unique_ptr<Buffer>> serial_in;std::vector<const void*> input;
            for(auto& values:y){serial_in.push_back(std::make_unique<Buffer>(values.size()));serial_in.back()->upload(values);input.push_back(serial_in.back()->data());}
            Buffer hidden(size_t(b)*length*c.vh*128),history(size_t(b)*ch*4),state(size_t(b)*c.vh*128*128),scratch(size_t(b)*length*ch);
            void* outs[]={hidden.data(),history.data(),state.data()};
            require(Garnet::RunQwen35DeltaRegister128(c,b,length,input.data(),outs,scratch.data(),stream)==0,"Serial native reference failed");
            check(cudaStreamSynchronize(stream));auto expected=hidden.read();history.read();state.read();scratch.read();
            for(int row=0;row<b;++row)for(int j=0;j<length;++j){
                auto source=expected.data()+(size_t(row)*length+j)*c.vh*128;
                auto dest=actual.data()+(size_t(row)*t+path[j])*c.vh*128;
                require(!std::memcmp(source,dest,size_t(c.vh)*128*4),"Tree node differs from native serial path");compared+=size_t(c.vh)*128;
            }
            for(size_t i=0;i<serial_in.size();++i)require(same(serial_in[i]->read(),y[i]),"Native reference input mutation");
            if(leaf==d.leaf_count-1){
                // Unchanged independent scalar equations and original bounds.
                auto cpu=reference(c,b,length,y);
                require(cpu[0].size()==expected.size(),"Oracle shape");
                for(size_t i=0;i<expected.size();++i){
                    double error=std::abs(double(expected[i])-cpu[0][i]),bound=2e-5+2e-4*std::abs(cpu[0][i]);
                    max_abs=std::max(max_abs,error);max_fraction=std::max(max_fraction,error/bound);
                    require(error<=bound,"Original independent FP64 bound failed");++cpu_words;
                }
            }
        }
        cudaGraph_t graph{};cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();check(cudaStreamEndCapture(stream,&graph));
        size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));require(nodes==2,"Tree graph must contain both kernels");
        check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int replay=0;replay<3;++replay){poison();check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));require(same(output.read(),actual),"Poisoned tree graph parity");}
        int rejected=0;auto before=output.read();
        auto bad=[&](Garnet::Qwen35DeltaConfig cfg,int batch,const Qwen35Tree::Descriptor& desc,const void* const* args,float* out,float* work){
            require(Qwen35Tree::Run(cfg,batch,desc,args,out,work,stream)!=0,"Invalid call accepted");++rejected;
        };
        auto cfg=c;cfg.kd=64;bad(cfg,b,d,in.data(),output.data(),workspace.data());
        cfg=c;cfg.epsilon=std::nanf("");bad(cfg,b,d,in.data(),output.data(),workspace.data());
        bad(c,3,d,in.data(),output.data(),workspace.data());bad(c,b,invalid,in.data(),output.data(),workspace.data());
        bad(c,b,d,nullptr,output.data(),workspace.data());bad(c,b,d,in.data(),nullptr,workspace.data());bad(c,b,d,in.data(),output.data(),nullptr);
        bad(c,b,d,in.data(),const_cast<float*>((const float*)in[0]),workspace.data());
        bad(c,b,d,in.data(),output.data(),const_cast<float*>((const float*)in[5]));
        bad(c,b,d,in.data(),output.data(),output.data());
        check(cudaStreamSynchronize(stream));require(same(output.read(),before),"Rejected call advanced output");
        for(size_t i=0;i<inputs.size();++i)require(same(inputs[i]->read(),x[i]),"Tree mutated canonical/input storage");workspace.read();
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        log<<json({{"event","seed_complete"},{"seed",seed},{"all_nodes_native_bit_exact",true},{"compared_path_output_words",compared},
            {"independent_fp64_output_words",cpu_words},{"independent_fp64_max_abs",max_abs},{"independent_fp64_max_bound_fraction",max_fraction},
            {"poisoned_graph_replays",3},{"invalid_calls_rejected_without_output_advance",rejected},
            {"inputs_canonical_bytes_and_all_canaries_preserved",true}}).dump()<<std::endl;
    }
    log<<json({{"event","complete"},{"model",model},{"batch",b},{"tree",tree},{"malformed_descriptor_rejections",malformed},
        {"all_controls_passed",true},{"full_attention_or_accepted_commit_implemented",false},{"model_or_speed_accepted",false}}).dump()<<std::endl;
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}}
