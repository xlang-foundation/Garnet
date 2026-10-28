// SPDX-License-Identifier: Apache-2.0
// Unwired branch-aware attention and accepted-path KV commit correctness.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <process.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../src/cuda/qwen35_delta.cu"
#include "../../src/cuda/qwen35_attention_core.cu"
#include "qwen35_tree_state.cuh"
#include "qwen35_tree_attention.cuh"
#include <nlohmann/json.hpp>
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

void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
bool same(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
std::vector<int> topology(const std::string& name){
    std::vector<int> p{-1};
    if(name=="chain8")for(int n=1;n<8;++n)p.push_back(n-1);
    else if(name=="ternary40")for(int n=1;n<40;++n)p.push_back((n-1)/3);
    else if(name=="star64")for(int n=1;n<64;++n)p.push_back(0);
    else if(name=="ragged32"){for(int n=1;n<8;++n)p.push_back(n-1);for(int n=8;n<32;++n)p.push_back((n-8)%7);}
    else throw std::runtime_error("Topology");return p;
}
std::vector<int> path(const Qwen35Tree::Descriptor& d,int node){std::vector<int> p;for(int n=node;n>=0;n=d.parent[n])p.push_back(n);std::reverse(p.begin(),p.end());return p;}
// CPU index/copy oracle, all bytes including untouched cache slots.
std::array<std::vector<float>,2> commit_cpu(Garnet::Qwen35AttentionCoreConfig c,const std::vector<std::vector<float>>& x,const Qwen35TreeAttention::Selection& s){
    std::array<std::vector<float>,2> result{x[4],x[5]};
    for(int b=0;b<c.batch;++b){int slot=s.prefix.start[b];for(int j=0;j<s.length[b];++j){int node=s.node[b][j];if(x[6][b*c.tokens+node]==0)continue;
        for(int h=0;h<c.kv;++h)for(int dim=0;dim<c.dim;++dim)for(int which=0;which<2;++which)
            result[which][((size_t(b)*c.kv+h)*c.capacity+slot)*c.dim+dim]=x[1+which][((size_t(b)*c.tokens+node)*c.kv+h)*c.dim+dim];++slot;
    }}return result;
}
// Independent scalar FP64 equations, no production CUDA helpers/reductions.
std::vector<double> cpu_attention(Garnet::Qwen35AttentionCoreConfig c,const std::vector<std::vector<float>>& y){
    std::array<std::vector<double>,2> cache{std::vector<double>(y[4].begin(),y[4].end()),std::vector<double>(y[5].begin(),y[5].end())};
    for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(y[7][b*c.tokens+t]!=0){int slot=int(y[6][b*c.tokens+t]);
        for(int h=0;h<c.kv;++h)for(int d=0;d<c.dim;++d)for(int w=0;w<2;++w)cache[w][((size_t(b)*c.kv+h)*c.capacity+slot)*c.dim+d]=y[1+w][((size_t(b)*c.tokens+t)*c.kv+h)*c.dim+d];
    }
    std::vector<double> output(size_t(c.batch)*c.tokens*c.heads*c.dim);
    for(int b=0;b<c.batch;++b)for(int t=0;t<c.tokens;++t)if(y[7][b*c.tokens+t]!=0)for(int h=0;h<c.heads;++h){
        int kh=h/(c.heads/c.kv),end=int(y[6][b*c.tokens+t])+1;std::vector<double> scores(end);double maximum=-INFINITY;
        auto query=y[0].data()+((size_t(b)*c.tokens+t)*c.heads+h)*c.dim;
        for(int pos=0;pos<end;++pos){double dot=0;for(int d=0;d<c.dim;++d)dot+=double(query[d])*cache[0][((size_t(b)*c.kv+kh)*c.capacity+pos)*c.dim+d];scores[pos]=dot/std::sqrt(double(c.dim));maximum=std::max(maximum,scores[pos]);}
        double sum=0;for(auto& value:scores){value=std::exp(value-maximum);sum+=value;}for(auto& value:scores)value/=sum;
        for(int d=0;d<c.dim;++d){double value=0;for(int pos=0;pos<end;++pos)value+=scores[pos]*cache[1][((size_t(b)*c.kv+kh)*c.capacity+pos)*c.dim+d];
            auto index=((size_t(b)*c.tokens+t)*c.heads+h)*c.dim+d;output[index]=value/(1+std::exp(-double(y[3][index])));
        }
    }return output;
}
int flag(Buffer& s){auto a=s.read();int value;std::memcpy(&value,a.data(),4);return value;}
int main(int argc,char** argv){try{
    require(argc==5,"Expected model,batch,topology,output");std::string model=argv[1],tree=argv[3];int b=std::stoi(argv[2]);
    require(model=="2b"||model=="4b","Model");require(b==1||b==2||b==4||b==8,"Batch");
    auto parents=topology(tree);Qwen35Tree::Descriptor d;require(Qwen35Tree::Build(parents.data(),int(parents.size()),d),"Descriptor");
    Garnet::Qwen35AttentionCoreConfig c{b,d.count,model=="2b"?8:16,model=="2b"?2:4,512,256};
    std::ofstream log(argv[4]);require(bool(log),"Output");int device;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
    char uuid[80];auto u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);
    std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
    require(std::string(uuid)=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Unexpected physical GPU");
    log<<json({{"event","identity"},{"process_id",_getpid()},{"model",model},{"batch",b},{"tree",tree},{"nodes",d.count},{"leaves",d.leaf_count},
        {"heads",c.heads},{"kv",c.kv},{"dim",c.dim},{"capacity",c.capacity},{"gpu_uuid",uuid},{"parent",parents},
        {"verification_full_kv_copy_bytes",0},{"score_workspace_bytes",size_t(b)*c.tokens*c.heads*c.capacity*4},{"model_or_speed_accepted",false}}).dump()<<std::endl;
    for(int scenario=0;scenario<4;++scenario){
        int seed=991101+scenario;std::mt19937 rng(seed);std::uniform_real_distribution<float> random(-.2f,.2f);
        std::vector<size_t> sizes={size_t(b)*c.tokens*c.heads*c.dim,size_t(b)*c.tokens*c.kv*c.dim,size_t(b)*c.tokens*c.kv*c.dim,
            size_t(b)*c.tokens*c.heads*c.dim,size_t(b)*c.kv*c.capacity*c.dim,size_t(b)*c.kv*c.capacity*c.dim,size_t(b)*c.tokens};
        std::vector<std::vector<float>> x;std::vector<std::unique_ptr<Buffer>> inputs;std::vector<const void*> in;
        for(auto n:sizes){std::vector<float> v(n);for(auto& value:v)value=random(rng);x.push_back(std::move(v));}
        Qwen35TreeAttention::Prefix prefix;
        for(int row=0;row<b;++row){prefix.start[row]=scenario==0?85:scenario==2?504:0;
            for(int node=0;node<c.tokens;++node)x[6][row*c.tokens+node]=scenario==3?0:scenario==1?float(!(row%4==3||node%5==2)):1;
        }
        for(auto& v:x){inputs.push_back(std::make_unique<Buffer>(v.size()));inputs.back()->upload(v);in.push_back(inputs.back()->data());}
        Buffer output(sizes[0]),scratch(size_t(b)*c.tokens*c.heads*c.capacity);cudaStream_t stream{};check(cudaStreamCreate(&stream));
        auto run=[&]{require(Qwen35TreeAttention::Run(c,d,prefix,in.data(),output.data(),scratch.data(),stream)==0,"Tree attention launch");};
        auto poison=[&]{check(cudaMemsetAsync(output.data(),0xff,output.n*4,stream));check(cudaMemsetAsync(scratch.data(),0xff,scratch.n*4,stream));};
        poison();run();check(cudaStreamSynchronize(stream));auto actual=output.read();require(std::all_of(actual.begin(),actual.end(),[](float v){return std::isfinite(v);}),"Nonfinite tree output");
        size_t compared=0,cpu_words=0,native_cache_words=0;double max_abs=0,max_fraction=0;
        for(int leaf=0;leaf<d.leaf_count;++leaf){
            auto nodes=path(d,d.leaf[leaf]);int length=int(nodes.size());auto cfg=c;cfg.tokens=length;
            std::vector<std::vector<float>> y(8);int width[]={c.heads*c.dim,c.kv*c.dim,c.kv*c.dim,c.heads*c.dim};
            for(int i=0;i<4;++i){y[i].resize(size_t(b)*length*width[i]);for(int row=0;row<b;++row)for(int t=0;t<length;++t)
                std::copy_n(x[i].data()+(size_t(row)*c.tokens+nodes[t])*width[i],width[i],y[i].data()+(size_t(row)*length+t)*width[i]);}
            y[4]=x[4];y[5]=x[5];y[6].resize(b*length);y[7].resize(b*length);
            for(int row=0;row<b;++row){int slot=prefix.start[row];for(int t=0;t<length;++t){y[6][row*length+t]=float(slot);y[7][row*length+t]=x[6][row*c.tokens+nodes[t]];slot+=y[7][row*length+t]!=0;}}
            std::vector<std::unique_ptr<Buffer>> serial;std::vector<const void*> args;
            for(auto& v:y){serial.push_back(std::make_unique<Buffer>(v.size()));serial.back()->upload(v);args.push_back(serial.back()->data());}
            Buffer h(size_t(b)*length*c.heads*c.dim),k(sizes[4]),v(sizes[5]),work(size_t(b)*length*c.heads*c.capacity);
            void* outs[]={h.data(),k.data(),v.data()};require(Garnet::RunQwen35AttentionCore(cfg,args.data(),outs,work.data(),stream)==0,"Native serial attention failed");check(cudaStreamSynchronize(stream));
            auto expected=h.read();
            for(int row=0;row<b;++row)for(int t=0;t<length;++t){auto a=actual.data()+(size_t(row)*c.tokens+nodes[t])*c.heads*c.dim;
                auto e=expected.data()+(size_t(row)*length+t)*c.heads*c.dim;require(!std::memcmp(a,e,size_t(c.heads)*c.dim*4),"Tree node differs from serial native attention");compared+=c.heads*c.dim;}
            int last[8]{};for(int row=0;row<b;++row)last[row]=d.leaf[leaf];Qwen35TreeAttention::Selection selected;
            require(Qwen35TreeAttention::BuildSelection(c,d,prefix,last,selected),"Native cache oracle selection");auto cpu_cache=commit_cpu(c,x,selected);
            require(same(k.read(),cpu_cache[0])&&same(v.read(),cpu_cache[1]),"Native cache layout differs from CPU oracle");native_cache_words+=2*sizes[4];
            for(size_t i=0;i<serial.size();++i)require(same(serial[i]->read(),y[i]),"Native reference input mutated");work.read();
            if(leaf==d.leaf_count-1){auto cpu=cpu_attention(cfg,y);for(size_t i=0;i<expected.size();++i){double error=std::abs(double(expected[i])-cpu[i]),bound=2e-5+2e-4*std::abs(cpu[i]);
                require(error<=bound,"Original independent FP64 bound failed");max_abs=std::max(max_abs,error);max_fraction=std::max(max_fraction,error/bound);++cpu_words;}}
        }
        // Perturb one terminal sibling's K/V. Every other node must remain exact.
        auto changedK=x[1],changedV=x[2];int changed=d.leaf[d.leaf_count-1];
        for(int row=0;row<b;++row)for(int h=0;h<c.kv;++h)for(int dim=0;dim<c.dim;++dim){auto i=((size_t(row)*c.tokens+changed)*c.kv+h)*c.dim+dim;changedK[i]+=128;changedV[i]+=256;}
        Buffer siblingK(sizes[1]),siblingV(sizes[2]);siblingK.upload(changedK);siblingV.upload(changedV);auto changedArgs=in;changedArgs[1]=siblingK.data();changedArgs[2]=siblingV.data();
        require(Qwen35TreeAttention::Run(c,d,prefix,changedArgs.data(),output.data(),scratch.data(),stream)==0,"Sibling perturb launch");check(cudaStreamSynchronize(stream));auto perturbed=output.read();size_t isolated=0;
        for(int row=0;row<b;++row)for(int node=0;node<c.tokens;++node)if(node!=changed){auto i=(size_t(row)*c.tokens+node)*c.heads*c.dim;require(!std::memcmp(actual.data()+i,perturbed.data()+i,size_t(c.heads)*c.dim*4),"Sibling influenced unrelated node");isolated+=c.heads*c.dim;}
        require(same(siblingK.read(),changedK)&&same(siblingV.read(),changedV),"Sibling inputs mutated");
        cudaGraph_t graph{};cudaGraphExec_t exec{};check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();check(cudaStreamEndCapture(stream,&graph));size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));require(count==3,"Expected three verification kernels");check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int replay=0;replay<3;++replay){poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(same(output.read(),actual),"Poisoned graph parity");}
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
        int rejected=0;auto before=output.read();auto bad=[&](Garnet::Qwen35AttentionCoreConfig cfg,Qwen35Tree::Descriptor desc,Qwen35TreeAttention::Prefix base,const void* const* args,float* out,float* work){
            require(Qwen35TreeAttention::Run(cfg,desc,base,args,out,work,stream)!=0,"Invalid verification accepted");++rejected;};
        auto cfg=c;cfg.batch=3;bad(cfg,d,prefix,in.data(),output.data(),scratch.data());cfg=c;cfg.dim=0;bad(cfg,d,prefix,in.data(),output.data(),scratch.data());
        cfg=c;cfg.tokens++;bad(cfg,d,prefix,in.data(),output.data(),scratch.data());auto desc=d;desc.owner[0]=d.leaf_count;bad(c,desc,prefix,in.data(),output.data(),scratch.data());
        auto base=prefix;base.start[0]=-1;bad(c,d,base,in.data(),output.data(),scratch.data());base=prefix;base.start[0]=512;bad(c,d,base,in.data(),output.data(),scratch.data());
        bad(c,d,prefix,nullptr,output.data(),scratch.data());bad(c,d,prefix,in.data(),nullptr,scratch.data());bad(c,d,prefix,in.data(),output.data(),nullptr);
        bad(c,d,prefix,in.data(),const_cast<float*>((const float*)in[0]),scratch.data());bad(c,d,prefix,in.data(),output.data(),const_cast<float*>((const float*)in[4]));bad(c,d,prefix,in.data(),output.data(),output.data());
        check(cudaStreamSynchronize(stream));require(same(output.read(),before),"Rejected verification advanced output");
        size_t committed_words=0;int commit_rejected=0,device_rejected=0;
        Buffer canonicalK(sizes[4]),canonicalV(sizes[5]),status(1),badActive(sizes[6]);
        for(int trial=0;trial<3;++trial){
            int last[8]{};auto base=prefix;
            for(int row=0;row<b;++row){last[row]=trial==0?d.leaf[0]:trial==1?(row%3==0?d.leaf[d.leaf_count-1]:row%3==1?0:-1):-1;if(trial==2)base.start[row]=512;}
            Qwen35TreeAttention::Selection selection;require(Qwen35TreeAttention::BuildSelection(c,d,base,last,selection),"Selection build");auto expected=commit_cpu(c,x,selection);
            auto commit=[&]{require(Qwen35TreeAttention::CommitSelected(c,d,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()),stream)==0,"Commit launch");};
            auto reset=[&]{canonicalK.upload(x[4]);canonicalV.upload(x[5]);};reset();commit();check(cudaStreamSynchronize(stream));require(flag(status)==0&&same(canonicalK.read(),expected[0])&&same(canonicalV.read(),expected[1]),"Selected KV differs from CPU oracle");committed_words+=2*sizes[4];
            if(trial==0){check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));commit();check(cudaStreamEndCapture(stream,&graph));check(cudaGraphGetNodes(graph,nullptr,&count));require(count==3,"Expected commit memset/check/kernel graph");check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                for(int replay=0;replay<3;++replay){reset();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(flag(status)==0&&same(canonicalK.read(),expected[0])&&same(canonicalV.read(),expected[1]),"Commit graph parity");committed_words+=2*sizes[4];}
                check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
                auto savedK=canonicalK.read(),savedV=canonicalV.read();
                auto badCommit=[&](Garnet::Qwen35AttentionCoreConfig cfg,const Qwen35TreeAttention::Selection& s,const float* key,const float* value,const float* active,float* k,float* v,int* e){
                    require(Qwen35TreeAttention::CommitSelected(cfg,d,s,key,value,active,k,v,e,stream)!=0,"Invalid commit accepted");++commit_rejected;};
                auto altered=selection;altered.node[0][0]=d.count;badCommit(c,altered,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                altered=selection;altered.length[0]=9;badCommit(c,altered,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                altered=selection;altered.prefix.start[0]=512;badCommit(c,altered,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                badCommit(c,selection,nullptr,(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                badCommit(c,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],nullptr,canonicalV.data(),reinterpret_cast<int*>(status.data()));
                badCommit(c,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),nullptr);
                badCommit(c,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalK.data(),reinterpret_cast<int*>(status.data()));
                badCommit(c,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],const_cast<float*>((const float*)in[1]),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                badCommit(c,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(canonicalK.data()));
                cfg=c;cfg.batch=3;badCommit(cfg,selection,(const float*)in[1],(const float*)in[2],(const float*)in[6],canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()));
                check(cudaStreamSynchronize(stream));require(same(canonicalK.read(),savedK)&&same(canonicalV.read(),savedV),"Rejected commit modified canonical storage");
            }
            for(float invalid:{.5f,std::nanf(""),INFINITY}){auto flags=x[6];flags.back()=invalid;badActive.upload(flags);reset();
                require(Qwen35TreeAttention::CommitSelected(c,d,selection,(const float*)in[1],(const float*)in[2],badActive.data(),canonicalK.data(),canonicalV.data(),reinterpret_cast<int*>(status.data()),stream)==0,"Bad mask kernel launch");check(cudaStreamSynchronize(stream));
                require(flag(status)==1&&same(canonicalK.read(),x[4])&&same(canonicalV.read(),x[5]),"Bad GPU mask permitted a canonical write");require(same(badActive.read(),flags),"Bad mask input mutated");++device_rejected;
            }
        }
        for(size_t i=0;i<inputs.size();++i)require(same(inputs[i]->read(),x[i]),"Tree verification/commit input mutation");scratch.read();output.read();canonicalK.read();canonicalV.read();status.read();check(cudaStreamDestroy(stream));
        log<<json({{"event","scenario_complete"},{"scenario",scenario},{"seed",seed},{"all_node_native_words_exact",true},{"compared_native_path_output_words",compared},
            {"native_cache_words_exact_cpu",native_cache_words},{"independent_fp64_words",cpu_words},{"independent_fp64_max_abs",max_abs},{"independent_fp64_max_bound_fraction",max_fraction},
            {"sibling_unrelated_words_exact",isolated},{"verification_graph_replays",3},{"commit_graph_replays",3},{"selected_canonical_words_exact_cpu",committed_words},
            {"verification_argument_rejections",rejected},{"commit_argument_rejections",commit_rejected},{"gpu_invalid_mask_no_write_rejections",device_rejected},
            {"all_inputs_canonical_bytes_canaries_preserved",true}}).dump()<<std::endl;
    }
    log<<json({{"event","complete"},{"model",model},{"batch",b},{"tree",tree},{"all_controls_passed",true},{"model_or_speed_or_process_memory_accepted",false}}).dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
