// SPDX-License-Identifier: Apache-2.0
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
#include "../../src/cuda/qwen35_tree_kernels.cuh"
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
// Independent scalar FP64 equations, fixed bounds shared by both native paths.
std::array<std::vector<double>,3> delta_reference(Garnet::Qwen35DeltaConfig c,int b,int t,const std::vector<std::vector<float>>& x){
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

// Independent scalar FP64 equations, no production CUDA helpers/reductions.
std::vector<double> attention_reference(Garnet::Qwen35AttentionCoreConfig c,const std::vector<std::vector<float>>& y){
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

void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
bool same(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
struct Inputs {
    std::vector<std::vector<float>> x;std::vector<std::unique_ptr<Buffer>> gpu;std::vector<const void*> ptr;
    explicit Inputs(const std::vector<size_t>& sizes){for(auto n:sizes){x.emplace_back(n);gpu.push_back(std::make_unique<Buffer>(n));ptr.push_back(gpu.back()->data());}}
    void upload(){for(size_t i=0;i<x.size();++i)gpu[i]->upload(x[i]);}
    void preserved(){for(size_t i=0;i<x.size();++i)require(same(gpu[i]->read(),x[i]),"Readonly native input changed");}
};
int main(int argc,char** argv){try{
    require(argc==5,"Expected fixtures,group,model,output");std::ifstream f(argv[1]);json all;f>>all;json group;
    for(auto& g:all.at("groups"))if(g.at("name")==argv[2])group=g;require(!group.is_null(),"Fixture group missing");std::string model=argv[3];require(model=="2b"||model=="4b","Model");
    int b=group.at("batch"),n=group.at("nodes");require(n==32||n==64,"Consumer profile");
    Garnet::Qwen35TreeMetadataConfig p{b,n,512};Garnet::Qwen35DeltaConfig dc{16,model=="2b"?16:32,128,128,4,1e-6f};
    Garnet::Qwen35AttentionCoreConfig ac{b,n,model=="2b"?8:16,model=="2b"?2:4,512,256};int ch=2*dc.kh*128+dc.vh*128;
    Inputs delta({size_t(b)*n*ch,size_t(b)*n*dc.vh*128,size_t(b)*n*dc.vh,size_t(b)*n*dc.vh,size_t(b)*ch*4,size_t(b)*dc.vh*128*128,size_t(b)*n,size_t(ch)*4,size_t(dc.vh),size_t(dc.vh),128});
    Inputs attn({size_t(b)*n*ac.heads*256,size_t(b)*n*ac.kv*256,size_t(b)*n*ac.kv*256,size_t(b)*n*ac.heads*256,size_t(b)*ac.kv*512*256,size_t(b)*ac.kv*512*256,size_t(b)*n});
    // Both consumers and the metadata producer read the same active tensor.
    attn.ptr[6]=delta.ptr[6];
    Buffer parent(size_t(b)*n),prefix(b),meta(size_t(b)*836),dout(delta.x[1].size()),aout(attn.x[0].size()),ds(b),as(b),
        dw(Garnet::Qwen35TreeDeltaWorkspace(dc,p)/4),aw(Garnet::Qwen35TreeAttentionWorkspace(ac)/4);
    auto mp=reinterpret_cast<Garnet::Qwen35TreeMetadata*>(meta.data());cudaStream_t stream{};check(cudaStreamCreate(&stream));
    auto run=[&]{require(Garnet::RunQwen35TreeMetadata(p,(float*)parent.data(),(float*)delta.ptr[6],(float*)prefix.data(),mp,stream)==0,"Metadata launch");
        require(Garnet::RunQwen35TreeDelta(dc,p,delta.ptr.data(),mp,dout.data(),ds.data(),dw.data(),stream)==0,"Native tree delta launch");
        require(Garnet::RunQwen35TreeAttention(ac,attn.ptr.data(),mp,aout.data(),as.data(),aw.data(),stream)==0,"Native tree attention launch");};
    auto poison=[&]{for(auto* v:{&dout,&aout,&ds,&as,&dw,&aw,&meta})check(cudaMemsetAsync(v->data(),0xff,v->n*4,stream));};
    std::ofstream log(argv[4]);require(bool(log),"Output");int device;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
    char uuid[80];auto u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);require(std::string(uuid)=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Wrong physical GPU");
    log<<json({{"event","identity"},{"process_id",_getpid()},{"gpu_uuid",uuid},{"model",model},{"batch",b},{"nodes",n},{"capacity",512},
        {"native_per_row_consumers",true},{"metadata_profile_bound_by_caller",true},{"serving_dll_modified",false},{"model_or_speed_accepted",false}}).dump()<<std::endl;
    cudaGraph_t graph{};cudaGraphExec_t exec{};size_t nodes=0;
    for(int scenario=0;scenario<4;++scenario){
        const auto& scene=group.at("scenarios").at(scenario);auto parents=scene.at("parents").get<std::vector<float>>(),active=scene.at("active").get<std::vector<float>>(),starts=scene.at("prefix").get<std::vector<float>>();
        auto expectedMeta=scene.at("expected_int32").get<std::vector<int32_t>>();std::mt19937 rng(772101+scenario);std::uniform_real_distribution<float> random(-.2f,.2f);
        for(auto* input:{&delta,&attn})for(auto& values:input->x)for(auto& value:values)value=random(rng);
        for(auto& value:delta.x[5])value*=.1f;for(auto& value:delta.x[10])value+=1;delta.x[6]=active;attn.x[6]=active;delta.upload();attn.upload();parent.upload(parents);prefix.upload(starts);
        poison();run();check(cudaStreamSynchronize(stream));auto dx=dout.read(),ax=aout.read();auto metadata=meta.read();
        require(!std::memcmp(metadata.data(),expectedMeta.data(),metadata.size()*4),"Native producer differs from Python metadata");
        require(std::all_of(dx.begin(),dx.end(),[](float v){return std::isfinite(v);})&&std::all_of(ax.begin(),ax.end(),[](float v){return std::isfinite(v);}),"Missing/nonfinite native consumer output");
        for(float value:ds.read())require(value==0,"Valid delta status");for(float value:as.read())require(value==0,"Valid attention status");
        if(scenario==0){check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();check(cudaStreamEndCapture(stream,&graph));check(cudaGraphGetNodes(graph,nullptr,&nodes));require(nodes==7,"Expected metadata2/delta2/attention3 kernels");check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));}
        for(int replay=0;replay<3;++replay){poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(same(dout.read(),dx)&&same(aout.read(),ax),"Poisoned combined graph parity");for(float v:ds.read())require(v==0,"Replay delta status");for(float v:as.read())require(v==0,"Replay attention status");}
        // Independent serial root-to-node paths differ in every batch row.
        const int t=8;std::vector<std::unique_ptr<Buffer>> dseq,aseq;
        for(int w:{ch,dc.vh*128,dc.vh,dc.vh})dseq.push_back(std::make_unique<Buffer>(size_t(b)*t*w));
        for(int w:{ac.heads*256,ac.kv*256,ac.kv*256,ac.heads*256})aseq.push_back(std::make_unique<Buffer>(size_t(b)*t*w));
        Buffer flags(b*t),slots(b*t),dh(size_t(b)*t*dc.vh*128),history(size_t(b)*ch*4),state(delta.x[5].size()),dwork(size_t(b)*t*ch),
            ah(size_t(b)*t*ac.heads*256),key(attn.x[4].size()),value(attn.x[5].size()),awork(size_t(b)*t*ac.heads*512);
        size_t dwords=0,awords=0,cpuwords=0;double maxAbs=0,maxFraction=0;
        for(int target=0;target<n;++target){
            std::vector<std::vector<int>> paths(b);std::vector<float> mask(b*t),slot(b*t);std::vector<std::vector<float>> qd(4),qa(4);
            int wd[]={ch,dc.vh*128,dc.vh,dc.vh},wa[]={ac.heads*256,ac.kv*256,ac.kv*256,ac.heads*256};
            for(int i=0;i<4;++i){qd[i].resize(size_t(b)*t*wd[i]);qa[i].resize(size_t(b)*t*wa[i]);}
            for(int row=0;row<b;++row){for(int node=target;node>=0;node=int(parents[row*n+node]))paths[row].push_back(node);std::reverse(paths[row].begin(),paths[row].end());int current=int(starts[row]);
                for(int j=0;j<t;++j){slot[row*t+j]=float(current);if(j>=int(paths[row].size()))continue;int node=paths[row][j];mask[row*t+j]=active[row*n+node];current+=mask[row*t+j]!=0;
                    for(int i=0;i<4;++i){std::copy_n(delta.x[i].data()+(size_t(row)*n+node)*wd[i],wd[i],qd[i].data()+(size_t(row)*t+j)*wd[i]);std::copy_n(attn.x[i].data()+(size_t(row)*n+node)*wa[i],wa[i],qa[i].data()+(size_t(row)*t+j)*wa[i]);}}
            }
            auto din=delta.ptr;std::vector<const void*> ain(8);for(int i=0;i<4;++i){dseq[i]->upload(qd[i]);aseq[i]->upload(qa[i]);din[i]=dseq[i]->data();ain[i]=aseq[i]->data();}
            flags.upload(mask);slots.upload(slot);din[6]=flags.data();ain[4]=attn.ptr[4];ain[5]=attn.ptr[5];ain[6]=slots.data();ain[7]=flags.data();void* douts[]={dh.data(),history.data(),state.data()},*aouts[]={ah.data(),key.data(),value.data()};auto cfg=ac;cfg.tokens=t;
            require(Garnet::RunQwen35DeltaRegister128(dc,b,t,din.data(),douts,dwork.data(),stream)==0,"Serial delta reference");require(Garnet::RunQwen35AttentionCore(cfg,ain.data(),aouts,awork.data(),stream)==0,"Serial attention reference");check(cudaStreamSynchronize(stream));auto de=dh.read(),ae=ah.read();
            for(int row=0;row<b;++row){int index=int(paths[row].size())-1;auto di=(size_t(row)*n+target)*dc.vh*128,dr=(size_t(row)*t+index)*dc.vh*128;
                auto ai=(size_t(row)*n+target)*ac.heads*256,ar=(size_t(row)*t+index)*ac.heads*256;
                require(!std::memcmp(dx.data()+di,de.data()+dr,size_t(dc.vh)*128*4),"Per-row native delta differs from serial node path");require(!std::memcmp(ax.data()+ai,ae.data()+ar,size_t(ac.heads)*256*4),"Per-row native attention differs from serial node path");dwords+=dc.vh*128;awords+=ac.heads*256;
            }
            if(target==n-1){auto yd=delta.x;for(int i=0;i<4;++i)yd[i]=qd[i];yd[6]=mask;auto expectedD=delta_reference(dc,b,t,yd);
                std::vector<std::vector<float>> ya(8);for(int i=0;i<4;++i)ya[i]=qa[i];ya[4]=attn.x[4];ya[5]=attn.x[5];ya[6]=slot;ya[7]=mask;auto expectedA=attention_reference(cfg,ya);
                auto compare=[&](const std::vector<float>& got,const std::vector<double>& expected){require(got.size()==expected.size(),"FP64 shape");for(size_t i=0;i<got.size();++i){double error=std::abs(double(got[i])-expected[i]),bound=2e-5+2e-4*std::abs(expected[i]);require(error<=bound,"Original FP64 bound failure");maxAbs=std::max(maxAbs,error);maxFraction=std::max(maxFraction,error/bound);++cpuwords;}};
                compare(de,expectedD[0]);compare(ae,expectedA);
            }
        }
        history.read();state.read();key.read();value.read();dwork.read();awork.read();flags.read();slots.read();for(auto& v:dseq)v->read();for(auto& v:aseq)v->read();
        int gpu_bad=0;for(int bad=0;bad<3;++bad){auto pp=parents,aa=active,ss=starts;if(bad==0)pp[0]=0;else if(bad==1)aa[0]=.5f;else ss[0]=-1;
            parent.upload(pp);delta.gpu[6]->upload(aa);prefix.upload(ss);poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));auto wrongD=dout.read(),wrongA=aout.read(),statusD=ds.read(),statusA=as.read();
            require(statusD[0]>0&&statusA[0]>0,"Invalid producer row failed to gate consumer");
            require(std::all_of(wrongD.begin(),wrongD.begin()+size_t(n)*dc.vh*128,[](float v){return v==0;})&&std::all_of(wrongA.begin(),wrongA.begin()+size_t(n)*ac.heads*256,[](float v){return v==0;}),"Invalid row consumer output not zero");
            for(int row=1;row<b;++row){require(statusD[row]==0&&statusA[row]==0,"Invalid row changed neighbor status");require(!std::memcmp(wrongD.data()+size_t(row)*n*dc.vh*128,dx.data()+size_t(row)*n*dc.vh*128,size_t(n)*dc.vh*128*4),"Invalid row changed neighbor delta");require(!std::memcmp(wrongA.data()+size_t(row)*n*ac.heads*256,ax.data()+size_t(row)*n*ac.heads*256,size_t(n)*ac.heads*256*4),"Invalid row changed neighbor attention");}
            parent.upload(parents);delta.gpu[6]->upload(active);prefix.upload(starts);poison();check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(same(dout.read(),dx)&&same(aout.read(),ax),"Native invalid-to-valid graph reset");++gpu_bad;
        }
        auto beforeD=dout.read(),beforeA=aout.read(),beforeDS=ds.read(),beforeAS=as.read();int rejectedD=0,rejectedA=0;
        auto bd=[&](Garnet::Qwen35DeltaConfig c,Garnet::Qwen35TreeMetadataConfig cfg,const void* const* in,const Garnet::Qwen35TreeMetadata* m,float* out,float* status,void* work){require(Garnet::RunQwen35TreeDelta(c,cfg,in,m,out,status,work,stream)!=0,"Bad delta host call accepted");++rejectedD;};
        auto ba=[&](Garnet::Qwen35AttentionCoreConfig c,const void* const* in,const Garnet::Qwen35TreeMetadata* m,float* out,float* status,void* work){require(Garnet::RunQwen35TreeAttention(c,in,m,out,status,work,stream)!=0,"Bad attention host call accepted");++rejectedA;};
        auto badp=p;badp.batch=3;bd(dc,badp,delta.ptr.data(),mp,dout.data(),ds.data(),dw.data());auto badc=dc;badc.kd=64;bd(badc,p,delta.ptr.data(),mp,dout.data(),ds.data(),dw.data());
        bd(dc,p,nullptr,mp,dout.data(),ds.data(),dw.data());bd(dc,p,delta.ptr.data(),nullptr,dout.data(),ds.data(),dw.data());bd(dc,p,delta.ptr.data(),mp,nullptr,ds.data(),dw.data());bd(dc,p,delta.ptr.data(),mp,dout.data(),nullptr,dw.data());bd(dc,p,delta.ptr.data(),mp,dout.data(),ds.data(),nullptr);
        bd(dc,p,delta.ptr.data(),mp,const_cast<float*>((const float*)delta.ptr[0]),ds.data(),dw.data());bd(dc,p,delta.ptr.data(),mp,dout.data(),dout.data(),dw.data());bd(dc,p,delta.ptr.data(),mp,dout.data(),ds.data(),meta.data());
        auto bada=ac;bada.batch=3;ba(bada,attn.ptr.data(),mp,aout.data(),as.data(),aw.data());bada=ac;bada.dim=0;ba(bada,attn.ptr.data(),mp,aout.data(),as.data(),aw.data());
        ba(ac,nullptr,mp,aout.data(),as.data(),aw.data());ba(ac,attn.ptr.data(),nullptr,aout.data(),as.data(),aw.data());ba(ac,attn.ptr.data(),mp,nullptr,as.data(),aw.data());ba(ac,attn.ptr.data(),mp,aout.data(),nullptr,aw.data());ba(ac,attn.ptr.data(),mp,aout.data(),as.data(),nullptr);
        ba(ac,attn.ptr.data(),mp,const_cast<float*>((const float*)attn.ptr[0]),as.data(),aw.data());ba(ac,attn.ptr.data(),mp,aout.data(),aout.data(),aw.data());ba(ac,attn.ptr.data(),mp,aout.data(),as.data(),meta.data());
        check(cudaStreamSynchronize(stream));require(same(dout.read(),beforeD)&&same(aout.read(),beforeA)&&same(ds.read(),beforeDS)&&same(as.read(),beforeAS),"Bad host call advanced consumer outputs/status");
        delta.preserved();attn.preserved();require(same(parent.read(),parents)&&same(prefix.read(),starts),"Metadata controls modified");meta.read();dw.read();aw.read();
        log<<json({{"event","scenario_complete"},{"scenario",scenario},{"seed",772101+scenario},{"delta_node_words_exact",dwords},{"attention_node_words_exact",awords},
            {"all_metadata_words_exact_python",true},{"independent_fp64_words",cpuwords},{"independent_fp64_max_abs",maxAbs},{"independent_fp64_max_bound_fraction",maxFraction},
            {"poisoned_same_graph_valid_replays",3},{"gpu_invalid_rows_zero_and_isolated",gpu_bad},{"invalid_to_valid_graph_resets",3},
            {"delta_host_rejections_no_advance",rejectedD},{"attention_host_rejections_no_advance",rejectedA},{"readonly_canonical_inputs_and_all_guards_preserved",true}}).dump()<<std::endl;
    }
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));log<<json({{"event","complete"},{"all_controls_passed",true},{"actual_native_src_consumers",true},{"actual_tensorrt_or_model_tree_integration_accepted",false},{"model_speed_memory_or_full_goal_accepted",false}}).dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
