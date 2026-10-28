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
#include <limits>
#include "../../src/cuda/qwen35_tree_commit.cuh"
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
int readStatus(Buffer& b){auto values=b.read();int value;std::memcpy(&value,values.data(),4);return value;}
template<class T> void typed(Buffer& b,const std::vector<T>& v){require(v.size()*sizeof(T)==b.n*4,"Typed buffer size");std::vector<float> words(b.n);std::memcpy(words.data(),v.data(),b.n*4);b.upload(words);}
int main(int argc,char** argv){try{
    require(argc==5,"Expected fixture,group,model,output");std::ifstream input(argv[1]);json all;input>>all;json group;
    for(auto& x:all.at("groups"))if(x.at("name")==argv[2])group=x;require(!group.is_null(),"Missing fixture");
    std::string model=argv[3];require(model=="2b"||model=="4b","Model");int b=group.at("batch"),n=group.at("nodes");require(n==32||n==64,"Nodes");
    Garnet::Qwen35DeltaConfig dc{16,model=="2b"?16:32,128,128,4,1e-6f};int ch=(2*dc.kh+dc.vh)*128,kv=model=="2b"?2:4;
    Garnet::Qwen35TreeCommitConfig c{};c.batch=b;c.nodes=n;c.capacity=512;c.key_heads=dc.kh;c.value_heads=dc.vh;c.kernel=4;c.kv_heads=kv;c.head_dim=256;c.linear_layers=c.full_layers=1;
    int64_t hw=int64_t(ch)*4,sw=int64_t(dc.vh)*128*128,cache=int64_t(kv)*512*256;
    int64_t lp=int64_t(n)*(ch+2*dc.vh),kp=int64_t(n)*kv*256;
    c.hidden_width=int64_t(n)*(model=="2b"?2048:2560);c.state_width=hw+sw+2*cache;c.status_offset=c.hidden_width+lp+2*kp;c.metadata_offset=c.status_offset+2;c.packed_width=c.metadata_offset+836;
    require(Garnet::ValidQwen35TreeCommitConfig(c),"Config");
    Inputs delta({size_t(b)*n*ch,size_t(b)*n*dc.vh*128,size_t(b)*n*dc.vh,size_t(b)*n*dc.vh,size_t(b)*hw,size_t(b)*sw,size_t(b)*n,size_t(ch)*4,size_t(dc.vh),size_t(dc.vh),128});
    Buffer packet(size_t(b)*c.packed_width),canonical(size_t(b)*c.state_width),parentsGpu(size_t(b)*n),prefixGpu(b),linearGpu(6),fullGpu(6),parameterGpu(ch*4+2*dc.vh),selectionGpu(16),workspace(Garnet::Qwen35TreeCommitWorkspace(c)/4),status(1);
    std::vector<Garnet::Qwen35TreeCommitLayer> linear{{0,c.hidden_width,0}},full{{hw+sw,c.hidden_width+lp,1}};
    typed(linearGpu,linear);typed(fullGpu,full);
    auto* selected=reinterpret_cast<Garnet::Qwen35TreeCommitSelection*>(selectionGpu.data());
    cudaStream_t stream{};check(cudaStreamCreate(&stream));
    auto run=[&]{return Garnet::RunQwen35TreeCommit(c,selected,packet.data(),canonical.data(),parentsGpu.data(),(const float*)delta.ptr[6],prefixGpu.data(),
        (const Garnet::Qwen35TreeCommitLayer*)linearGpu.data(),(const Garnet::Qwen35TreeCommitLayer*)fullGpu.data(),parameterGpu.data(),workspace.data(),(int*)status.data(),stream);};
    std::ofstream log(argv[4]);require(bool(log),"Output");int device;check(cudaGetDevice(&device));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,device));
    char uuid[80];auto u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);require(std::string(uuid)=="GPU-472bb21a-0ca8-9d60-47a0-c527bb3d2d81","Wrong GPU");
    log<<json({{"event","identity"},{"process_id",_getpid()},{"gpu_uuid",uuid},{"model",model},{"batch",b},{"nodes",n},{"capacity",512},{"linear_layers",1},{"full_layers",1},{"trained_checkpoint_payload_used",false},{"speed_memory_or_full_goal_accepted",false}}).dump()<<std::endl;
    cudaGraph_t graph{};cudaGraphExec_t exec{};size_t graphNodes=0;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));require(run()==0,"Capture");check(cudaStreamEndCapture(stream,&graph));check(cudaGraphGetNodes(graph,nullptr,&graphNodes));require(graphNodes==8,"Expected8 captured commit nodes");check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    const int t=8;std::vector<std::unique_ptr<Buffer>> seq;for(int width:{ch,dc.vh*128,dc.vh,dc.vh})seq.push_back(std::make_unique<Buffer>(size_t(b)*t*width));
    Buffer flags(size_t(b)*t),out(size_t(b)*t*dc.vh*128),history(size_t(b)*hw),state(size_t(b)*sw),serialWork(size_t(b)*t*ch);
    for(int scenario=0;scenario<4;++scenario){
        auto scene=group.at("scenarios").at(scenario);auto parents=scene.at("parents").get<std::vector<float>>(),active=scene.at("active").get<std::vector<float>>(),prefix=scene.at("prefix").get<std::vector<float>>();
        auto metadata=scene.at("expected_int32").get<std::vector<int32_t>>();std::mt19937 rng(882201+scenario);std::uniform_real_distribution<float> random(-.2f,.2f);
        for(auto& values:delta.x)for(auto& v:values)v=random(rng);for(auto& v:delta.x[5])v*=.1f;for(auto& v:delta.x[10])v+=1;delta.x[6]=active;delta.upload();parentsGpu.upload(parents);prefixGpu.upload(prefix);
        std::vector<float> source(size_t(b)*c.packed_width),base(size_t(b)*c.state_width);for(auto& v:source)v=random(rng);for(auto& v:base)v=random(rng);
        std::vector<float> parameters=delta.x[7];parameters.insert(parameters.end(),delta.x[8].begin(),delta.x[8].end());parameters.insert(parameters.end(),delta.x[9].begin(),delta.x[9].end());parameterGpu.upload(parameters);
        for(int row=0;row<b;++row){size_t ps=size_t(row)*c.packed_width,cs=size_t(row)*c.state_width;
            std::copy_n(delta.x[4].data()+size_t(row)*hw,hw,base.data()+cs);std::copy_n(delta.x[5].data()+size_t(row)*sw,sw,base.data()+cs+hw);
            std::copy_n(delta.x[0].data()+size_t(row)*n*ch,size_t(n)*ch,source.data()+ps+c.hidden_width);
            std::copy_n(delta.x[2].data()+size_t(row)*n*dc.vh,size_t(n)*dc.vh,source.data()+ps+c.hidden_width+size_t(n)*ch);
            std::copy_n(delta.x[3].data()+size_t(row)*n*dc.vh,size_t(n)*dc.vh,source.data()+ps+c.hidden_width+size_t(n)*(ch+dc.vh));
            source[ps+c.status_offset]=source[ps+c.status_offset+1]=0;
            for(int j=0;j<836;++j)source[ps+c.metadata_offset+j]=float(metadata[row*836+j]);
        }
        packet.upload(source);Garnet::Qwen35TreeCommitSelection selection{};std::vector<float> expected;
        size_t compared=0,cpuWords=0,replays=0;double maxFraction=0;
        for(int target=-1;target<n;++target){
            std::vector<std::vector<float>> path(4);int widths[]={ch,dc.vh*128,dc.vh,dc.vh};for(int i=0;i<4;++i)path[i].resize(size_t(b)*t*widths[i]);std::vector<float> mask(size_t(b)*t);expected=base;
            for(int row=0;row<b;++row){selection.prefix[row]=int(prefix[row]);selection.last[row]=target<0?-1:(target+row*7)%n;std::vector<int> nodes;
                for(int node=selection.last[row];node>=0;node=int(parents[row*n+node]))nodes.push_back(node);std::reverse(nodes.begin(),nodes.end());require(nodes.size()<=8,"Path bound");int slot=int(prefix[row]);
                for(size_t j=0;j<nodes.size();++j){int node=nodes[j];mask[row*t+j]=active[row*n+node];for(int i=0;i<4;++i)std::copy_n(delta.x[i].data()+(size_t(row)*n+node)*widths[i],widths[i],path[i].data()+(size_t(row)*t+j)*widths[i]);
                    if(!mask[row*t+j])continue;for(int head=0;head<kv;++head)for(int d=0;d<256;++d)for(int which=0;which<2;++which){size_t dst=size_t(row)*c.state_width+hw+sw+size_t(which)*cache+(size_t(head)*512+slot)*256+d;
                        size_t src=size_t(row)*c.packed_width+c.hidden_width+lp+size_t(which)*kp+(size_t(node)*kv+head)*256+d;expected[dst]=source[src];}++slot;
                }
            }
            auto in=delta.ptr;for(int i=0;i<4;++i){seq[i]->upload(path[i]);in[i]=seq[i]->data();}flags.upload(mask);in[6]=flags.data();void* outputs[]={out.data(),history.data(),state.data()};
            require(Garnet::RunQwen35DeltaRegister128(dc,b,t,in.data(),outputs,serialWork.data(),stream)==0,"Original serial state reference");check(cudaStreamSynchronize(stream));auto eh=history.read(),es=state.read();
            for(int row=0;row<b;++row){std::copy_n(eh.data()+size_t(row)*hw,hw,expected.data()+size_t(row)*c.state_width);std::copy_n(es.data()+size_t(row)*sw,sw,expected.data()+size_t(row)*c.state_width+hw);}
            typed(selectionGpu,std::vector<Garnet::Qwen35TreeCommitSelection>{selection});canonical.upload(base);require(run()==0,"Plain selected commit");check(cudaStreamSynchronize(stream));
            require(readStatus(status)==0,"Valid commit status");require(same(canonical.read(),expected),"Full canonical words differ from original serial/copy oracle");compared+=expected.size();
            auto rawMeta=workspace.read();require(!std::memcmp(rawMeta.data(),metadata.data(),metadata.size()*4),"Private metadata differs from Python");
            for(int replay=0;replay<2;++replay){canonical.upload(base);check(cudaMemsetAsync(workspace.data(),0xff,workspace.n*4,stream));check(cudaMemsetAsync(status.data(),0xff,4,stream));check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
                require(readStatus(status)==0&&same(canonical.read(),expected),"Dynamic selected captured replay changed canonical words");compared+=expected.size();++replays;}
            if(target==n-1){auto inputs=delta.x;for(int i=0;i<4;++i)inputs[i]=path[i];inputs[6]=mask;auto oracle=delta_reference(dc,b,t,inputs);
                for(int which=1;which<3;++which){const auto& got=which==1?eh:es;require(got.size()==oracle[which].size(),"FP64 shape");for(size_t j=0;j<got.size();++j){double error=std::abs(double(got[j])-oracle[which][j]),bound=2e-5+2e-4*std::abs(oracle[which][j]);require(error<=bound,"Original independent FP64 state bound failure");maxFraction=std::max(maxFraction,error/bound);++cpuWords;}}
            }
        }
        int gpuBad=0,recoveries=0;
        for(int bad=0;bad<11;++bad){auto pp=parents,aa=active,ss=prefix,pk=source;auto ls=linear,fs=full;auto sel=selection;int row=b-1;
            if(bad==0)pp[row*n]=0;else if(bad==1)aa[row*n]=.5f;else if(bad==2)ss[row]=-1;
            else if(bad==3)pk[size_t(row)*c.packed_width+c.status_offset]=1;else if(bad==4)pk[size_t(row)*c.packed_width+c.metadata_offset]=2;
            else if(bad==5)pk[size_t(row)*c.packed_width]=std::numeric_limits<float>::quiet_NaN();else if(bad==6)pk[size_t(row)*c.packed_width+c.hidden_width+(n-1)*ch]=std::numeric_limits<float>::quiet_NaN();
            else if(bad==7)sel.last[row]=n;else if(bad==8)++sel.prefix[row];else if(bad==9)ls[0].state=INT64_MAX;else fs[0].payload=ls[0].payload;
            parentsGpu.upload(pp);delta.gpu[6]->upload(aa);prefixGpu.upload(ss);packet.upload(pk);typed(linearGpu,ls);typed(fullGpu,fs);typed(selectionGpu,std::vector<Garnet::Qwen35TreeCommitSelection>{sel});canonical.upload(base);
            check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(readStatus(status)==1,"Bad GPU packet/layout/selection not rejected");require(same(canonical.read(),base),"Bad GPU input advanced ANY canonical row");++gpuBad;
            parentsGpu.upload(parents);delta.gpu[6]->upload(active);prefixGpu.upload(prefix);packet.upload(source);typed(linearGpu,linear);typed(fullGpu,full);typed(selectionGpu,std::vector<Garnet::Qwen35TreeCommitSelection>{selection});
            check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(readStatus(status)==0&&same(canonical.read(),expected),"Bad-to-valid graph recovery");++recoveries;
        }
        struct Call{Garnet::Qwen35TreeCommitConfig c;const Garnet::Qwen35TreeCommitSelection* s;const float *packet,*parents,*active,*prefix;float* canonical;const Garnet::Qwen35TreeCommitLayer *linear,*full;const float* parameters;void* workspace;int* status;};
        Call valid{c,selected,packet.data(),parentsGpu.data(),(const float*)delta.ptr[6],prefixGpu.data(),canonical.data(),(const Garnet::Qwen35TreeCommitLayer*)linearGpu.data(),(const Garnet::Qwen35TreeCommitLayer*)fullGpu.data(),parameterGpu.data(),workspace.data(),(int*)status.data()};
        std::vector<Call> badCalls;for(int kind=0;kind<29;++kind){auto x=valid;
            switch(kind){case 0:x.c.nodes=0;break;case 1:x.c.batch=3;break;case 2:x.c.state_width=1;break;case 3:x.c.packed_width=1;break;case 4:++x.c.hidden_width;break;case 5:x.c.linear_layers=129;break;
            case 6:x.packet=nullptr;break;case 7:x.canonical=nullptr;break;case 8:x.parents=nullptr;break;case 9:x.active=nullptr;break;case 10:x.prefix=nullptr;break;case 11:x.linear=nullptr;break;case 12:x.full=nullptr;break;case 13:x.parameters=nullptr;break;case 14:x.s=nullptr;break;case 15:x.workspace=nullptr;break;case 16:x.status=nullptr;break;
            case 17:x.canonical=packet.data();break;case 18:x.workspace=canonical.data();break;case 19:x.status=(int*)parentsGpu.data();break;case 20:x.s=reinterpret_cast<const Garnet::Qwen35TreeCommitSelection*>((char*)selectionGpu.data()+1);break;case 21:x.linear=reinterpret_cast<const Garnet::Qwen35TreeCommitLayer*>((char*)linearGpu.data()+4);break;case 22:x.canonical=(float*)((char*)canonical.data()+1);break;
            case 23:x.c.value_heads=0;break;case 24:x.c.full_layers=129;break;case 25:x.c.capacity=0;break;case 26:x.c.kernel=5;break;case 27:x.c.key_heads=0;break;case 28:x.c.metadata_offset=0;break;}
            badCalls.push_back(x);
        }
        canonical.upload(base);check(cudaMemsetAsync(workspace.data(),0x33,workspace.n*4,stream));check(cudaMemsetAsync(status.data(),0x33,4,stream));check(cudaStreamSynchronize(stream));auto oldWorkspace=workspace.read(),oldStatus=status.read();
        for(auto& x:badCalls){require(Garnet::RunQwen35TreeCommit(x.c,x.s,x.packet,x.canonical,x.parents,x.active,x.prefix,x.linear,x.full,x.parameters,x.workspace,x.status,stream)!=0,"Bad host call accepted");
            require(same(canonical.read(),base)&&same(workspace.read(),oldWorkspace)&&same(status.read(),oldStatus),"Bad host call advanced output/workspace/status");}
        check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));require(same(canonical.read(),expected)&&readStatus(status)==0,"Final valid recovery");
        delta.preserved();require(same(packet.read(),source)&&same(parentsGpu.read(),parents)&&same(prefixGpu.read(),prefix)&&same(parameterGpu.read(),parameters),"Readonly packet/control/parameter changed");
        require(!std::memcmp(linearGpu.read().data(),linear.data(),sizeof(linear[0]))&&!std::memcmp(fullGpu.read().data(),full.data(),sizeof(full[0]))&&!std::memcmp(selectionGpu.read().data(),&selection,sizeof(selection)),"Readonly layout/selection changed");
        out.read();history.read();state.read();serialWork.read();flags.read();for(auto& v:seq)v->read();
        log<<json({{"event","scenario_complete"},{"scenario",scenario},{"seed",882201+scenario},{"endpoints_including_empty",n+1},{"canonical_words_exact_serial_copy",compared},
            {"independent_fp64_state_history_words",cpuWords},{"independent_fp64_max_bound_fraction",maxFraction},{"same_graph_dynamic_selection_replays",replays},
            {"gpu_bad_input_whole_batch_no_write",gpuBad},{"same_graph_invalid_to_valid_recoveries",recoveries},{"host_no_advance_rejections",badCalls.size()},
            {"all_private_metadata_words_exact_python",true},{"readonly_inputs_and_all_canaries_preserved",true}}).dump()<<std::endl;
    }
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));log<<json({{"event","complete"},{"all_controls_passed",true},{"learned_model_or_speed_memory_accepted",false}}).dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"TREE_COMMIT_NATIVE_CONTROL_FAILURE "<<e.what()<<std::endl;return 1;}}
