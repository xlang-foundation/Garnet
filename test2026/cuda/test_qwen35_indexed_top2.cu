// SPDX-License-Identifier: Apache-2.0
// Standalone, no serving. FP64 oracle fixed before GPU execution.
#include "../../src/cuda/qwen35_indexed_head.cuh"
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>std::vector<T> read(const json& file,size_t n){
    std::ifstream in(file.get<std::string>(),std::ios::binary|std::ios::ate);
    require(in&&in.tellg()==std::streamoff(n*sizeof(T)),"Fixture byte size mismatch");
    std::vector<T> out(n);in.seekg(0);in.read((char*)out.data(),n*sizeof(T));require(bool(in),"Fixture read failed");return out;
}
struct Buffer {
    unsigned char* raw=nullptr;size_t bytes;static constexpr size_t guard=256;
    explicit Buffer(size_t n):bytes(n){check(cudaMalloc((void**)&raw,n+2*guard));check(cudaMemset(raw,0xA7,n+2*guard));}
    ~Buffer(){if(raw)cudaFree(raw);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    void* data(){return raw+guard;}
    template<class T>T* ptr(){return (T*)data();}
    template<class T>void put(const std::vector<T>& v){require(v.size()*sizeof(T)==bytes,"Upload size");check(cudaMemcpy(data(),v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(){require(bytes%sizeof(T)==0,"Download size");std::vector<T> v(bytes/sizeof(T));check(cudaMemcpy(v.data(),data(),bytes,cudaMemcpyDeviceToHost));return v;}
    void checkGuard(){
        unsigned char ends[2*guard];check(cudaMemcpy(ends,raw,guard,cudaMemcpyDeviceToHost));check(cudaMemcpy(ends+guard,raw+guard+bytes,guard,cudaMemcpyDeviceToHost));
        for(auto x:ends)require(x==0xA7,"Buffer guard overwritten");
    }
};
template<class T>bool equal(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&!memcmp(a.data(),b.data(),a.size()*sizeof(T));}
float bf(uint16_t x){uint32_t bits=uint32_t(x)<<16;float value;memcpy(&value,&bits,4);return value;}
struct Graph {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Graph(){check(cudaStreamCreate(&stream));}
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
}
int main(int argc,char**argv){try{
    require(argc==2,"Expected fixtures.json");std::ifstream in(argv[1]);json cases;in>>cases;check(cudaSetDevice(0));int number=0;
    for(auto& item:cases){auto& cfg=item.at("config");Garnet::Qwen35IndexedHeadConfig c{cfg.at("rows"),cfg.at("hidden"),cfg.at("vocab"),cfg.at("candidates")};
        size_t count=size_t(c.rows)*c.candidates;auto& f=item.at("files");
        auto query=read<float>(f.at("queries"),size_t(c.rows)*c.hidden);auto table=read<uint16_t>(f.at("table"),size_t(c.vocab)*c.hidden);
        auto indices=read<int>(f.at("indices"),count);auto dots=read<double>(f.at("dots"),count);auto sumAbs=read<double>(f.at("sum_abs"),count);
        Buffer q(query.size()*4),w(table.size()*2),ix(count*4),scores(count*2),ids(c.rows*2*8),values(c.rows*2*4);
        q.put(query);w.put(table);ix.put(indices);Graph graph;
        auto run=[&](float* val){return Garnet::RunQwen35IndexedHeadTop2(c,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),val,graph.stream);};
        auto oracle=[&](const std::vector<int>& basket){
            auto actual=scores.get<uint16_t>();auto gotIds=ids.get<long long>();auto gotValues=values.get<float>();
            for(int row=0;row<c.rows;++row){
                std::vector<std::pair<float,int>> sorted;
                for(int j=0;j<c.candidates;++j){size_t i=size_t(row)*c.candidates+j;float value=bf(actual[i]);int id=basket[i];
                    if(id>=0&&id<c.vocab&&std::isfinite(value))sorted.emplace_back(value,id);
                }
                std::sort(sorted.begin(),sorted.end(),[](const auto&a,const auto&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
                std::set<int> used;std::vector<std::pair<float,int>> expected;
                for(auto value:sorted)if(used.insert(value.second).second)expected.push_back(value);
                for(int j=0;j<2;++j){int id=j<expected.size()?expected[j].second:-1;float value=j<expected.size()?expected[j].first:-INFINITY;
                    require(gotIds[2*row+j]==id&&gotValues[2*row+j]==value,"Independent CPU distinct top2/tie oracle mismatch");}
            }
        };
        require(run(values.ptr<float>())==0,"Top2 enqueue");check(cudaStreamSynchronize(graph.stream));oracle(indices);
        auto direct=scores.get<uint16_t>();auto directIds=ids.get<long long>();auto directValues=values.get<float>();
        double n=(c.hidden+255)/256+8,u=std::ldexp(1.,-24),gamma=n*u/(1-n*u),maxFraction=0;
        for(size_t i=0;i<count;++i){
            float actual=bf(direct[i]);if(indices[i]<0||indices[i]>=c.vocab){require(actual==-INFINITY,"Invalid index score");continue;}
            double bound=gamma*sumAbs[i]+(std::abs(dots[i])+gamma*sumAbs[i])/256.+std::ldexp(1.,-133);
            double error=std::abs(double(actual)-dots[i]);require(std::isfinite(actual)&&error<=bound,"Unchanged FP64 dot bound failed");
            maxFraction=std::max(maxFraction,error/std::max(bound,1e-300));
        }
        check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeGlobal));require(run(values.ptr<float>())==0,"Top2 capture");
        check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
        for(int replay=0;replay<3;++replay){check(cudaMemsetAsync(scores.data(),0,count*2,graph.stream));check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            require(equal(scores.get<uint16_t>(),direct)&&equal(ids.get<long long>(),directIds)&&equal(values.get<float>(),directValues),"Top2 replay changed");}
        require(run(nullptr)==0,"Optional values rejected");check(cudaStreamSynchronize(graph.stream));require(equal(ids.get<long long>(),directIds),"Optional values changed IDs");
        auto reject=[&](Garnet::Qwen35IndexedHeadConfig bad,const float* qp,const void* wp,const int* ip,void* sp,long long* out,float* val){
            require(Garnet::RunQwen35IndexedHeadTop2(bad,qp,wp,ip,sp,out,val,graph.stream)!=0,"Invalid top2 accepted");};
        for(int which=0;which<8;++which){auto bad=c;
            if(which==0)bad.rows=0;if(which==1)bad.rows=9;if(which==2)bad.hidden=0;if(which==3)bad.hidden=16385;
            if(which==4)bad.vocab=1048577;if(which==5)bad.candidates=0;if(which==6)bad.candidates=65537;if(which==7)bad.candidates=c.vocab+1;
            reject(bad,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());}
        reject(c,nullptr,w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,q.ptr<float>(),nullptr,ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,q.ptr<float>(),w.data(),ix.ptr<int>(),q.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),scores.ptr<float>());
        reject(c,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),ids.ptr<float>());
        reject(c,reinterpret_cast<const float*>(UINTPTR_MAX-3),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,reinterpret_cast<const float*>(q.raw+Buffer::guard+1),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        check(cudaStreamSynchronize(graph.stream));require(equal(scores.get<uint16_t>(),direct)&&equal(ids.get<long long>(),directIds)&&equal(values.get<float>(),directValues),"Rejection mutated output");
        require(equal(q.get<float>(),query)&&equal(w.get<uint16_t>(),table)&&equal(ix.get<int>(),indices),"Top2 input mutation");
        auto reversed=indices;
        for(int row=0;row<c.rows;++row)std::reverse(reversed.begin()+row*c.candidates,reversed.begin()+(row+1)*c.candidates);
        ix.put(reversed);require(run(values.ptr<float>())==0,"Reverse basket enqueue");check(cudaStreamSynchronize(graph.stream));oracle(reversed);
        require(equal(ids.get<long long>(),directIds)&&equal(values.get<float>(),directValues),"Basket order changed top2");
        q.put(std::vector<float>(query.size(),0));require(run(values.ptr<float>())==0,"Tie enqueue");check(cudaStreamSynchronize(graph.stream));oracle(reversed);
        auto duplicate=std::vector<int>(count,3);ix.put(duplicate);require(run(values.ptr<float>())==0,"Duplicate enqueue");check(cudaStreamSynchronize(graph.stream));oracle(duplicate);
        for(int row=0;row<c.rows;++row)require(ids.get<long long>()[row*2+1]==-1,"Duplicate ID became second");
        std::fill(duplicate.begin(),duplicate.end(),-1);ix.put(duplicate);require(run(values.ptr<float>())==0,"All invalid enqueue");check(cudaStreamSynchronize(graph.stream));oracle(duplicate);
        ix.put(indices);q.put(std::vector<float>(query.size(),std::numeric_limits<float>::quiet_NaN()));require(run(values.ptr<float>())==0,"Nonfinite query enqueue");check(cudaStreamSynchronize(graph.stream));oracle(indices);
        for(auto id:ids.get<long long>())require(id==-1,"Nonfinite won top2");
        q.put(query);auto only=c;only.candidates=1;
        require(Garnet::RunQwen35IndexedHeadTop2(only,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>(),graph.stream)==0,"One-candidate enqueue");
        check(cudaStreamSynchronize(graph.stream));auto onlyIds=ids.get<long long>();
        for(int row=0;row<c.rows;++row){require(onlyIds[row*2+1]==-1,"One candidate has second");int id=indices[row];require(onlyIds[row*2]==(id>=0&&id<c.vocab?id:-1),"One candidate winner");}
        require(equal(q.get<float>(),query)&&equal(w.get<uint16_t>(),table)&&equal(ix.get<int>(),indices),"Final input mutation");
        for(Buffer* p:{&q,&w,&ix,&scores,&ids,&values})p->checkGuard();
        std::cout<<json{{"case",number++},{"config",cfg},{"max_fp64_bound_fraction",maxFraction},{"exact_cpu_distinct_top2_oracle",true},{"three_exact_graph_replays",true},{"tie_duplicate_reverse_invalid_nonfinite_single_alias_guard",true},{"primary_speed_or_model_quality",false}}.dump()<<"\n";
    }
    std::cout<<"QWEN35_INDEXED_TOP2_PASS "<<number<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
