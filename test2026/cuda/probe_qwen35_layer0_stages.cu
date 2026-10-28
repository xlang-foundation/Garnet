// Diagnostic-only first-layer TensorRT component using unchanged production ops/plugins.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "qwen35_lowering.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
using namespace nvinfer1;
using Ops=Garnet::Qwen35TRTOps;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cu(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Logger:ILogger{void log(Severity s,const char* message) noexcept override{if(s<=Severity::kWARNING)std::cerr<<message<<'\n';}};
struct Buffer{
    void* p=nullptr;size_t bytes=0;
    explicit Buffer(size_t n):bytes(n){cu(cudaMalloc(&p,n));}
    ~Buffer(){if(p)cudaFree(p);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
std::vector<unsigned char> bytes(const std::string& name){
    std::ifstream f(name,std::ios::binary|std::ios::ate);need(bool(f),"Input file open failed");
    auto n=f.tellg();need(n>0,"Empty input file");std::vector<unsigned char> result(size_t(n),0);
    f.seekg(0);f.read((char*)result.data(),result.size());need(bool(f),"Input file truncated");return result;
}
std::vector<float> floats(const std::string& name){
    auto raw=bytes(name);need(raw.size()%4==0,"FP32 file alignment invalid");std::vector<float> result(raw.size()/4);
    std::memcpy(result.data(),raw.data(),raw.size());for(float x:result)need(std::isfinite(x),"Nonfinite FP32 fixture");return result;
}
size_t volume(Dims d){size_t result=1;for(int i=0;i<d.nbDims;++i){need(d.d[i]>0,"Nonstatic IO shape");result*=size_t(d.d[i]);}return result;}
std::string uuid(const cudaDeviceProp& p){
    std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)s<<'-';s<<std::setw(2)<<unsigned((unsigned char)p.uuid.bytes[i]);}return s.str();
}
size_t different(const float* a,const float* b,size_t n){size_t count=0;for(size_t i=0;i<n;++i){need(std::isfinite(a[i])&&std::isfinite(b[i]),"Nonfinite control output");count+=std::memcmp(a+i,b+i,4)!=0;}return count;}
void exclusive(const std::filesystem::path& p){need(!std::filesystem::exists(p),"Preserve prior component evidence");}

int main(int argc,char** argv){
    if(argc!=5)return 2;
    const std::filesystem::path output=argv[3];exclusive(output);
    json result={{"admitted_exact_original_reference",false},{"production_model_modified",false},{"chunks",json::array()},{"observations",json::array()}};
    auto save=[&]{std::ofstream f(output);f<<result.dump(2)<<'\n';};
    try{
        std::ifstream fixtureFile(argv[1]);json fixture;fixtureFile>>fixture;const int batch=std::stoi(argv[2]);need(batch==1||batch==2,"Batch must be1or2");
        result["batch"]=batch;result["fixture"]=std::filesystem::absolute(argv[1]).string();
        json c=fixture.at("config"),test;
        for(auto item:fixture.at("cases"))if(item.at("batch")==batch)test=item;
        need(!test.is_null(),"Reference case missing");
        const int h=c.at("hidden_size"),vh=c.at("linear_num_value_heads"),kh=c.at("linear_num_key_heads");
        const int kd=c.at("linear_key_head_dim"),vd=c.at("linear_value_head_dim"),kernel=c.at("linear_conv_kernel_dim");
        const int t=fixture.at("tokens_per_chunk"),tokens=test.at("tokens"),channels=2*kh*kd+vh*vd;
        const size_t historyWidth=size_t(channels)*kernel,stateWidth=size_t(vh)*kd*vd;
        need(h==2560&&t==64&&tokens==85&&kd==128&&vd==128,"Unexpected component geometry");
        cu(cudaSetDevice(0));cudaDeviceProp properties{};cu(cudaGetDeviceProperties(&properties,0));
        need(uuid(properties)==argv[4],"Unexpected GPU UUID");
        result["gpu_uuid"]=uuid(properties);result["gpu_name"]=properties.name;result["sm"]=properties.major*10+properties.minor;
        Logger logger;
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));need(bool(builder),"TensorRT builder unavailable");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U<<uint32_t(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());need(network&&config,"TensorRT network/config unavailable");
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,256ULL<<20);config->setBuilderOptimizationLevel(3);config->setMaxAuxStreams(0);
        config->setFlag(BuilderFlag::kREFIT_INDIVIDUAL);config->setFlag(BuilderFlag::kSTRIP_PLAN);
        std::deque<std::vector<float>> constants;std::vector<IPluginV2*> plugins;
        std::map<std::string,ITensor*> weights;std::map<std::string,std::vector<unsigned char>> storage;
        std::map<std::string,Weights> refit;std::map<std::string,std::string> externalFiles;
        const std::string prefix="model.language_model.layers.0.";
        auto* input=network->addInput("x",DataType::kFLOAT,Ops::dims({batch,t,h}));
        auto* history=network->addInput("history",DataType::kFLOAT,Ops::dims({batch,channels,kernel}));
        auto* state=network->addInput("state",DataType::kFLOAT,Ops::dims({batch,vh,kd,vd}));
        auto* active=network->addInput("active",DataType::kFLOAT,Ops::dims({batch,t}));
        for(auto suffix:fixture.at("large_weight_order")){
            const auto name=prefix+suffix.get<std::string>();json meta;
            for(auto item:fixture.at("tensors"))if(item.at("name")==name)meta=item;
            need(!meta.is_null()&&meta.at("dtype")=="BF16","Large BF16 weight missing");
            auto shape=meta.at("shape").get<std::vector<int>>();auto* tensor=network->addInput(name.c_str(),DataType::kBF16,Ops::dims(shape));
            need(tensor,"Weight input creation failed");weights[name]=tensor;externalFiles[name]=meta.at("file").get<std::string>();
        }
        auto lookup=[&](const std::string& name)->ITensor*{
            auto found=weights.find(name);if(found!=weights.end())return found->second;
            json meta;for(auto item:fixture.at("tensors"))if(item.at("name")==name)meta=item;
            need(!meta.is_null()&&meta.at("dtype")=="F32","Small weight metadata missing");
            auto raw=bytes(meta.at("file").get<std::string>());DataType type=DataType::kFLOAT;
            if(meta.at("checkpoint_dtype")=="BF16"){
                std::vector<unsigned char> narrowed(raw.size()/2);
                for(size_t i=0;i<raw.size()/4;++i){uint32_t word;std::memcpy(&word,raw.data()+4*i,4);need((word&65535)==0,"Widened weight not exactlyBF16");uint16_t small=uint16_t(word>>16);std::memcpy(narrowed.data()+2*i,&small,2);}
                raw=std::move(narrowed);type=DataType::kBF16;
            }
            storage[name]=std::move(raw);auto& value=storage[name];const size_t item=type==DataType::kBF16?2:4;
            Weights w{type,value.data(),int64_t(value.size()/item)};
            auto* layer=network->addConstant(Ops::dims(meta.at("shape").get<std::vector<int>>()),w);need(layer,"Small constant failed");
            need(network->setWeightsName(w,name.c_str())&&network->markWeightsRefittable(name.c_str()),"Weight refit metadata failed");
            layer->setName(name.c_str());auto* tensor=layer->getOutput(0);tensor->setName(name.c_str());weights[name]=tensor;refit[name]=w;return tensor;
        };
        Ops ops(*network,constants,lookup,plugins);ops.SetProjectionWorkspace(0);
        struct Observation{std::string name,input,output;int k,n,rows;};std::vector<Observation> observations;
        auto project=[&](ITensor* x,ITensor* w,const std::string& label)->ITensor*{
            auto xs=Ops::shape(x),ws=Ops::shape(w);need(ws.size()==2&&xs.back()==ws[1]&&w->getType()==DataType::kBF16,"Projection contract mismatch");
            need(Garnet::EnsureBF16ProjectionPluginRegistered(),"Projection plugin registration failed");
            auto* plugin=new Garnet::BF16ProjectionPlugin(false);plugins.push_back(plugin);
            int rows=1;for(size_t i=0;i+1<xs.size();++i)rows*=xs[i];
            ITensor* args[]={ops.reshape(ops.cast(x,DataType::kBF16),{rows,ws[1]}),w};
            auto* layer=network->addPluginV2(args,2,*plugin);need(layer,"Projection plugin construction failed");auto* y=layer->getOutput(0);
            const std::string inName="observe_"+label+"_input",outName="observe_"+label+"_output";
            args[0]->setName(inName.c_str());y->setName(outName.c_str());
            observations.push_back({label,inName,outName,ws[1],ws[0],rows});
            xs.back()=ws[0];return ops.cast(ops.reshape(y,xs));
        };
        auto w=[&](const char* suffix){return lookup(prefix+suffix);};
        auto* masked=ops.select(ops.reshape(active,{batch,t,1}),input,ops.scalar(0,3));
        auto* normalized=ops.norm(masked,prefix+"input_layernorm.weight",c.at("rms_norm_eps"));
        auto* qkv=project(normalized,w("linear_attn.in_proj_qkv.weight"),"qkv");
        auto* z=project(normalized,w("linear_attn.in_proj_z.weight"),"z");
        auto* beta=project(normalized,w("linear_attn.in_proj_b.weight"),"b");
        auto* a=project(normalized,w("linear_attn.in_proj_a.weight"),"a");
        need(Garnet::EnsureQwen35DeltaPluginRegistered(),"Delta plugin registration failed");
        auto* delta=new Garnet::Qwen35DeltaPlugin({kh,vh,kd,vd,kernel,float(c.at("rms_norm_eps"))},true);plugins.push_back(delta);
        ITensor* deltaInputs[]={ops.cast(qkv),ops.cast(z),ops.cast(beta),ops.cast(a),ops.cast(history),ops.cast(state),ops.cast(active),
            ops.cast(w("linear_attn.conv1d.weight")),ops.cast(w("linear_attn.dt_bias")),ops.cast(w("linear_attn.A_log")),ops.cast(w("linear_attn.norm.weight"))};
        auto* deltaLayer=network->addPluginV2(deltaInputs,11,*delta);need(deltaLayer,"Delta plugin construction failed");
        auto* attention=project(deltaLayer->getOutput(0),w("linear_attn.out_proj.weight"),"out");
        attention=ops.select(ops.reshape(active,{batch,t,1}),attention,ops.scalar(0,3));
        auto* residual=ops.add(masked,attention);
        auto* mlpNorm=ops.norm(residual,prefix+"post_attention_layernorm.weight",c.at("rms_norm_eps"));
        // Preserve the production MLP expression, observing only BF16 plugin IO.
        auto* down=project(ops.mul(ops.silu(project(mlpNorm,w("mlp.gate_proj.weight"),"gate")),
            project(mlpNorm,w("mlp.up_proj.weight"),"up")),w("mlp.down_proj.weight"),"down");
        auto* hidden=ops.add(residual,down);
        hidden->setName("hidden");network->markOutput(*hidden);
        deltaLayer->getOutput(1)->setName("next_history");network->markOutput(*deltaLayer->getOutput(1));
        deltaLayer->getOutput(2)->setName("next_state");network->markOutput(*deltaLayer->getOutput(2));
        for(const auto& item:observations){
            for(int i=0;i<network->getNbLayers();++i){auto* layer=network->getLayer(i);for(int j=0;j<layer->getNbOutputs();++j){auto* value=layer->getOutput(j);if(value&&(item.input==value->getName()||item.output==value->getName()))network->markOutput(*value);}}
        }
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));need(bool(plan),"TensorRT component build failed");
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(),plan->size()));need(bool(engine),"Component deserialize failed");
        {std::unique_ptr<IRefitter> refitter(createInferRefitter(*engine,logger));need(bool(refitter),"Refitter unavailable");for(auto& pair:refit)need(refitter->setNamedWeights(pair.first.c_str(),pair.second),"Refit binding failed");need(refitter->refitCudaEngine(),"Component refit failed");}
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());need(bool(context),"Execution context unavailable");
        std::map<std::string,std::unique_ptr<Buffer>> buffers;cudaStream_t stream;cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(int i=0;i<engine->getNbIOTensors();++i){std::string name=engine->getIOTensorName(i);auto type=engine->getTensorDataType(name.c_str());need(type==DataType::kFLOAT||type==DataType::kBF16,"Unexpected IO dtype");size_t n=volume(engine->getTensorShape(name.c_str()))*(type==DataType::kBF16?2:4);buffers[name]=std::make_unique<Buffer>(n);need(context->setTensorAddress(name.c_str(),buffers[name]->p),"IO binding failed");}
        for(const auto& pair:externalFiles){storage[pair.first]=bytes(pair.second);auto& data=storage[pair.first];need(data.size()==buffers.at(pair.first)->bytes,"Weight storage size mismatch");cu(cudaMemcpyAsync(buffers.at(pair.first)->p,data.data(),data.size(),cudaMemcpyHostToDevice,stream));}
        auto inputValues=floats(test.at("files").at("input").at("file"));auto expected=floats(test.at("files").at("expected_output").at("file"));auto expectedState=floats(test.at("files").at("expected_state").at("file"));
        need(inputValues.size()==size_t(batch)*tokens*h&&expected.size()==inputValues.size()&&expectedState.size()==size_t(batch)*(historyWidth+stateWidth),"Reference file size mismatch");
        std::vector<float> x(size_t(batch)*t*h,0),flags(size_t(batch)*t,0);
        for(int row=0;row<batch;++row){std::copy_n(inputValues.data()+size_t(row)*tokens*h,size_t(t)*h,x.data()+size_t(row)*t*h);std::fill_n(flags.data()+row*t,t,1.f);}
        cu(cudaMemcpyAsync(buffers.at("x")->p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));cu(cudaMemcpyAsync(buffers.at("active")->p,flags.data(),flags.size()*4,cudaMemcpyHostToDevice,stream));
        cu(cudaMemsetAsync(buffers.at("history")->p,0,buffers.at("history")->bytes,stream));cu(cudaMemsetAsync(buffers.at("state")->p,0,buffers.at("state")->bytes,stream));cu(cudaStreamSynchronize(stream));
        need(context->enqueueV3(stream),"Component warmup failed");cu(cudaStreamSynchronize(stream));
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));bool enqueued=context->enqueueV3(stream);auto captured=cudaStreamEndCapture(stream,&graph);need(enqueued,"Component capture enqueue failed");cu(captured);cu(cudaGraphInstantiate(&executable,graph,0));
        auto stagePath=output;stagePath.replace_extension(".stages.u16");exclusive(stagePath);std::ofstream stageFile(stagePath,std::ios::binary);size_t stageWords=0;
        std::vector<float> actual(size_t(batch)*t*h),nextHistory(size_t(batch)*historyWidth),nextState(size_t(batch)*stateWidth);
        result["stage_file"]=std::filesystem::absolute(stagePath).string();result["cuda_graph_capture"]=true;result["references_exact"]=true;
        for(int repeat=0;repeat<2;++repeat){
            auto actualPath=output;actualPath.replace_extension(".actual-r"+std::to_string(repeat)+".f32");exclusive(actualPath);std::ofstream actualFile(actualPath,std::ios::binary);
            result["actual_output_files"][std::to_string(repeat)]=std::filesystem::absolute(actualPath).string();
            cu(cudaMemsetAsync(buffers.at("history")->p,0,buffers.at("history")->bytes,stream));cu(cudaMemsetAsync(buffers.at("state")->p,0,buffers.at("state")->bytes,stream));
            for(int start=0;start<tokens;start+=t){int count=std::min(t,tokens-start);std::fill(x.begin(),x.end(),0);std::fill(flags.begin(),flags.end(),0);
                for(int row=0;row<batch;++row){std::copy_n(inputValues.data()+(size_t(row)*tokens+start)*h,size_t(count)*h,x.data()+size_t(row)*t*h);std::fill_n(flags.data()+row*t,count,1.f);}
                cu(cudaMemcpyAsync(buffers.at("x")->p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));cu(cudaMemcpyAsync(buffers.at("active")->p,flags.data(),flags.size()*4,cudaMemcpyHostToDevice,stream));cu(cudaGraphLaunch(executable,stream));
                cu(cudaMemcpyAsync(actual.data(),buffers.at("hidden")->p,actual.size()*4,cudaMemcpyDeviceToHost,stream));cu(cudaMemcpyAsync(nextHistory.data(),buffers.at("next_history")->p,nextHistory.size()*4,cudaMemcpyDeviceToHost,stream));cu(cudaMemcpyAsync(nextState.data(),buffers.at("next_state")->p,nextState.size()*4,cudaMemcpyDeviceToHost,stream));cu(cudaStreamSynchronize(stream));
                size_t mismatches=0;
                for(int row=0;row<batch;++row){auto* value=actual.data()+size_t(row)*t*h;mismatches+=different(value,expected.data()+(size_t(row)*tokens+start)*h,size_t(count)*h);actualFile.seekp((size_t(row)*tokens+start)*h*4);actualFile.write((const char*)value,size_t(count)*h*4);}
                result["chunks"].push_back({{"repeat",repeat},{"start",start},{"count",count},{"output_word_mismatches",mismatches}});
                for(const auto& item:observations)for(int row=0;row<batch;++row)for(int part=0;part<2;++part){const std::string name=part?item.output:item.input;const int width=part?item.n:item.k;std::vector<uint16_t> words(width);
                    cu(cudaMemcpyAsync(words.data(),(const char*)buffers.at(name)->p+size_t(row)*t*width*2,size_t(width)*2,cudaMemcpyDeviceToHost,stream));cu(cudaStreamSynchronize(stream));
                    for(auto small:words){uint32_t bits=uint32_t(small)<<16;float v;std::memcpy(&v,&bits,4);need(std::isfinite(v),"Nonfinite observed BF16 projection IO");}
                    stageFile.write((const char*)words.data(),words.size()*2);result["observations"].push_back({{"repeat",repeat},{"chunk_start",start},{"row",row},{"projection",item.name},{"part",part?"output":"input"},{"word_offset",stageWords},{"words",width},{"gemm_rows",item.rows},{"inner",item.k},{"columns",item.n}});stageWords+=words.size();
                }
                actualFile.flush();stageFile.flush();save();need(mismatches==0,"Component output differs from exact actual layer0 reference; reject attribution");
                cu(cudaMemcpyAsync(buffers.at("history")->p,buffers.at("next_history")->p,nextHistory.size()*4,cudaMemcpyDeviceToDevice,stream));cu(cudaMemcpyAsync(buffers.at("state")->p,buffers.at("next_state")->p,nextState.size()*4,cudaMemcpyDeviceToDevice,stream));
            }
            size_t stateMismatch=0;for(int row=0;row<batch;++row){auto* reference=expectedState.data()+size_t(row)*(historyWidth+stateWidth);stateMismatch+=different(nextHistory.data()+size_t(row)*historyWidth,reference,historyWidth);stateMismatch+=different(nextState.data()+size_t(row)*stateWidth,reference+historyWidth,stateWidth);}
            auto statePath=output;statePath.replace_extension(".state-r"+std::to_string(repeat)+".f32");exclusive(statePath);
            {std::ofstream f(statePath,std::ios::binary);for(int row=0;row<batch;++row){f.write((const char*)(nextHistory.data()+size_t(row)*historyWidth),historyWidth*4);f.write((const char*)(nextState.data()+size_t(row)*stateWidth),stateWidth*4);}need(bool(f),"Actual state write failed");}
            result["actual_state_files"][std::to_string(repeat)]=std::filesystem::absolute(statePath).string();
            result["repeat_state_mismatches"][std::to_string(repeat)]=stateMismatch;save();need(stateMismatch==0,"Component final states differ from original layer0; reject attribution");
        }
        stageFile.close();cu(cudaStreamSynchronize(stream));cu(cudaGraphExecDestroy(executable));cu(cudaGraphDestroy(graph));context.reset();engine.reset();runtime.reset();network.reset();config.reset();builder.reset();for(auto* p:plugins)p->destroy();buffers.clear();cu(cudaStreamDestroy(stream));
        result["stage_words"]=stageWords;result["whole_program_cuda"]=Garnet::Qwen35WholeProgramCUDA();need(!Garnet::Qwen35WholeProgramCUDA(),"Component must use original separate CUDA compilation");result["admitted_exact_original_reference"]=true;result["scope"]="Standalone diagnostic component only. All85 native layer0 outputs and complete final conv/recurrent states must be bit-exact on both reset repeats before stage attribution. No serving, independent quality, timing, memory saving or2x claim.";save();std::cout<<"LAYER0_STAGE_COMPONENT_ADMITTED "<<batch<<'\n';return 0;
    }catch(const std::exception& e){result["failure"]=e.what();save();std::cerr<<"LAYER0_STAGE_COMPONENT_FAILED "<<e.what()<<'\n';return 1;}
}
