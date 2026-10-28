// SPDX-License-Identifier: Apache-2.0
// Unwired documented TensorRT V3 alias contract probe. No model arithmetic.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../src/backends/tensorrt/qwen35_packed_store_plugin.h"
using namespace nvinfer1;
using json=nlohmann::json;
static void need(bool value,const char* text){if(!value)throw std::runtime_error(text);}
static void cu(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
struct Logger final:ILogger{void log(Severity level,const char* text) noexcept override{if(level<=Severity::kWARNING)std::cerr<<text<<'\n';}};
static Logger logger;
static bool overlap(const void* a,size_t an,const void* b,size_t bn){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<an:x-y<bn;
}
// root width, hidden width, part offset, part width, write hidden, alias root.
using Config=std::array<int32_t,6>;
static bool valid(const Config& c){return c[0]>0&&c[1]>0&&c[1]<=c[0]&&c[2]>=c[1]&&c[3]>0&&c[2]<=c[0]&&c[3]<=c[0]-c[2]&&(c[4]==0||c[4]==1)&&(c[5]==0||c[5]==1);}
class Store final:public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuildV2,public IPluginV3OneRuntime{
    Config c_;PluginField field_{};PluginFieldCollection fields_{};
    bool descriptors(const PluginTensorDesc* in,int ni,const PluginTensorDesc* out,int no)const noexcept{
        if(!in||!out||ni!=3||no!=2)return false;
        for(int i=0;i<ni;++i)if(in[i].dims.nbDims!=2||in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return false;
        int64_t b=in[0].dims.d[0];if(b<1||b>8||in[0].dims.d[1]!=c_[0]||in[1].dims.d[0]!=b||in[2].dims.d[0]!=b||in[1].dims.d[1]!=c_[1]||in[2].dims.d[1]!=c_[3])return false;
        for(int i=0;i<no;++i)if(out[i].dims.nbDims!=2||out[i].type!=DataType::kFLOAT||out[i].format!=TensorFormat::kLINEAR||out[i].dims.d[0]!=b||out[i].dims.d[1]!=(i==0?c_[0]:c_[1]))return false;
        return true;
    }
public:
    explicit Store(Config c):c_(c){need(valid(c),"invalid store config");field_={"config",c_.data(),PluginFieldType::kINT32,6};fields_={1,&field_};}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override{
        if(type==PluginCapabilityType::kCORE)return static_cast<IPluginV3OneCore*>(this);
        if(type==PluginCapabilityType::kBUILD)return static_cast<IPluginV3OneBuildV2*>(this);
        if(type==PluginCapabilityType::kRUNTIME)return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override{try{return new Store(c_);}catch(...){return nullptr;}}
    const char* getPluginName()const noexcept override{return "GarnetUnwiredPackedStoreAlias";}
    const char* getPluginVersion()const noexcept override{return "1";}
    const char* getPluginNamespace()const noexcept override{return "";}
    int32_t getNbOutputs()const noexcept override{return 2;}
    int32_t getAliasedInput(int32_t output) noexcept override{return output==0&&c_[5]?0:-1;}
    int32_t configurePlugin(const DynamicPluginTensorDesc* in,int32_t ni,const DynamicPluginTensorDesc* out,int32_t no) noexcept override{
        if(!in||!out||ni!=3||no!=2)return 1;
        PluginTensorDesc a[3]={in[0].desc,in[1].desc,in[2].desc},b[2]={out[0].desc,out[1].desc};return descriptors(a,ni,b,no)?0:1;
    }
    int32_t getOutputDataTypes(DataType* out,int32_t no,const DataType* in,int32_t ni)const noexcept override{
        if(!out||!in||no!=2||ni!=3)return 1;
        for(int i=0;i<ni;++i)if(in[i]!=DataType::kFLOAT)return 1;
        out[0]=out[1]=DataType::kFLOAT;return 0;
    }
    int32_t getOutputShapes(const DimsExprs* in,int32_t ni,const DimsExprs*,int32_t ns,DimsExprs* out,int32_t no,IExprBuilder&) noexcept override{
        if(!in||!out||ni!=3||no!=2||ns||in[0].nbDims!=2||in[1].nbDims!=2||in[2].nbDims!=2)return 1;
        out[0]=in[0];out[1]=in[1];return 0;
    }
    bool supportsFormatCombination(int32_t pos,const DynamicPluginTensorDesc* io,int32_t ni,int32_t no) noexcept override{
        return io&&ni==3&&no==2&&pos>=0&&pos<5&&io[pos].desc.type==DataType::kFLOAT&&io[pos].desc.format==TensorFormat::kLINEAR;
    }
    int32_t onShapeChange(const PluginTensorDesc* in,int32_t ni,const PluginTensorDesc* out,int32_t no) noexcept override{return descriptors(in,ni,out,no)?0:1;}
    int32_t enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void*const* inputs,void*const* outputs,void*,cudaStream_t stream) noexcept override{
        if(!descriptors(in,3,out,2)||!inputs||!outputs||!inputs[0]||!inputs[1]||!inputs[2]||!outputs[0]||!outputs[1])return 1;
        size_t b=size_t(in[0].dims.d[0]),root=b*size_t(c_[0])*4,hidden=b*size_t(c_[1])*4,part=b*size_t(c_[3])*4;
        if(bool(c_[5])!=(inputs[0]==outputs[0]))return 1;
        if(!c_[5]&&overlap(outputs[0],root,inputs[0],root))return 1;
        if(overlap(outputs[0],root,inputs[1],hidden)||overlap(outputs[0],root,inputs[2],part)||overlap(outputs[0],root,outputs[1],hidden))return 1;
        if(overlap(outputs[1],hidden,inputs[0],root)||overlap(outputs[1],hidden,inputs[2],part))return 1;
        if(outputs[1]!=inputs[1]&&overlap(outputs[1],hidden,inputs[1],hidden))return 1;
        if(!c_[5]&&cudaMemcpyAsync(outputs[0],inputs[0],root,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        auto* destination=static_cast<char*>(outputs[0]);
        if(cudaMemcpy2DAsync(destination+size_t(c_[2])*4,size_t(c_[0])*4,inputs[2],size_t(c_[3])*4,size_t(c_[3])*4,b,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        if(outputs[1]!=inputs[1]&&cudaMemcpyAsync(outputs[1],inputs[1],hidden,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        if(c_[4]&&cudaMemcpy2DAsync(destination,size_t(c_[0])*4,inputs[1],size_t(c_[1])*4,size_t(c_[1])*4,b,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override{return clone();}
    const PluginFieldCollection* getFieldsToSerialize() noexcept override{return &fields_;}
};
class Creator final:public IPluginCreatorV3One{
    PluginFieldCollection fields_{};
public:
    IPluginV3* createPlugin(const char*,const PluginFieldCollection* fields,TensorRTPhase) noexcept override{
        if(!fields||fields->nbFields!=1||!fields->fields)return nullptr;
        const auto& f=fields->fields[0];if(!f.name||std::strcmp(f.name,"config")||!f.data||f.type!=PluginFieldType::kINT32||f.length!=6)return nullptr;
        Config c{};std::memcpy(c.data(),f.data,sizeof(c));if(!valid(c))return nullptr;
        try{return new Store(c);}catch(...){return nullptr;}
    }
    const PluginFieldCollection* getFieldNames() noexcept override{return &fields_;}
    const char* getPluginName()const noexcept override{return "GarnetUnwiredPackedStoreAlias";}
    const char* getPluginVersion()const noexcept override{return "1";}
    const char* getPluginNamespace()const noexcept override{return "";}
};
static Creator creator;
static bool productionPlugin=false;
static std::vector<int32_t> productionConfig(const Config& c){return {c[0],c[1],c[4],c[5],1,c[2],c[3]};}
struct Buffer{
    void* raw=nullptr;void* data=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){cu(cudaMalloc(&raw,n+1536));data=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw)+1023)&~uintptr_t(511));cu(cudaMemset(raw,0xa5,n+1536));}
    ~Buffer(){if(raw)cudaFree(raw);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<uint32_t>& v,cudaStream_t stream){need(v.size()*4==bytes,"upload bytes");cu(cudaMemcpyAsync(data,v.data(),bytes,cudaMemcpyHostToDevice,stream));}
    std::vector<uint32_t> read()const{std::vector<uint32_t> v(bytes/4);cu(cudaMemcpy(v.data(),data,bytes,cudaMemcpyDeviceToHost));return v;}
    void guards()const{std::array<unsigned char,512> a{},b{};cu(cudaMemcpy(a.data(),static_cast<char*>(data)-512,512,cudaMemcpyDeviceToHost));cu(cudaMemcpy(b.data(),static_cast<char*>(data)+bytes,512,cudaMemcpyDeviceToHost));for(auto x:a)need(x==0xa5,"prefix guard");for(auto x:b)need(x==0xa5,"suffix guard");}
};
struct Stream{cudaStream_t value=nullptr;Stream(){cu(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
static int negativeContracts(){
    int checks=0;
    const Config baseline{16,4,4,6,1,1};
    auto* checkedCreator=productionPlugin?dynamic_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("GarnetQwen35PackedStore","1","")):&creator;
    need(checkedCreator,"negative creator");
    for(auto bad:std::vector<Config>{{0,4,4,6,1,1},{16,17,4,6,1,1},{16,4,3,6,1,1},{16,4,4,13,1,1},{16,4,4,6,2,1},{16,4,4,6,1,2}}){
        auto c=productionConfig(bad);PluginField f{"config",productionPlugin?c.data():bad.data(),PluginFieldType::kINT32,productionPlugin?7:6};PluginFieldCollection fc{1,&f};
        need(checkedCreator->createPlugin("invalid",&fc,TensorRTPhase::kRUNTIME)==nullptr,"invalid config accepted");++checks;
    }
    PluginField wrong{"config",baseline.data(),PluginFieldType::kINT32,5};PluginFieldCollection fc{1,&wrong};
    need(checkedCreator->createPlugin("invalid",&fc,TensorRTPhase::kRUNTIME)==nullptr,"short serialized config accepted");++checks;
    Buffer root(64),hidden(16),part(24),otherRoot(64),otherHidden(16);Stream stream;
    auto beforeRoot=root.read(),beforeHidden=hidden.read(),beforePart=part.read(),beforeOther=otherRoot.read(),beforeOutput=otherHidden.read();
    PluginTensorDesc in[3]{},out[2]{};
    for(int i=0;i<3;++i){in[i].type=DataType::kFLOAT;in[i].format=TensorFormat::kLINEAR;in[i].dims=Dims2(1,i==0?16:i==1?4:6);}
    out[0]=in[0];out[1]=in[1];
    const void* inputs[]={root.data,hidden.data,part.data};void* outputs[]={root.data,otherHidden.data};
    Store alias(baseline),copy(Config{16,4,4,6,1,0});
    std::unique_ptr<IPluginV3> actualAlias,actualCopy;
    if(productionPlugin){actualAlias.reset(Garnet::CreateQwen35PackedStorePlugin(productionConfig(baseline)));actualCopy.reset(Garnet::CreateQwen35PackedStorePlugin(productionConfig(Config{16,4,4,6,1,0})));}
    auto& aliasRuntime=productionPlugin?*static_cast<IPluginV3OneRuntime*>(actualAlias->getCapabilityInterface(PluginCapabilityType::kRUNTIME)):static_cast<IPluginV3OneRuntime&>(alias);
    auto& copyRuntime=productionPlugin?*static_cast<IPluginV3OneRuntime*>(actualCopy->getCapabilityInterface(PluginCapabilityType::kRUNTIME)):static_cast<IPluginV3OneRuntime&>(copy);
    auto reject=[&](IPluginV3OneRuntime& store){need(store.enqueue(in,out,inputs,outputs,nullptr,stream.value)!=0,"invalid pointer/descriptor contract accepted");++checks;};
    outputs[0]=otherRoot.data;reject(aliasRuntime); // Missing required alias.
    outputs[0]=root.data;reject(copyRuntime); // Undeclared exact overlap.
    outputs[0]=static_cast<char*>(root.data)+4;reject(copyRuntime); // Undeclared partial overlap.
    outputs[0]=root.data;outputs[1]=hidden.data;inputs[1]=root.data;reject(aliasRuntime); // Writable root overlaps readonly hidden.
    inputs[1]=hidden.data;inputs[2]=root.data;outputs[1]=otherHidden.data;reject(aliasRuntime); // Writable root overlaps part.
    inputs[2]=part.data;outputs[1]=root.data;reject(aliasRuntime); // Outputs overlap.
    outputs[1]=static_cast<char*>(hidden.data)+4;reject(aliasRuntime); // Partial hidden forwarding overlap.
    outputs[1]=otherHidden.data;out[0].dims.d[1]=15;reject(aliasRuntime);out[0]=in[0];
    inputs[2]=nullptr;reject(aliasRuntime);inputs[2]=part.data;
    cu(cudaStreamSynchronize(stream.value));
    need(root.read()==beforeRoot&&hidden.read()==beforeHidden&&part.read()==beforePart&&otherRoot.read()==beforeOther&&otherHidden.read()==beforeOutput,"rejected call changed buffer bytes");
    root.guards();hidden.guards();part.guards();otherRoot.guards();otherHidden.guards();
    return checks;
}
struct Capture{
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~Capture(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    void build(IExecutionContext& context,cudaStream_t stream){cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));bool ok=context.enqueueV3(stream);auto status=cudaStreamEndCapture(stream,&graph);cu(status);need(ok&&graph,"capture enqueue");cu(cudaGraphInstantiateWithFlags(&executable,graph,0));}
};
static std::vector<uint32_t> pattern(size_t count,int seed){
    static constexpr uint32_t special[]={0,0x80000000,0x7fc12345,0xffc54321,0x7f800000,0xff800000,0x3f800000,0x00000001};
    std::vector<uint32_t> v(count);for(size_t i=0;i<count;++i)v[i]=i%11<8?special[(i+seed)%8]:uint32_t(i*2654435761ULL+seed*104729ULL);return v;
}
struct Bank{
    int batch,rootWidth,hiddenWidth,partWidth;Buffer root,hidden,a,b,hiddenOut;std::unique_ptr<Buffer> rootOut;
    std::vector<uint32_t> initial,h,pa,pb,expected;
    Bank(int n,int rw,int hw,int pw,bool alias):batch(n),rootWidth(rw),hiddenWidth(hw),partWidth(pw),root(size_t(n)*rw*4),hidden(size_t(n)*hw*4),a(size_t(n)*pw*4),b(size_t(n)*pw*4),hiddenOut(size_t(n)*hw*4){if(!alias)rootOut=std::make_unique<Buffer>(root.bytes);}
    void* output(){return rootOut?rootOut->data:root.data;}
    void bind(IExecutionContext& c,bool internalRoot){need((internalRoot||c.setTensorAddress("destination",root.data))&&c.setTensorAddress("hidden",hidden.data)&&c.setTensorAddress("part_a",a.data)&&c.setTensorAddress("part_b",b.data)&&c.setTensorAddress("root_output",output())&&c.setTensorAddress("hidden_output",hiddenOut.data),"bindings");}
    void reset(int seed,cudaStream_t stream){
        initial=pattern(size_t(batch)*rootWidth,seed);h=pattern(size_t(batch)*hiddenWidth,seed+1);pa=pattern(size_t(batch)*partWidth,seed+2);pb=pattern(size_t(batch)*partWidth,seed+3);expected=initial;
        for(int row=0;row<batch;++row){auto begin=expected.begin()+size_t(row)*rootWidth;std::copy_n(h.begin()+size_t(row)*hiddenWidth,hiddenWidth,begin);std::copy_n(pa.begin()+size_t(row)*partWidth,partWidth,begin+hiddenWidth);std::copy_n(pb.begin()+size_t(row)*partWidth,partWidth,begin+hiddenWidth+partWidth);}
        root.upload(initial,stream);hidden.upload(h,stream);a.upload(pa,stream);b.upload(pb,stream);cu(cudaStreamSynchronize(stream));
    }
    void check(){
        auto result=rootOut?rootOut->read():root.read();need(result==expected,"full root bits differ");need(hiddenOut.read()==h,"forwarded hidden bits differ");need(hidden.read()==h&&a.read()==pa&&b.read()==pb,"readonly input bytes changed");if(rootOut)need(root.read()==initial,"non-alias input root changed");
        root.guards();hidden.guards();a.guards();b.guards();hiddenOut.guards();if(rootOut)rootOut->guards();
    }
};
static std::vector<uint8_t> plan(int batch,int rootWidth,int hiddenWidth,int partWidth,bool alias,bool internalRoot,int partCount){
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));need(bool(builder),"builder");
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());need(bool(config),"config");config->clearFlag(BuilderFlag::kTF32);config->setMaxAuxStreams(0);config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);config->setPreviewFeature(PreviewFeature::kALIASED_PLUGIN_IO_10_03,true);
    std::vector<std::unique_ptr<IPluginV3>> plugins;
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(productionPlugin?(1U<<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)):0));need(bool(network),"network");
    ITensor* root=nullptr;const float fillAlpha=0,fillBeta[2]={0,0};
    if(internalRoot){
        auto* fill=network->addFill(Dims2(batch,rootWidth),FillOperation::kLINSPACE);need(fill,"root fill");
        Dims deltaDims{};deltaDims.nbDims=1;deltaDims.d[0]=2;
        auto* alpha=network->addConstant(Dims{},Weights{DataType::kFLOAT,&fillAlpha,1});auto* beta=network->addConstant(deltaDims,Weights{DataType::kFLOAT,fillBeta,2});need(alpha&&beta,"ND fill parameters");
        fill->setInput(1,*alpha->getOutput(0));fill->setInput(2,*beta->getOutput(0));root=fill->getOutput(0);
    }
    else root=network->addInput("destination",DataType::kFLOAT,Dims2(batch,rootWidth));
    auto* hidden=network->addInput("hidden",DataType::kFLOAT,Dims2(batch,hiddenWidth));auto* a=network->addInput("part_a",DataType::kFLOAT,Dims2(batch,partWidth));auto* b=network->addInput("part_b",DataType::kFLOAT,Dims2(batch,partWidth));need(root&&hidden&&a&&b,"inputs");
    for(int i=0;i<2;++i){
        std::vector<ITensor*> inputs{root,hidden};
        if(productionPlugin){
            std::vector<int32_t> c{rootWidth,hiddenWidth,i,int(alias),partCount};int offset=0;
            for(int part=0;part<partCount;++part){int width=part+1==partCount?partWidth-offset:partWidth/partCount;auto* slice=network->addSlice(*(i?b:a),Dims2(0,offset),Dims2(batch,width),Dims2(1,1));need(slice,"production part slice");inputs.push_back(slice->getOutput(0));c.push_back(hiddenWidth+i*partWidth+offset);c.push_back(width);offset+=width;}
            plugins.emplace_back(Garnet::CreateQwen35PackedStorePlugin(c));
        }else{plugins.push_back(std::make_unique<Store>(Config{rootWidth,hiddenWidth,hiddenWidth+i*partWidth,partWidth,i,int(alias)}));inputs.push_back(i?b:a);}
        auto* layer=network->addPluginV3(inputs.data(),int(inputs.size()),nullptr,0,*plugins.back());need(layer,"store layer");root=layer->getOutput(0);hidden=layer->getOutput(1);
    }
    root->setName("root_output");hidden->setName("hidden_output");network->markOutput(*root);network->markOutput(*hidden);
    std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network,*config));need(bool(serialized),"serialized alias plan rejected");
    auto* bytes=static_cast<const uint8_t*>(serialized->data());return {bytes,bytes+serialized->size()};
}
static json exercise(int batch,int chunk,bool alias,bool internalRoot,int partCount){
    const int hiddenWidth=chunk*31,partWidth=131089,rootWidth=hiddenWidth+2*partWidth;
    auto bytes=plan(batch,rootWidth,hiddenWidth,partWidth,alias,internalRoot,partCount);int checks=0;int64_t scratchBytes=0;
    for(int rebuild=0;rebuild<2;++rebuild){
        Bank first(batch,rootWidth,hiddenWidth,partWidth,alias&&!internalRoot),second(batch,rootWidth,hiddenWidth,partWidth,alias&&!internalRoot);
        {
            std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));need(bool(runtime),"runtime");
            std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));need(bool(engine),"engine deserialize");
            auto* declared=engine->getAliasedInputTensor("root_output");need(alias&&!internalRoot?(declared&&std::string(declared)=="destination"):declared==nullptr,"network root alias descriptor did not match declared contract");
            need(engine->getNbIOTensors()==(internalRoot?5:6),"unexpected network IO count");
            need(engine->getAliasedInputTensor("hidden_output")==nullptr,"hidden copy unexpectedly aliases network input");scratchBytes=engine->getDeviceMemorySizeV2();Buffer scratch(size_t(std::max<int64_t>(256,scratchBytes)));Stream stream;
            std::unique_ptr<IExecutionContext> context(engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));need(bool(context),"context");context->setDeviceMemoryV2(scratch.data,scratch.bytes);
            for(int bank=0;bank<2;++bank){
                Bank& target=bank?second:first;target.bind(*context,internalRoot);target.reset(100+rebuild*20+bank*5,stream.value);need(context->enqueueV3(stream.value),"plain enqueue");cu(cudaStreamSynchronize(stream.value));target.check();++checks;
                {
                    Capture captured;captured.build(*context,stream.value);cu(cudaStreamSynchronize(stream.value));
                    for(int repeat=0;repeat<3;++repeat){target.reset(101+rebuild*20+bank*5+repeat,stream.value);cu(cudaGraphLaunch(captured.executable,stream.value));cu(cudaStreamSynchronize(stream.value));target.check();++checks;}
                }
                if(bank){first.check();++checks;}
            }
            scratch.guards();context.reset();
        }
        first.check();second.check();checks+=2; // Caller output survives context/engine/runtime destruction.
    }
    return {{"batch",batch},{"chunk",chunk},{"alias_root",alias},{"production_plugin",productionPlugin},{"parts_per_store",partCount},{"internal_fill_root",internalRoot},{"checks",checks},{"context_upper_bound_bytes_not_model_savings",scratchBytes},{"serialized_bytes",bytes.size()},{"all_bits_and_guards_passed",true},{"network_root_alias_confirmed",alias&&!internalRoot}};
}
int main(int argc,char** argv){
    if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="internal-fill"&&std::string(argv[2])!="production-internal-fill")){std::cerr<<"Output JSON path and optional internal-fill/production-internal-fill required\n";return 2;}
    const bool internalRoot=argc==3;
    productionPlugin=argc==3&&std::string(argv[2])=="production-internal-fill";
    json output{{"scope","Unwired copy-only documented V3 alias/serialization/capture/ownership contract; not a model memory or speed result."},{"cases",json::array()},{"all_passed",false}};
    try{
        cu(cudaSetDevice(0));need(getPluginRegistry()->registerCreator(creator,""),"creator registration");
        if(productionPlugin)need(Garnet::EnsureQwen35PackedStorePluginRegistered(),"production creator registration");
        output["negative_contract_checks"]=negativeContracts();
        for(int partCount:productionPlugin?std::vector<int>{1,2,6}:std::vector<int>{1})for(int batch:{1,2,4,8})for(int chunk:{4,64})for(bool alias:{false,true}){
            output["cases"].push_back(exercise(batch,chunk,alias,internalRoot,partCount));std::ofstream(argv[1])<<output.dump(2);std::cout<<"PACKED_STORE_ALIAS_CASE_PASS "<<batch<<' '<<chunk<<' '<<alias<<' '<<internalRoot<<' '<<partCount<<std::endl;
        }
        output["all_passed"]=true;output["unfiltered_sanitizer_status_is_separate"]=true;std::ofstream(argv[1])<<output.dump(2);std::cout<<"TRT_PACKED_STORE_ALIAS_PROBE_COMPLETE"<<std::endl;return 0;
    }catch(const std::exception& error){output["error"]=error.what();std::ofstream(argv[1])<<output.dump(2);std::cerr<<error.what()<<'\n';return 1;}
}
