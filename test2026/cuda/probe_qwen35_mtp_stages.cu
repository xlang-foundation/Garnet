// Diagnostic only: unchanged production lowering, projection boundaries exposed.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "qwen35_lowering.h"
#include <Windows.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
using json=nlohmann::json;
using namespace nvinfer1;
using Ops=Garnet::Qwen35TRTOps;
void need(bool b,const char* message){if(!b)throw std::runtime_error(message);}
void cu(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Logger:ILogger{void log(Severity s,const char* m)noexcept override{if(s<=Severity::kWARNING)std::cerr<<m<<'\n';}};
struct Buffer{void* p=nullptr;size_t size;Buffer(size_t n):size(n){cu(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}};
std::vector<unsigned char> load(const std::string& file){
 std::ifstream f(file,std::ios::binary|std::ios::ate);need(bool(f),"Cannot open input");
 auto n=f.tellg();need(n>0,"Empty input");std::vector<unsigned char> v(size_t(n),0);f.seekg(0);f.read((char*)v.data(),v.size());need(bool(f),"Truncated input");return v;
}
size_t volume(Dims d){size_t n=1;for(int i=0;i<d.nbDims;++i){need(d.d[i]>0,"Nonstatic dimension");n*=d.d[i];}return n;}
std::string gpuUUID(const cudaDeviceProp& p){std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)s<<'-';s<<std::setw(2)<<unsigned((unsigned char)p.uuid.bytes[i]);}return s.str();}
void write(const std::string& name,const std::vector<unsigned char>& v){need(!std::filesystem::exists(name),"Preserve raw output");std::ofstream f(name,std::ios::binary);f.write((const char*)v.data(),v.size());need(bool(f),"Output write failed");}
int main(int argc,char** argv){
 if(argc!=6)return 2;
 const std::string prefix=argv[4];need(!std::filesystem::exists(prefix+".json"),"Preserve result");
 json result={{"diagnostic_complete",false},{"admitted_exact_native",false},{"runs",json::array()},{"pid",GetCurrentProcessId()}};
 auto save=[&]{std::ofstream f(prefix+".json");f<<result.dump(2);};
 try{
  auto manifest=json::parse(load(argv[1]));int index=std::stoi(argv[2]);bool observe=std::stoi(argv[3])!=0;
  auto test=manifest.at(index),c=test.at("config");int b=test.at("batch"),t=test.at("chunk"),h=c.at("hidden_size"),kv=c.at("num_key_value_heads"),dim=c.at("head_dim"),capacity=test.at("capacity");
  cu(cudaSetDevice(0));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,0));need(gpuUUID(prop)==argv[5],"Wrong GPU UUID");
  result["case"]=index;result["observe"]=observe;result["gpu_uuid"]=gpuUUID(prop);result["batch"]=b;result["chunk"]=t;
  Logger logger;std::unique_ptr<IBuilder> builder(createInferBuilder(logger));need(bool(builder),"Builder unavailable");
  std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U<<uint32_t(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
  std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());need(network&&config,"Network/config unavailable");
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,256ULL<<20);config->setBuilderOptimizationLevel(3);config->setMaxAuxStreams(0);
  std::deque<std::vector<float>> constants;std::vector<IPluginV2*> plugins;std::map<std::string,ITensor*> weights;
  std::map<std::string,std::vector<unsigned char>> weightBytes;std::map<std::string,std::vector<float>> normBytes;
  auto safe=load(test.at("checkpoint"));uint64_t headerSize=0;need(safe.size()>8,"Bad safetensors");std::memcpy(&headerSize,safe.data(),8);need(headerSize<safe.size()-8,"Bad safetensors header");
  auto header=json::parse(safe.begin()+8,safe.begin()+8+headerSize);
  for(auto entry=header.begin();entry!=header.end();++entry){
   std::string name=entry.key();if(name.rfind("mtp.",0)!=0)continue;auto meta=entry.value();need(meta.at("dtype")=="BF16","Expected BF16 fixture");
   auto shape=meta.at("shape").get<std::vector<int>>();size_t start=8+headerSize+meta.at("data_offsets")[0].get<size_t>(),end=8+headerSize+meta.at("data_offsets")[1].get<size_t>();need(end<=safe.size()&&end>start,"Bad tensor bounds");
   IConstantLayer* layer=nullptr;
   if(shape.size()==1){auto& v=normBytes[name];v.resize((end-start)/2);for(size_t i=0;i<v.size();++i){uint16_t u;std::memcpy(&u,safe.data()+start+2*i,2);uint32_t bits=uint32_t(u)<<16;std::memcpy(&v[i],&bits,4);}layer=network->addConstant(Ops::dims(shape),{DataType::kFLOAT,v.data(),int64_t(v.size())});}
   else {auto& v=weightBytes[name];v.assign(safe.begin()+start,safe.begin()+end);layer=network->addConstant(Ops::dims(shape),{DataType::kBF16,v.data(),int64_t(v.size()/2)});}
   need(layer,"Weight constant failed");weights[name]=layer->getOutput(0);weights[name]->setName(name.c_str());
  }
  auto lookup=[&](const std::string& name){return weights.at(name);};Ops ops(*network,constants,lookup,plugins);
  std::vector<std::vector<int>> shapes={{b,t,h},{b,t,h},{b,kv,capacity,dim},{b,kv,capacity,dim},{b,t,3},{b,t},{b,t}};
  std::vector<ITensor*> inputs;for(size_t i=0;i<shapes.size();++i){std::string name="input"+std::to_string(i);inputs.push_back(network->addInput(name.c_str(),DataType::kFLOAT,Ops::dims(shapes[i])));need(inputs.back(),"Input failed");}
  auto rope=c.at("rope_parameters");auto section=rope.at("mrope_section").get<std::vector<int>>();std::array<int,3> sections={section.at(0),section.at(1),section.at(2)};
  auto* packed=ops.mtpChunk(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],c.at("num_attention_heads"),kv,dim,int(dim*rope.at("partial_rotary_factor").get<float>()),sections,rope.at("rope_theta"),c.at("rms_norm_eps"));
  packed->setName("packed");network->markOutput(*packed);
  std::map<ITensor*,ILayer*> producers;for(int i=0;i<network->getNbLayers();++i){auto* l=network->getLayer(i);for(int j=0;j<l->getNbOutputs();++j)producers[l->getOutput(j)]=l;}
  json observations=json::array();std::map<ITensor*,std::string> observedNames;
  for(int i=0;i<network->getNbLayers();++i){auto* l=network->getLayer(i);if(l->getType()!=LayerType::kPLUGIN_V2)continue;
   auto* plugin=static_cast<IPluginV2Layer*>(l);if(std::string(plugin->getPlugin().getPluginType())!="GarnetBF16Projection")continue;
   std::string weight=l->getInput(1)->getName();need(weights.count(weight),"Projection weight identification failed");
   auto* input=l->getInput(0);auto* output=l->getOutput(0);auto* value=input;
   while(producers.count(value)&&producers.at(value)->getType()==LayerType::kSHUFFLE)value=producers.at(value)->getInput(0);
   need(producers.count(value)&&producers.at(value)->getType()==LayerType::kCAST,"Projection pre-cast source missing");auto* precast=producers.at(value)->getInput(0);need(precast->getType()==DataType::kFLOAT,"Precast must be FP32");
   json parts=json::array();int partIndex=0;for(auto* tensor:{precast,input,output}){std::string name="observe"+std::to_string(observations.size())+"_"+std::to_string(partIndex++);if(observe){if(observedNames.count(tensor))name=observedNames.at(tensor);else{observedNames[tensor]=name;tensor->setName(name.c_str());network->markOutput(*tensor);}}parts.push_back({{"name",name},{"shape",Ops::shape(tensor)},{"bf16",tensor->getType()==DataType::kBF16}});}
   observations.push_back({{"weight",weight},{"parts",parts}});
  }
  need(observations.size()==8,"Expected eight projection boundaries");result["observations"]=observations;save();
  std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));need(bool(plan),"Build failed");
  std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(),plan->size()));need(bool(engine),"Deserialize failed");
  std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());need(bool(context),"Context failed");
  std::map<std::string,std::unique_ptr<Buffer>> buffers;for(int i=0;i<engine->getNbIOTensors();++i){std::string name=engine->getIOTensorName(i);auto type=engine->getTensorDataType(name.c_str());need(type==DataType::kFLOAT||type==DataType::kBF16,"IO type invalid");buffers[name]=std::make_unique<Buffer>(volume(engine->getTensorShape(name.c_str()))*(type==DataType::kBF16?2:4));need(context->setTensorAddress(name.c_str(),buffers[name]->p),"Bind failed");}
  cudaStream_t stream;cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  const size_t hiddenW=size_t(t)*h,cacheW=size_t(kv)*capacity*dim,packedW=hiddenW+2*cacheW;
  std::vector<unsigned char> old[2];old[0].resize(size_t(b)*cacheW*4);old[1]=old[0];
  for(int step=0;step<int(test.at("steps").size());++step){
   auto s=test.at("steps")[step];std::vector<std::vector<unsigned char>> host;
   for(int i=0;i<7;++i){auto raw=(i==2||i==3)?old[i-2]:load(s.at("inputs")[i<2?i:i-2].at("path"));auto name="input"+std::to_string(i);need(raw.size()==buffers.at(name)->size,"Input size mismatch");cu(cudaMemcpyAsync(buffers.at(name)->p,raw.data(),raw.size(),cudaMemcpyHostToDevice,stream));host.push_back(std::move(raw));}
   need(context->enqueueV3(stream),"Enqueue failed");cu(cudaStreamSynchronize(stream));
   json run={{"step",step},{"tensors",json::array()}};std::vector<unsigned char> packedRaw;
   for(auto& item:buffers){need(item.second->size>0,"Empty buffer");std::vector<unsigned char> raw(item.second->size);cu(cudaMemcpy(raw.data(),item.second->p,raw.size(),cudaMemcpyDeviceToHost));
    if(item.first.rfind("input",0)==0){need(raw==host.at(std::stoi(item.first.substr(5))),"Input bytes mutated");continue;}
    const std::string path=prefix+"-s"+std::to_string(step)+"-"+item.first+".bin";write(path,raw);run["tensors"].push_back({{"name",item.first},{"file",path},{"bytes",raw.size()}});if(item.first=="packed")packedRaw=raw;
   }
   need(packedRaw.size()==size_t(b)*packedW*4,"Packed size invalid");
   std::vector<unsigned char> unpacked[3];unpacked[0].resize(size_t(b)*hiddenW*4);unpacked[1].resize(size_t(b)*cacheW*4);unpacked[2]=unpacked[1];
   for(int row=0;row<b;++row){auto* base=packedRaw.data()+size_t(row)*packedW*4;std::memcpy(unpacked[0].data()+size_t(row)*hiddenW*4,base,hiddenW*4);std::memcpy(unpacked[1].data()+size_t(row)*cacheW*4,base+hiddenW*4,cacheW*4);std::memcpy(unpacked[2].data()+size_t(row)*cacheW*4,base+(hiddenW+cacheW)*4,cacheW*4);}
   for(int part=0;part<3;++part){const char* name=part==0?"hidden":part==1?"key":"value";std::string reference=std::filesystem::path(prefix).parent_path().string()+"/qwen35-mtp-stage-v1-native-"+(index<5?"2b":"4b")+"-c"+std::to_string(index)+"-r0-s"+std::to_string(step)+"-"+name+".f32";auto expected=load(reference);need(unpacked[part]==expected,"Complete output differs from current native; attribution rejected");}
   old[0]=unpacked[1];old[1]=unpacked[2];run["complete_native_output_exact"]=true;result["runs"].push_back(run);save();
  }
  cu(cudaStreamDestroy(stream));context.reset();engine.reset();runtime.reset();network.reset();config.reset();builder.reset();for(auto* p:plugins)p->destroy();
  result["diagnostic_complete"]=true;result["admitted_exact_native"]=true;result["scope"]="Exact current-native synthetic component output admission only; no full model, quality, performance or memory acceptance.";save();std::cout<<"MTP_STAGE_COMPONENT_ADMITTED "<<index<<' '<<observe<<'\n';return 0;
 }catch(const std::exception& e){result["failure"]=e.what();save();std::cerr<<e.what()<<'\n';return 1;}
}
