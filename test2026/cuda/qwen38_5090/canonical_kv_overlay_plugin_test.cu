// SPDX-License-Identifier: Apache-2.0
// Standalone TensorRT operator qualification; no Garnet DLL/model mutation.
#include <NvInfer.h>
#include "canonical_kv_overlay_plugin.h"
#include "canonical_kv_overlay_fixture.cuh"
#include <array>
#include <Windows.h>
#include <cstdint>
#include <fstream>
#include <memory>
#include <limits>
using namespace nvinfer1;
static int assertions=0;
void Expect(bool value,const char* message){if(!value)throw std::runtime_error(message);++assertions;}
struct Logger:ILogger{void log(Severity s,const char* m) noexcept override{if(s<=Severity::kWARNING)fprintf(stderr,"TRT: %s\n",m);}};
template<class T> std::unique_ptr<T> Own(T* p){if(!p)throw std::runtime_error("TensorRT creation failed");return std::unique_ptr<T>(p);}
struct DestroyPlugin{void operator()(IPluginV2* p)const noexcept{if(p)p->destroy();}};
using PluginOwner=std::unique_ptr<IPluginV2,DestroyPlugin>;
std::vector<float> Load(const std::string& name,size_t words){std::ifstream f(name,std::ios::binary|std::ios::ate);Expect(bool(f),"Missing qualified context");Expect(size_t(f.tellg())==words*4,"Qualified context size mismatch");f.seekg(0);std::vector<float> result(words);f.read((char*)result.data(),words*4);Expect(bool(f),"Qualified context read failed");return result;}
std::array<PluginTensorDesc,8> Descriptors(const Garnet::CanonicalKVOverlayConfig& c){
 std::array<PluginTensorDesc,8> d{};
 d[0].dims=Dims4{4,4,24,256};d[1].dims=Dims4{4,4,4,256};d[2].dims=d[1].dims;d[3].dims=d[0].dims;
 d[4].dims=Dims2{4,int(c.state_width)};d[5].dims=Dims2{4,4};d[6].dims=d[5].dims;d[7].dims=d[0].dims;
 for(auto& x:d){x.type=DataType::kFLOAT;x.format=TensorFormat::kLINEAR;}return d;
}
void Metadata(){
 Expect(Garnet::EnsureCanonicalKVOverlayPluginRegistered(),"Plugin registry failure");
 auto* creator=getPluginRegistry()->getPluginCreator("GarnetQwen35CanonicalKVOverlay","1","");Expect(creator!=nullptr,"Creator missing");
 for(int layer=0;layer<16;++layer)for(int strategy=0;strategy<2;++strategy){
  const std::int64_t key=2482176+std::int64_t(layer)*10870784;
  Garnet::CanonicalKVOverlayConfig c{{4,4,24,4,4096,256},173932544,key,key+4194304,strategy};
  Expect(Garnet::ValidCanonicalKVOverlayConfig(c),"Valid config rejected");Garnet::CanonicalKVOverlayPlugin p(c);auto desc=Descriptors(c);
  Expect(p.getNbOutputs()==1&&p.getSerializationSize()==80,"Output/wire contract mismatch");
  Expect(p.getWorkspaceSize(desc.data(),7,&desc[7],1)==(strategy?Garnet::Qwen35AttentionCoreWorkspace(c.attention):0),"Workspace contract mismatch");
  for(int i=0;i<8;++i)Expect(p.supportsFormatCombination(i,desc.data(),7,1),"FP32 linear format rejected");
  auto bad=desc;bad[4].type=DataType::kINT32;Expect(!p.supportsFormatCombination(4,bad.data(),7,1),"Integer canonical format accepted");
  Expect(!p.supportsFormatCombination(8,desc.data(),7,1)&&!p.supportsFormatCombination(0,desc.data(),8,1),"Invalid format indices/count accepted");
  p.setPluginNamespace("study-clone");PluginOwner cloned(p.clone());Expect(cloned&&std::string(cloned->getPluginNamespace())=="study-clone","Clone namespace failure");
  std::array<std::int64_t,10> wire{};p.serialize(wire.data());std::array<std::int64_t,10> again{};cloned->serialize(again.data());Expect(wire==again,"Clone serialization changed");
  PluginOwner restored(creator->deserializePlugin("restore",wire.data(),sizeof(wire)));Expect(restored!=nullptr,"Deserialize failed");again.fill(0);restored->serialize(again.data());Expect(wire==again,"Roundtrip wire changed");
  Expect(creator->deserializePlugin("bad",wire.data(),sizeof(wire)-1)==nullptr&&creator->deserializePlugin("bad",nullptr,sizeof(wire))==nullptr,"Malformed wire accepted");
  for(int field=0;field<10;++field){auto invalid=wire;invalid[field]=field==9?2:wire[field]+1;PluginOwner rejected(creator->deserializePlugin("bad",invalid.data(),sizeof(invalid)));Expect(!rejected,"Invalid serialized profile accepted");}
  auto invalid=c;invalid.strategy=2;Expect(!Garnet::ValidCanonicalKVOverlayConfig(invalid),"Unqualified hybrid operator admitted");
 }
 printf("CANONICAL_PLUGIN_METADATA_PASS checks=%d\n",assertions);fflush(stdout);
}
struct Engine {std::unique_ptr<IRuntime> runtime;std::unique_ptr<ICudaEngine> engine;std::unique_ptr<IExecutionContext> context;};
Engine Build(Logger& logger,Garnet::CanonicalKVOverlayConfig c,const std::string& prefix){
 auto builder=Own(createInferBuilder(logger));auto network=Own(builder->createNetworkV2(1U<<int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
 auto config=Own(builder->createBuilderConfig());config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
 auto desc=Descriptors(c);const char* names[7]={"q","k","v","gate","canonical","slots","active"};ITensor* inputs[7];
 for(int i=0;i<7;++i){inputs[i]=network->addInput(names[i],DataType::kFLOAT,desc[i].dims);Expect(inputs[i]!=nullptr,"Network input creation failed");}
 Garnet::CanonicalKVOverlayPlugin plugin(c);auto* layer=network->addPluginV2(inputs,7,plugin);Expect(layer!=nullptr,"Canonical plugin layer creation failed");
 layer->setName("readonly_canonical_kv_overlay");auto* output=layer->getOutput(0);output->setName("context");network->markOutput(*output);
 auto plan=Own(builder->buildSerializedNetwork(*network,*config));std::ofstream file(prefix+".engine",std::ios::binary);file.write((const char*)plan->data(),plan->size());file.close();Expect(bool(file),"Engine evidence write failed");
 Engine result;result.runtime=Own(createInferRuntime(logger));result.engine=Own(result.runtime->deserializeCudaEngine(plan->data(),plan->size()));result.context=Own(result.engine->createExecutionContext());
 auto inspector=Own(result.engine->createEngineInspector());std::ofstream details(prefix+"-inspector.json");details<<inspector->getEngineInformation(LayerInformationFormat::kJSON);details.close();Expect(bool(details),"Inspector evidence write failed");
 Expect(result.engine->getNbIOTensors()==8,"Unexpected engine IO count");auto shape=result.engine->getTensorShape("canonical");Expect(shape.nbDims==2&&shape.d[0]==4&&shape.d[1]==173932544,"Engine canonical input was reshaped");return result;
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("Qualified strided result folder and output prefix required");Metadata();Logger logger;
 auto module=GetModuleHandleA("nvinfer_10.dll");char libraryPath[32768]{};Expect(module!=nullptr,"TensorRT module not loaded");
 const DWORD libraryLength=GetModuleFileNameA(module,libraryPath,DWORD(sizeof(libraryPath)));Expect(libraryLength>0&&libraryLength<sizeof(libraryPath),"TensorRT module path unavailable");
 printf("TRT_LOADED_LIBRARY path=%s\n",libraryPath);fflush(stdout);
 std::vector<Engine> engines;for(int which=0;which<2;++which)for(int strategy=0;strategy<2;++strategy){
  Garnet::CanonicalKVOverlayConfig c{{4,4,24,4,4096,256},std::int64_t(kCanonicalStride),std::int64_t(kCanonicalKeyOffsets[which]),std::int64_t(kCanonicalValueOffsets[which]),strategy};
  engines.push_back(Build(logger,c,std::string(argv[2])+"-layer"+std::to_string(kCanonicalLayers[which])+"-strategy"+std::to_string(strategy)));
 }
 std::vector<Case> cases={{4,4,24,4,4096,256,32,0},{4,4,24,4,4096,256,128,0},{4,4,24,4,4096,256,512,0},{4,4,24,4,4096,256,3968,0},{4,4,24,4,4096,256,4096,0},{4,4,24,4,4096,256,128,1},{4,4,24,4,4096,256,128,2},{4,4,24,4,4096,256,128,3},{4,4,24,4,4096,256,511,0},{4,4,24,4,4096,256,513,0},{4,4,24,4,4096,256,1024,0},{4,4,24,4,4096,256,2048,0}};
 cudaStream_t stream;Check(cudaStreamCreate(&stream));cudaEvent_t start,stop;Check(cudaEventCreate(&start));Check(cudaEventCreate(&stop));
 for(size_t ci=0;ci<cases.size();++ci){const auto a=cases[ci];Garnet::Qwen35AttentionCoreConfig c{a.batch,a.tokens,a.heads,a.kv,a.capacity,a.dim};
  size_t qsize=size_t(c.batch)*c.tokens*c.heads*c.dim,kvsize=size_t(c.batch)*c.tokens*c.kv*c.dim,cache=size_t(c.batch)*c.kv*c.capacity*c.dim;
  std::vector<std::vector<float>> host(8);host[0].resize(qsize);host[1].resize(kvsize);host[2].resize(kvsize);host[3].resize(qsize);host[4].resize(cache);host[5].resize(cache);host[6].resize(c.batch*c.tokens);host[7].resize(c.batch*c.tokens);
  for(int i=0;i<6;++i)for(size_t j=0;j<host[i].size();++j)host[i][j]=Value(j,11+i*19)*(i==3?16.f:1.f);
  auto isActive=[&](int b,int t){return a.pattern==0 || (a.pattern==1 && b!=3 && t%2==0) || (a.pattern==3 && (b==0 || (b==1 && t==0) || (b==2 && t%2==0)));};
  for(int b=0;b<c.batch;++b){
   int count=0;for(int t=0;t<c.tokens;++t)count+=isActive(b,t)?1:0;
   const int mixedPrefixes[4]={32,128,512,3968};
   int prefix=a.pattern==3?mixedPrefixes[b]:a.prefix,slot=std::min(c.capacity-std::max(count,1),prefix-1);
   for(int t=0;t<c.tokens;++t){bool active=isActive(b,t);host[6][b*c.tokens+t]=float(std::min(slot,c.capacity-1));host[7][b*c.tokens+t]=active?1.f:0.f;if(active)++slot;}
  }
  printf("B4_ATTENTION_CONTROLS case=%zu pattern=%d slots=",ci,a.pattern);
  for(float v:host[6])printf("%.0f,",v);printf(" active=");for(float v:host[7])printf("%.0f,",v);printf("\n");fflush(stdout);
  const auto expected=Load(std::string(argv[1])+"/context-case"+std::to_string(ci)+"-baseline.f32",qsize);
  std::vector<std::unique_ptr<Buffer>> inputs;for(int i=0;i<8;++i){inputs.emplace_back(new Buffer(host[i].size()));inputs.back()->upload(host[i]);}
  const size_t which=ci%2;CanonicalFixture canonical(kCanonicalStride,kCanonicalKeyOffsets[which],kCanonicalValueOffsets[which],size_t(c.kv)*c.capacity*c.dim);canonical.initialize(host[4],host[5]);
  Buffer output(qsize),workspace(Garnet::Qwen35AttentionCoreWorkspace(c)/4);
  std::array<const void*,7> bindings={inputs[0]->data(),inputs[1]->data(),inputs[2]->data(),inputs[3]->data(),canonical.base+CanonicalFixture::guard,inputs[6]->data(),inputs[7]->data()};
  const char* names[7]={"q","k","v","gate","canonical","slots","active"};
  for(int strategy=0;strategy<2;++strategy){
   auto& engine=engines[which*2+strategy];Garnet::CanonicalKVOverlayConfig configuration{c,std::int64_t(kCanonicalStride),std::int64_t(kCanonicalKeyOffsets[which]),std::int64_t(kCanonicalValueOffsets[which]),strategy};
   Garnet::CanonicalKVOverlayPlugin plugin(configuration);auto desc=Descriptors(configuration);void* outs[1]={output.data()};void* scratch=strategy?workspace.data():nullptr;
   for(int i=0;i<7;++i)Expect(engine.context->setTensorAddress(names[i],const_cast<void*>(bindings[i])),"Input binding failed");Expect(engine.context->setTensorAddress("context",output.data()),"Output binding failed");
   Expect(engine.context->enqueueV3(stream),"Engine eager enqueue failed");Check(cudaStreamSynchronize(stream));Expect(Equal(output.read(),expected),"Engine context differs from qualified legacy");
   const auto prefix=std::string(argv[2])+"-case"+std::to_string(ci)+"-strategy"+std::to_string(strategy);
   Save(prefix+"-eager.f32",output.read());Graph graph;
   Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));Expect(engine.context->enqueueV3(stream),"Engine capture enqueue failed");Check(cudaStreamEndCapture(stream,&graph.graph));Check(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
   for(int replay=0;replay<3;++replay){Check(cudaGraphLaunch(graph.executable,stream));Check(cudaStreamSynchronize(stream));Expect(Equal(output.read(),expected),"Engine graph context differs");}
   Save(prefix+"-graph.f32",output.read());
   PluginOwner cloned(plugin.clone());Expect(cloned!=nullptr,"Clone allocation failed");auto* dynamicClone=static_cast<IPluginV2DynamicExt*>(cloned.get());
   Expect(dynamicClone->enqueue(desc.data(),&desc[7],bindings.data(),outs,scratch,stream)==0,"Cloned direct enqueue failed");Check(cudaStreamSynchronize(stream));Expect(Equal(output.read(),expected),"Clone context differs");Save(prefix+"-clone.f32",output.read());
   std::array<std::int64_t,10> wire{};plugin.serialize(wire.data());auto* creator=getPluginRegistry()->getPluginCreator("GarnetQwen35CanonicalKVOverlay","1","");PluginOwner restored(creator->deserializePlugin("roundtrip",wire.data(),sizeof(wire)));Expect(restored!=nullptr,"Restore allocation failed");auto* dynamicRestored=static_cast<IPluginV2DynamicExt*>(restored.get());
   Expect(dynamicRestored->enqueue(desc.data(),&desc[7],bindings.data(),outs,scratch,stream)==0,"Restored direct enqueue failed");Check(cudaStreamSynchronize(stream));Expect(Equal(output.read(),expected),"Restored context differs");Save(prefix+"-restored.f32",output.read());
   auto rejected=[&](const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* pointers,void* target,void* work){void* invalidOut[1]={target};Expect(plugin.enqueue(in,out,pointers,invalidOut,work,stream)!=0,"Invalid operator enqueue accepted");};
   auto bad=desc;bad[4].dims.d[1]-=1;rejected(bad.data(),&desc[7],bindings.data(),output.data(),scratch);
   bad=desc;bad[0].type=DataType::kINT32;rejected(bad.data(),&desc[7],bindings.data(),output.data(),scratch);
   bad=desc;bad[7].dims.d[3]=128;rejected(desc.data(),&bad[7],bindings.data(),output.data(),scratch);
   rejected(desc.data(),&desc[7],bindings.data(),canonical.key(),scratch);
   rejected(desc.data(),&desc[7],bindings.data(),canonical.key()-1,scratch);
   rejected(desc.data(),&desc[7],bindings.data(),inputs[0]->data()+1,scratch);
   auto nullBindings=bindings;nullBindings[4]=nullptr;rejected(desc.data(),&desc[7],nullBindings.data(),output.data(),scratch);
   if(strategy){rejected(desc.data(),&desc[7],bindings.data(),output.data(),nullptr);rejected(desc.data(),&desc[7],bindings.data(),output.data(),canonical.value());}
   Check(cudaStreamSynchronize(stream));Expect(Equal(output.read(),expected),"Rejected operations changed context");
   Expect(Equal(canonical.read(false),host[4])&&Equal(canonical.read(true),host[5]),"Readonly operator mutated KV");canonical.padding(stream);
   for(int i=0;i<8;++i){inputs[i]->guards();Expect(Equal(inputs[i]->read(),host[i]),"Operator mutated an input");}output.guards();workspace.guards();
   printf("{\"case\":%zu,\"strategy\":%d,\"canonical_layer\":%d,\"status\":\"operator_correctness_pass\",\"engine_legacy_bit_exact\":true,\"graph_replays\":3,\"clone_exact\":true,\"restored_exact\":true,\"readonly_canonical_preserved\":true,\"negative_enqueues_preserved_state\":true}\n",ci,strategy,kCanonicalLayers[which]);fflush(stdout);
  }
 }
 Check(cudaEventDestroy(start));Check(cudaEventDestroy(stop));Check(cudaStreamDestroy(stream));
 printf("CANONICAL_PLUGIN_QUALIFICATION_PASS checks=%d cases=24 engines=4; no model or timing claim\n",assertions);return 0;
}catch(const std::exception& error){fprintf(stderr,"CANONICAL_PLUGIN_FAILURE %s\n",error.what());return 1;}}
