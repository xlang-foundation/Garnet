#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
struct Params {uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
struct Pitch {uint32_t packed,cacheScore,liveScore;};
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float word(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
static float input(size_t i,uint32_t salt){uint32_t x=uint32_t(i)^salt;x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return float(int(x%20001)-10000)*0.001f;}
static bool admit(const Params& p,const float* slots,const float* active){
 if(p.batch>4||!p.chunk||p.chunk>512||!p.capacity||p.capacity>4096)return false;
 for(size_t b=0;b<p.batch;++b){size_t count=0;while(count<p.chunk&&active[b*p.aStride+count]!=0){if(!std::isfinite(active[b*p.aStride+count]))return false;++count;}
  for(size_t t=count;t<p.chunk;++t)if(active[b*p.aStride+t]!=0)return false;
  if(count){float first=slots[b*p.sStride];if(!std::isfinite(first)||first<0||first!=std::floor(first)||double(first)+count>p.capacity)return false;
   for(size_t t=0;t<count;++t)if(slots[b*p.sStride+t]!=first+float(t))return false;}
 }return true;
}
static void save(const fs::path& path,const void* data,size_t bytes){std::ofstream f(path,std::ios::binary);f.write((const char*)data,bytes);require(bool(f),"payload write failed");}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==3,"usage: probe combined.metal new-output-directory");fs::path out=argv[2];require(!fs::exists(out),"fresh directory required");fs::create_directories(out);
 NSError* error=nil;auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"No Metal device");
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];require(source!=nil,"Missing shader");
 auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto queue=[device newCommandQueue];require(queue!=nil,"No queue");
 json report={{"scope","Standalone untimed correctness screen; unchanged original MPS softmax versus contiguous-control specialization. No native integration, inference, timing, or serving claim."},{"cases",json::array()},{"device",device.name.UTF8String}};
 const size_t guard=16;const uint32_t poison=0x7fc12345;
 // Dynamic actual counts, ragged owners, offsets and physically larger context bounds.
 const size_t geometries[][5]={{1,8,128,0,128},{4,17,256,128,256},{4,128,4096,1920,2048},{4,128,4096,3840,4096},{4,128,4096,1920,2048}};
 for(size_t ci=0;ci<5;++ci){auto& shape=geometries[ci];size_t batch=shape[0],chunk=shape[1],capacity=shape[2],first=shape[3],extent=shape[4],heads=32,queries=batch*heads*chunk;
  Params p{batch,chunk,heads,8,128,capacity,4096,1024,1024,8*capacity*128,8*capacity*128,chunk+3,chunk+5,4096,4096,0};Pitch pitch{128,uint32_t(capacity+4),uint32_t(chunk+4)};
  size_t cacheWords=guard*2+queries*pitch.cacheScore,liveWords=guard*2+queries*pitch.liveScore;
  std::vector<float> cacheSource(cacheWords,word(poison)),liveSource(liveWords,word(poison)),slotSource(guard*2+batch*p.sStride,word(poison)),activeSource(guard*2+batch*p.aStride,word(poison));
  for(size_t q=0;q<queries;++q){for(size_t c=0;c<capacity;++c)cacheSource[guard+q*pitch.cacheScore+c]=input(q*capacity+c,17);for(size_t c=0;c<chunk;++c)liveSource[guard+q*pitch.liveScore+c]=input(q*chunk+c,23);}
  std::vector<size_t> counts;
  for(size_t b=0;b<batch;++b){size_t count=chunk;if(batch==4&&ci!=4){if(b==1)count=chunk-3;if(b==2)count=1;if(b==3)count=0;}counts.push_back(count);
   for(size_t t=0;t<chunk;++t){slotSource[guard+b*p.sStride+t]=float(first+t);activeSource[guard+b*p.aStride+t]=t<count?1.0f:0.0f;}}
  auto slots=slotSource.data()+guard,active=activeSource.data()+guard;require(admit(p,slots,active),"Valid controls rejected");
  size_t rejected=0;auto changed=slotSource;changed[guard]=NAN;require(!admit(p,changed.data()+guard,active),"NaN admitted");++rejected;
  changed=slotSource;changed[guard]=0.5f;require(!admit(p,changed.data()+guard,active),"Fractional slot admitted");++rejected;
  changed=slotSource;changed[guard+1]=changed[guard];require(!admit(p,changed.data()+guard,active),"Overwrite admitted");++rejected;
  auto gap=activeSource;gap[guard+1]=0;require(!admit(p,slots,gap.data()+guard),"Gap admitted");++rejected;
  auto pending=p;pending.capacity=first+chunk-1;require(!admit(pending,slots,active),"Out-of-bound prefix admitted");++rejected;
  auto make=[&](const std::vector<float>& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];require(b!=nil,"Buffer allocation failed");return b;};
  auto cache=make(cacheSource),live=make(liveSource),position=make(slotSource),flags=make(activeSource);
  std::vector<float> probabilitySource(liveWords,word(poison));auto probability=make(probabilitySource);
  std::vector<float> controlCache,controlProb;json cell={{"batch",batch},{"chunk",chunk},{"capacity",capacity},{"extent",extent},{"first",first},{"active_counts",counts},{"rejected_control_variants",rejected},{"variants",json::array()}};
  for(int variant=0;variant<2;++variant){auto constants=[MTLFunctionConstantValues new];uint32_t bound=uint32_t(extent);[constants setConstantValue:&bound type:MTLDataTypeUInt atIndex:31];
   NSString* name=variant?@"chunk_mps_softmax_contiguous":@"chunk_mps_softmax";auto function=[library newFunctionWithName:name constantValues:constants error:&error];if(!function)throw std::runtime_error(error.localizedDescription.UTF8String);
   auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);require(pipeline.threadExecutionWidth==32&&pipeline.maxTotalThreadsPerThreadgroup>=128,"Invalid pipeline envelope");
   std::memcpy(cache.contents,cacheSource.data(),cacheWords*4);std::memcpy(probability.contents,probabilitySource.data(),liveWords*4);
   auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];require(encoder!=nil,"No encoder");[encoder setComputePipelineState:pipeline];
   id<MTLBuffer> buffers[]={cache,live,probability,position,flags};for(size_t i=0;i<5;++i)[encoder setBuffer:buffers[i] offset:guard*4 atIndex:i];
   [encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];[encoder setThreadgroupMemoryLength:(extent*4+15)&~size_t(15) atIndex:0];if(!variant)[encoder setThreadgroupMemoryLength:(extent*2+15)&~size_t(15) atIndex:1];
   [encoder dispatchThreadgroups:MTLSizeMake(queries,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
   require(!std::memcmp(live.contents,liveSource.data(),liveWords*4)&&!std::memcmp(position.contents,slotSource.data(),slotSource.size()*4)&&!std::memcmp(flags.contents,activeSource.data(),activeSource.size()*4),"Readonly input changed");
   auto actual=(const float*)cache.contents;auto probs=(const float*)probability.contents;size_t guardChecks=0,logical=0;
   for(size_t i=0;i<cacheWords;++i){bool output=i>=guard&&i<guard+queries*pitch.cacheScore&&(i-guard)%pitch.cacheScore<extent;if(output){require(std::isfinite(actual[i]),"Nonfinite cache probability");++logical;}else{require(bits(actual[i])==bits(cacheSource[i]),"Cache guard/tail changed");++guardChecks;}}
   for(size_t i=0;i<liveWords;++i){bool output=i>=guard&&i<guard+queries*pitch.liveScore&&(i-guard)%pitch.liveScore<chunk;if(output){require(std::isfinite(probs[i]),"Nonfinite live probability");++logical;}else{require(bits(probs[i])==poison,"Live output guard changed");++guardChecks;}}
   if(!variant){controlCache.assign(actual,actual+cacheWords);controlProb.assign(probs,probs+liveWords);}else{require(!std::memcmp(actual,controlCache.data(),cacheWords*4)&&!std::memcmp(probs,controlProb.data(),liveWords*4),"Candidate differs from original bits");}
   fs::path prefix=out/("case"+std::to_string(ci)+(variant?"-candidate":"-control"));save(prefix.string()+"-cache.f32",actual,cacheWords*4);save(prefix.string()+"-live.f32",probs,liveWords*4);
   cell["variants"].push_back({{"variant",variant?"candidate":"control"},{"logical_words",logical},{"guard_words",guardChecks},{"threads",128},{"static_threadgroup_bytes",pipeline.staticThreadgroupMemoryLength},{"dynamic_threadgroup_bytes",extent*(variant?4:6)},{"exact_to_control",true}});
  }
  save(out/("case"+std::to_string(ci)+"-input-cache.f32"),cacheSource.data(),cacheWords*4);save(out/("case"+std::to_string(ci)+"-input-live.f32"),liveSource.data(),liveWords*4);save(out/("case"+std::to_string(ci)+"-slots.f32"),slotSource.data(),slotSource.size()*4);save(out/("case"+std::to_string(ci)+"-active.f32"),activeSource.data(),activeSource.size()*4);
  report["cases"].push_back(cell);std::cout<<"Correctness case "<<ci<<" exact, guarded, readonly inputs unchanged\n"<<std::flush;
 }
 report["passed"]=true;report["timing_collected"]=false;report["native_integration"]=false;report["goal_complete"]=false;std::ofstream(out/"report1.json")<<report.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
