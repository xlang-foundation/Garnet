// Reuse only standalone fixture/ABI utilities; the correctness entry point is not run.
#define main standalone_correctness_main
#include "softmax_probe1.mm"
#undef main
#include <array>
#include <map>
static std::vector<float> load(const fs::path& path){
 std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"Missing retained input");auto bytes=f.tellg();require(bytes>0&&size_t(bytes)%4==0,"Bad payload size");std::vector<float> v(size_t(bytes)/4);f.seekg(0);f.read((char*)v.data(),bytes);require(bool(f),"Payload read failed");return v;
}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==4,"usage: screen combined.metal audited-fixture-directory new-output-directory");fs::path fixtures=argv[2],out=argv[3];require(!fs::exists(out),"Fresh output directory required");fs::create_directories(out);
 json admission;std::ifstream(fixtures/"report1.json")>>admission;require(admission["passed"]==true&&admission["cases"].size()==5,"Invalid fixtures");
 NSError* error=nil;auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];require(device&&queue,"Metal unavailable");auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];auto library=[device newLibraryWithSource:source options:options error:&error];if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
 // Every specialization is created before the first measurement.
 std::map<size_t,std::array<id<MTLComputePipelineState>,2>> plans;
 for(size_t extent:{128,256,2048,4096})for(int variant=0;variant<2;++variant){auto constants=[MTLFunctionConstantValues new];uint32_t bound=uint32_t(extent);[constants setConstantValue:&bound type:MTLDataTypeUInt atIndex:31];auto fn=[library newFunctionWithName:variant?@"chunk_mps_softmax_contiguous":@"chunk_mps_softmax" constantValues:constants error:&error];if(!fn)throw std::runtime_error(error.localizedDescription.UTF8String);auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);require(pipeline.threadExecutionWidth==32&&pipeline.maxTotalThreadsPerThreadgroup>=128,"Pipeline envelope differs");plans[extent][variant]=pipeline;}
 json results={{"scope","Serial isolated softmax command-buffer GPU spans; retained inputs and complete word/guard regression checks outside timestamps. Not model or serving performance."},{"warmups_per_variant",3},{"pairs_per_case",9},{"cases",json::array()},{"device",device.name.UTF8String}};
 constexpr size_t guard=16;
 for(size_t ci=0;ci<5;++ci){@autoreleasepool{
  auto c=admission["cases"][ci];size_t batch=c["batch"],chunk=c["chunk"],capacity=c["capacity"],extent=c["extent"],queries=batch*32*chunk;
  Params p{batch,chunk,32,8,128,capacity,4096,1024,1024,8*capacity*128,8*capacity*128,chunk+3,chunk+5,4096,4096,0};Pitch pitch{128,uint32_t(capacity+4),uint32_t(chunk+4)};
  auto get=[&](const char* name){return load(fixtures/("case"+std::to_string(ci)+name));};
  auto original=get("-input-cache.f32"),liveInput=get("-input-live.f32"),slotInput=get("-slots.f32"),activeInput=get("-active.f32"),expected=get("-control-cache.f32"),expectedLive=get("-control-live.f32");
  require(admit(p,slotInput.data()+guard,activeInput.data()+guard),"Fixture is ineligible");
  auto make=[&](const std::vector<float>& v){auto b=[device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];require(b!=nil,"Buffer allocation failed");return b;};auto cache=make(original),live=make(liveInput),slots=make(slotInput),active=make(activeInput);std::vector<float> blank(expectedLive.size(),word(0x7fc12345));auto probability=make(blank);
  auto run=[&](int variant){
   // Reset mutable probabilities before each GPU span. No readback/check in the span.
   std::memcpy(cache.contents,original.data(),original.size()*4);std::memcpy(probability.contents,blank.data(),blank.size()*4);
   auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];require(encoder!=nil,"No encoder");[encoder setComputePipelineState:plans.at(extent)[variant]];id<MTLBuffer> buffers[]={cache,live,probability,slots,active};for(size_t i=0;i<5;++i)[encoder setBuffer:buffers[i] offset:guard*4 atIndex:i];[encoder setBytes:&p length:sizeof(p) atIndex:5];[encoder setBytes:&pitch length:sizeof(pitch) atIndex:6];[encoder setThreadgroupMemoryLength:(extent*4+15)&~size_t(15) atIndex:0];if(!variant)[encoder setThreadgroupMemoryLength:(extent*2+15)&~size_t(15) atIndex:1];[encoder dispatchThreadgroups:MTLSizeMake(queries,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
   double elapsed=(command.GPUEndTime-command.GPUStartTime)*1000;require(std::isfinite(elapsed)&&elapsed>0,"Invalid GPU span");
   require(!std::memcmp(cache.contents,expected.data(),expected.size()*4)&&!std::memcmp(probability.contents,expectedLive.data(),expectedLive.size()*4),"Timed output/guard differs from audited control");
   require(!std::memcmp(live.contents,liveInput.data(),liveInput.size()*4)&&!std::memcmp(slots.contents,slotInput.data(),slotInput.size()*4)&&!std::memcmp(active.contents,activeInput.data(),activeInput.size()*4),"Readonly timed input changed");
   return elapsed;
  };
  for(int warm=0;warm<3;++warm){run(warm%2);run(1-warm%2);}
  json samples=json::array();for(int round=0;round<9;++round)for(int order=0;order<2;++order){int variant=(round+order+1)%2;double ms=run(variant);samples.push_back({{"round",round},{"order",order},{"variant",variant?"candidate":"control"},{"gpu_ms",ms},{"exact_parent_words",expected.size()+expectedLive.size()},{"readonly_inputs_unchanged",true}});}
  c["samples"]=samples;results["cases"].push_back(c);std::cout<<"Completed serial component cell "<<ci<<'\n'<<std::flush;
 }}
 results["passed"]=true;results["serving_speed_claim"]=false;results["native_integration"]=false;results["goal_complete"]=false;std::ofstream(out/"screen2.json")<<results.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
