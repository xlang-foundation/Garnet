#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(int argc,char** argv){@autoreleasepool{try{
 need(argc==3,"Usage: coordinates source.metal output.json");auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* e=nil;
 auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&e];auto opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion4_0;opts.mathMode=MTLMathModeSafe;auto lib=[device newLibraryWithSource:source options:opts error:&e];if(!lib)throw std::runtime_error(e.localizedDescription.UTF8String);
 auto run=[&](NSString* name,size_t n){auto fn=[lib newFunctionWithName:name];need(fn!=nil,"Missing probe function");auto p=[device newComputePipelineStateWithFunction:fn error:&e];if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);std::vector<uint32_t> init(n+32,0xdeadbeef);auto buffer=[device newBufferWithBytes:init.data() length:init.size()*4 options:MTLResourceStorageModeShared];auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];[enc setBuffer:buffer offset:64 atIndex:0];[enc dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Probe failed");std::vector<uint32_t> result((uint32_t*)buffer.contents,(uint32_t*)buffer.contents+init.size());for(size_t i=0;i<16;++i)need(result[i]==0xdeadbeef&&result[n+16+i]==0xdeadbeef,"Probe guard changed");return std::vector<uint32_t>(result.begin()+16,result.end()-16);};
 json report={{"device",device.name.UTF8String},{"math_mode","Safe"},{"relaxed_precision",false},{"capacities",run(@"strict_mpp_layout",192)},{"old_register_guard_flags",run(@"strict_mpp_register_layout",32)},{"coordinates",run(@"strict_mpp_raw_coords",9216)},{"parent_guards_intact",true},{"attention_kernel_launched",false}};std::ofstream(argv[2])<<report.dump(2)<<'\n';std::cout<<"Coordinate-only probe completed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
