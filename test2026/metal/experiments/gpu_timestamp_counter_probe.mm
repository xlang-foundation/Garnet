#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstring>
int main() { @autoreleasepool {
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 if(!d) return 2;
 NSMutableDictionary* out=[@{@"device":d.name,@"stage_boundary":@([d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]),@"dispatch_boundary":@([d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),@"diagnostic_only":@YES} mutableCopy];
 NSMutableArray* sets=[NSMutableArray array]; id<MTLCounterSet> ts=nil;
 for(id<MTLCounterSet> s in d.counterSets){[sets addObject:s.name];if([s.name isEqualToString:MTLCommonCounterSetTimestamp])ts=s;} out[@"counter_sets"]=sets;
 if(!ts || ![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])return 3;
 NSError* err=nil; MTLCounterSampleBufferDescriptor* sd=[MTLCounterSampleBufferDescriptor new];sd.counterSet=ts;sd.storageMode=MTLStorageModeShared;sd.sampleCount=4096;
 id<MTLCounterSampleBuffer> sb=[d newCounterSampleBufferWithDescriptor:sd error:&err];if(!sb){fprintf(stderr,"counter allocation: %s\n",err.description.UTF8String);return 4;}
 NSString* src=@"#include <metal_stdlib>\nusing namespace metal; kernel void nop(uint i [[thread_position_in_grid]]) {} kernel void work(device uint* out [[buffer(0)]],uint i [[thread_position_in_grid]]) {uint x=i+1; for(uint k=0;k<4096;k++)x=x*1664525u+1013904223u;out[i]=x;}";
 id<MTLLibrary> lib=[d newLibraryWithSource:src options:nil error:&err];if(!lib)return 5;
 id<MTLComputePipelineState> nop=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nop"] error:&err];
 id<MTLComputePipelineState> work=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"work"] error:&err];if(!nop||!work)return 6;
 id<MTLBuffer> b=[d newBufferWithLength:4096 options:MTLResourceStorageModeShared]; id<MTLCommandQueue> q=[d newCommandQueue];
 NSMutableArray* rows=[NSMutableArray array];bool ok=true;
 for(int mode=0;mode<3;mode++)for(int repeat=0;repeat<6;repeat++){
  std::memset(b.contents,0,4096);id<MTLCommandBuffer> cb=[q commandBuffer];
  auto encoder=[&](int start,int end){MTLComputePassDescriptor* p=[MTLComputePassDescriptor computePassDescriptor];p.sampleBufferAttachments[0].sampleBuffer=sb;p.sampleBufferAttachments[0].startOfEncoderSampleIndex=start;p.sampleBufferAttachments[0].endOfEncoderSampleIndex=end;return [cb computeCommandEncoderWithDescriptor:p];};
  MTLTimestamp c0,g0,c1,g1;[d sampleTimestamps:&c0 gpuTimestamp:&g0];
  if(mode){id<MTLComputeCommandEncoder> e=encoder(0,1);if(mode==2){[e setComputePipelineState:nop];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];}[e endEncoding];}
  id<MTLComputeCommandEncoder> e=mode?encoder(2,3):[cb computeCommandEncoder];[e setComputePipelineState:work];[e setBuffer:b offset:0 atIndex:0];[e dispatchThreads:MTLSizeMake(1024,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
  if(mode){e=encoder(4,5);if(mode==2){[e setComputePipelineState:nop];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];}[e endEncoding];}
  [cb commit];[cb waitUntilCompleted];[d sampleTimestamps:&c1 gpuTimestamp:&g1];
  bool valid=cb.status==MTLCommandBufferStatusCompleted;
  for(uint32_t i=0;i<1024;i++){uint32_t x=i+1;for(int k=0;k<4096;k++)x=x*1664525u+1013904223u;if(((uint32_t*)b.contents)[i]!=x)valid=false;}
  NSMutableDictionary* row=[@{@"mode":@(mode),@"repeat":@(repeat),@"output_correct":@(valid),@"cpu_before":@(c0),@"gpu_before":@(g0),@"cpu_after":@(c1),@"gpu_after":@(g1),@"command_gpu_ms":@((cb.GPUEndTime-cb.GPUStartTime)*1000)} mutableCopy];
  if(mode){NSData* samples=[sb resolveCounterRange:NSMakeRange(0,6)];valid=valid&&samples.length==6*sizeof(MTLCounterResultTimestamp);NSMutableArray* raw=[NSMutableArray array];const MTLCounterResultTimestamp* t=(const MTLCounterResultTimestamp*)samples.bytes;for(int j=0;j<6&&samples.length>=6*sizeof(*t);j++){[raw addObject:@(t[j].timestamp)];if(mode==2&&(!t[j].timestamp||t[j].timestamp==MTLCounterErrorValue))valid=false;}row[@"raw_samples"]=raw;
   if(valid&&g1>g0&&c1>c0){double scale=double(c1-c0)/double(g1-g0)/1e6;row[@"work_ms"]=@((t[3].timestamp-t[2].timestamp)*scale);valid=t[2].timestamp>0&&t[3].timestamp>=t[2].timestamp&&t[2].timestamp>=g0&&t[3].timestamp<=g1;if(mode==2)for(int j=0;j<6;j+=2)valid=valid&&t[j+1].timestamp>=t[j].timestamp;row[@"markers_bracket_work"]=@(t[1].timestamp<=t[2].timestamp&&t[4].timestamp>=t[3].timestamp);}else valid=false;
  }
  row[@"passed"]=@(valid);ok&=valid;[rows addObject:row];
 }
 out[@"rows"]=rows;out[@"passed"]=@(ok);out[@"counter_buffer_bytes"]=@(4096*sizeof(MTLCounterResultTimestamp));
 NSData* json=[NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingPrettyPrinted error:&err];fwrite(json.bytes,1,json.length,stdout);fputs("\n",stdout);return ok?0:7;
}}
