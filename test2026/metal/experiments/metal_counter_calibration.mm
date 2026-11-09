#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <mach/mach_time.h>
int main(){@autoreleasepool{
 auto d=MTLCreateSystemDefaultDevice();NSError* error=nil;
 auto lib=[d newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;kernel void work(device float* p [[buffer(0)]],uint t [[thread_position_in_grid]]) {float x=float(t)*0.001f;for(uint i=0;i<8192;++i)x=fma(x,0.99999f,0.0001f);p[t]=x;}" options:nil error:&error];
 if(!lib){std::cerr<<error.localizedDescription.UTF8String;return 1;}
 auto pipeline=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"work"] error:&error];
 auto desc=[MTLCounterSampleBufferDescriptor new];for(id<MTLCounterSet> set in d.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp])desc.counterSet=set;
 desc.storageMode=MTLStorageModeShared;desc.sampleCount=4;
 auto samples=[d newCounterSampleBufferWithDescriptor:desc error:&error];
 if(!samples||!pipeline){std::cerr<<"setup failed";return 1;}
 auto queue=[d newCommandQueue];auto data=[d newBufferWithLength:65536*sizeof(float) options:MTLResourceStorageModeShared];
 MTLTimestamp c0=0,g0=0,c1=0,g1=0;[d sampleTimestamps:&c0 gpuTimestamp:&g0];auto mach0=mach_absolute_time();
 auto command=[queue commandBuffer];
 for(NSUInteger i=0;i<2;++i){auto pass=[MTLComputePassDescriptor computePassDescriptor];auto a=pass.sampleBufferAttachments[0];a.sampleBuffer=samples;a.startOfEncoderSampleIndex=i*2;a.endOfEncoderSampleIndex=i*2+1;
 auto e=[command computeCommandEncoderWithDescriptor:pass];[e setComputePipelineState:pipeline];[e setBuffer:data offset:0 atIndex:0];[e dispatchThreads:MTLSizeMake(65536,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];}
 [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)return 2;
 [d sampleTimestamps:&c1 gpuTimestamp:&g1];auto mach1=mach_absolute_time();
 auto resolved=[samples resolveCounterRange:NSMakeRange(0,4)];if(resolved.length!=4*sizeof(MTLCounterResultTimestamp))return 3;
 const auto* t=static_cast<const MTLCounterResultTimestamp*>(resolved.bytes);auto hz=[d queryTimestampFrequency];
 mach_timebase_info_data_t base;mach_timebase_info(&base);
 double nsPerTick=double(c1-c0)/double(g1-g0);double span=double(t[3].timestamp-t[0].timestamp)*nsPerTick/1e6;double gpu=(command.GPUEndTime-command.GPUStartTime)*1000.0;
 std::cout<<"{\"timestamp_hz\":"<<hz<<",\"ns_per_counter_tick\":"<<nsPerTick<<",\"cpu0\":"<<c0<<",\"cpu1\":"<<c1<<",\"gpu0\":"<<g0<<",\"gpu1\":"<<g1<<",\"mach0\":"<<mach0<<",\"mach1\":"<<mach1<<",\"mach_numer\":"<<base.numer<<",\"mach_denom\":"<<base.denom<<",\"encoder_span_ms\":"<<span<<",\"command_gpu_ms\":"<<gpu<<",\"ticks\":[";
 for(int i=0;i<4;++i){if(i)std::cout<<',';std::cout<<t[i].timestamp;if(t[i].timestamp==MTLCounterErrorValue)return 4;}std::cout<<"]}\n";
 if(!(span>0&&span<=gpu+0.01&&span>=gpu*0.5))return 5;
}}
