// SPDX-License-Identifier: Apache-2.0
// Process-local diagnostic capture scope; does not change Garnet kernels/API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <mutex>
namespace { std::mutex lock; char failure[2048]={}; bool owned=false;
int Fail(const char* message) {std::snprintf(failure,sizeof(failure),"%s",message?message:"Unknown Metal capture error");std::fprintf(stderr,"Diagnostic capture: %s\n",failure);return 1;}}
extern "C" const char* garnet_diagnostic_capture_error(){return failure;}
extern "C" int garnet_diagnostic_capture_start(const char* path){std::lock_guard<std::mutex> guard(lock);@autoreleasepool {
 failure[0]=0; if(!path||path[0]!='/')return Fail("Capture path must be absolute");
 NSString* name=[NSString stringWithUTF8String:path]; if(!name)return Fail("Capture path must be UTF8");
 if([[NSFileManager defaultManager] fileExistsAtPath:name])return Fail("Refusing to replace existing capture");
 id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if(!device)return Fail("No default Metal device");
 MTLCaptureManager* manager=[MTLCaptureManager sharedCaptureManager];
 if(owned||manager.isCapturing)return Fail("Refusing to replace an active capture");
 if(![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])return Fail("GPU document unsupported; require process-local MTL_CAPTURE_ENABLED=1 before device initialization");
 MTLCaptureDescriptor* desc=[MTLCaptureDescriptor new]; desc.captureObject=device; desc.destination=MTLCaptureDestinationGPUTraceDocument; desc.outputURL=[NSURL fileURLWithPath:name]; NSError* error=nil;
 if(![manager startCaptureWithDescriptor:desc error:&error])return Fail(error.localizedDescription.UTF8String);
 owned=manager.isCapturing; return owned?0:Fail("Capture manager did not start");
}}
extern "C" int garnet_diagnostic_capture_stop(){std::lock_guard<std::mutex> guard(lock);@autoreleasepool {
 if(!owned)return Fail("No owned capture to stop"); MTLCaptureManager* manager=[MTLCaptureManager sharedCaptureManager]; [manager stopCapture]; owned=false; return manager.isCapturing?Fail("Capture manager did not stop"):0;
}}
