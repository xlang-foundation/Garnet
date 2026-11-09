// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../src/backends/metal/bounded_affine_residency.h"
#include "metal_context.h"
#include <string>
using namespace Garnet;
#include <iostream>
#include <functional>
#include <algorithm>
#include <cstring>
#include <array>
#include <mach/mach.h>
static void Need(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static uint64_t Compressed(){task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;Need(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&count)==KERN_SUCCESS,"task snapshot failed");return info.compressed;}
int main(){@autoreleasepool{try{
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];Need(device&&queue,"Metal unavailable");
 constexpr uint64_t budget=32ull*1024*1024, bytes=1024*1024;int rejected=0;
 auto reject=[&](const char* label,const std::function<void()>& f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}Need(threw,label);++rejected;};
 BoundedAffineResidencyPlan<true> plan;auto a=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(a&&b,"buffer allocation");
 std::fill_n((uint32_t*)a.contents,bytes/4,0x173c9a55u);
 reject("renew-unprepared",[&]{plan.Renew();});reject("nil-device",[&]{plan.Prepare(nil,queue,{a},budget);});reject("nil-queue",[&]{plan.Prepare(device,nil,{a},budget);});reject("no-buffers",[&]{plan.Prepare(device,queue,{},budget);});reject("nil-buffer",[&]{plan.Prepare(device,queue,{a,nil},budget);});reject("zero-budget",[&]{plan.Prepare(device,queue,{a},0);});reject("payload-budget",[&]{plan.Prepare(device,queue,{a,b},a.allocatedSize);});reject("system-envelope",[&]{plan.Prepare(device,queue,{a},device.recommendedMaxWorkingSetSize);});Need(!plan.Ready(),"rejected prepare published state");
 auto hd=[MTLHeapDescriptor new];hd.size=8*bytes;hd.storageMode=MTLStorageModeShared;auto heap=[device newHeapWithDescriptor:hd];Need(heap!=nil,"heap allocation");auto hb=[heap newBufferWithLength:bytes options:MTLResourceStorageModeShared];Need(hb!=nil,"heap buffer allocation");reject("unaccounted-heap",[&]{plan.Prepare(device,queue,{hb},budget);});
 reject("creation-fault",[&]{plan.Prepare(device,queue,{a},budget,1);});reject("commit-fault",[&]{plan.Prepare(device,queue,{a},budget,2);});Need(!plan.Ready(),"failed creation/commit published state");
 uint64_t charged=0,payload=0,compressedBefore=Compressed(),compressedAfter=0;size_t renewals=0;
 plan.Prepare(device,queue,{a,a,b,a},budget);Need(plan.Ready()&&plan.Count()==2&&plan.PayloadBytes()==a.allocatedSize+b.allocatedSize&&plan.ChargedBytes()<=budget,"alias deduplication/budget");charged=plan.ChargedBytes();payload=plan.PayloadBytes();
 reject("double-prepare",[&]{plan.Prepare(device,queue,{a},budget);});Need(plan.Count()==2&&plan.PayloadBytes()==payload,"failed reprepare changed plan");
 auto copy=[queue commandBuffer];auto blit=[copy blitCommandEncoder];[blit copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:bytes];[blit endEncoding];[copy commit];
 plan.Renew();++renewals;plan.Close();Need(copy.status==MTLCommandBufferStatusCompleted,"close did not drain submitted work");Need(!std::memcmp(a.contents,b.contents,bytes),"residency changed GPU copy");Need(!plan.Ready()&&plan.Count()==0&&plan.ChargedBytes()==0&&plan.PayloadBytes()==0,"close retained plan state");plan.Close();
 // The set must release its retained resources independently of queue lifetime.
 __weak id<MTLBuffer> weak=nil;
 @autoreleasepool{auto owned=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];weak=owned;{BoundedAffineResidency raii;raii.Prepare(device,queue,{owned,owned},budget);owned=nil;Need(weak!=nil,"set did not retain resource");raii.Renew();++renewals;} }
 Need(weak==nil,"destroyed set/queue registration retained resource");
 // Reuse after close must build fresh state and retain no prior allocations.
 for(int i=0;i<16;i++){plan.Prepare(device,queue,{a},budget);plan.Renew();++renewals;Need(plan.Count()==1,"stale allocations across reprepare");plan.Close();}
 {
  MetalContext gpu;std::vector<uint32_t> words(16*64/8,0x76543210);std::vector<uint16_t> scales(16,0x3c00),biases(16,0xb000);
  auto weight=gpu.UploadAffineInt4(words.data(),words.size()*4,scales.data(),scales.size()*2,biases.data(),biases.size()*2,16,64,true);
  auto x=gpu.Upload(std::vector<float>(64,.125f),1,64),y=gpu.Allocate(1,16);
  Need(gpu.AffineResidencyStatsJson().find("\"enabled\":false")!=std::string::npos,"residency enabled by default");
  reject("native-empty-weight",[&]{gpu.PrepareAffineResidency({MetalAffineInt4Weight()},budget);});
  reject("native-empty-list",[&]{gpu.PrepareAffineResidency({},budget);});
  reject("native-low-budget",[&]{gpu.PrepareAffineResidency({weight},1);});
  gpu.Begin();reject("native-open-command",[&]{gpu.PrepareAffineResidency({weight},budget);});gpu.EncodeAffineInt4MatMul(x,weight,y);gpu.SubmitAndWait();auto before=gpu.Download(y);
  gpu.PrepareAffineResidency({weight,weight},budget);auto stats=gpu.AffineResidencyStatsJson();Need(stats.find("\"allocation_count\":3")!=std::string::npos,"native weight aliases not deduplicated");
  reject("native-reprepare",[&]{gpu.PrepareAffineResidency({weight},budget);});
  for(int i=0;i<3;i++){gpu.Begin();gpu.EncodeAffineInt4MatMul(x,weight,y);gpu.SubmitAndWait();auto after=gpu.Download(y);Need(after.size()==before.size()&&!std::memcmp(after.data(),before.data(),before.size()*sizeof(float)),"residency changed native output bits");}
  Need(gpu.AffineResidencyStatsJson().find("\"renewals\":4")!=std::string::npos,"native Begin did not renew residency");
 }
 compressedAfter=Compressed();
 std::cout<<"{\"passed\":true,\"scope\":\"Public Metal residency API/lifetime only; no model, paging or speed qualification\",\"unique_buffers\":2,\"payload_bytes\":"<<payload<<",\"charged_bytes\":"<<charged<<",\"budget_bytes\":"<<budget<<",\"rejected_invalid_operations\":"<<rejected<<",\"renewals\":"<<renewals<<",\"reprepare_cycles\":16,\"gpu_copy_bits_equal\":true,\"close_drained_submitted_work\":true,\"weak_resource_released\":true,\"closed_count\":"<<plan.Count()<<",\"compressed_before_snapshot_bytes\":"<<compressedBefore<<",\"compressed_after_snapshot_bytes\":"<<compressedAfter<<"}\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
