// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
int main(){try{{MetalContext gpu;auto need=[](bool b){if(!b)throw std::runtime_error("KV hoist control guard failure");};
 const size_t batch=4,chunk=16,cap=256;std::vector<float> slots(batch*chunk),active(batch*chunk,1);
 for(size_t b=0;b<batch;++b)for(size_t t=0;t<chunk;++t)slots[b*chunk+t]=128+t;
 auto s=gpu.Upload(slots,batch,chunk),a=gpu.Upload(active,batch,chunk),z=gpu.Upload(std::vector<float>(batch*chunk,0),batch,chunk),pending=gpu.Allocate(batch,chunk);
 auto copy=s;need(s.SameView(copy)&&!s.SameView(a)&&!MetalMatrix().SameView(MetalMatrix()));
 need(gpu.LastChunkKVHoistedCommits()==0);
#ifdef GARNET_METAL_CHUNK_KV_COMMIT_HOIST
 need(gpu.CanHoistKVControls(s,a,batch,chunk,cap));
 active[3]=0;gpu.Update(a,active);need(!gpu.CanHoistKVControls(s,a,batch,chunk,cap));active[3]=1;
 slots[5]=slots[4];gpu.Update(s,slots);gpu.Update(a,active);need(!gpu.CanHoistKVControls(s,a,batch,chunk,cap));slots[5]=133;gpu.Update(s,slots);
 active[0]=NAN;gpu.Update(a,active);need(!gpu.CanHoistKVControls(s,a,batch,chunk,cap));active[0]=1;gpu.Update(a,active);
 need(!gpu.CanHoistKVControls(pending,a,batch,chunk,cap));need(!gpu.CanHoistKVControls(s,a,batch,chunk+1,cap));
 gpu.Begin();gpu.EncodeAdd(s,z,pending);need(!gpu.CanHoistKVControls(pending,a,batch,chunk,cap));gpu.Abort();need(gpu.LastChunkKVHoistedCommits()==0);
 for(size_t t=0;t<chunk;++t)active[3*chunk+t]=0;gpu.Update(a,active);need(gpu.CanHoistKVControls(s,a,batch,chunk,cap));
#endif
 std::cout<<"Native KV hoist ready/pending/gap/overwrite/nonfinite/empty-owner/shape and exact-view guards pass\n";
 }auto owners=MetalContext::ProfileOwnerStats();if(owners.matrices||owners.affineWeights||owners.bf16Weights||owners.pipelineSets)throw std::runtime_error("KV hoist guard test retained native owners");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
