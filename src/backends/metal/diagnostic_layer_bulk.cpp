// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#include "lowering_context.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <iomanip>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* BulkGraph(X::Value value,X3PackageHost* host){if(value.runtime()!=host->runtime)throw std::invalid_argument("Bulk graph runtime mismatch");auto g=value.NativeData<MetalCompiledGraph>();if(!g)throw std::invalid_argument("Bulk needs compiled graphs");return g;}
X::Value CopyInput(X3PackageHost* host,X::Value value,const std::vector<int64_t>& shape){X::Tensor t(value);auto lease=t.Acquire();auto info=t.Info();size_t count=1;for(auto n:shape)count*=size_t(n);if(info.dtype!=X3_TENSOR_FLOAT32||info.symbolic||info.device_type||!info.data||info.rank!=shape.size()||!std::equal(shape.begin(),shape.end(),info.shape)||info.byte_size!=count*4||!IsContiguousTensor(t))throw std::invalid_argument("Invalid bulk host input profile");return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,info.data,count*4);}
}
bool MetalDecoderSession::PrepareDiagnosticLayerBulk(X::Value graphs,int64_t chunks) {
 std::lock_guard<std::mutex> lock(mutex_);
 if(!gpu_||poisoned_||diagnosticLayerBulk_||chunks<1||chunks>16||states_.size()!=32||states_[0].rows!=4||!head_||!head_->ready_||graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()!=32)throw std::invalid_argument("Prepare one bounded B4/32-layer bulk diagnostic");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
 if(managedRows_)throw std::logic_error("Bulk diagnostic does not support leased rows");
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
 if(mtp_)throw std::logic_error("Bulk diagnostic does not support MTP");
#endif
 std::vector<MetalCompiledGraph*> locked{head_};for(size_t i=0;i<32;++i){locked.push_back(BulkGraph(graphs.Get(i),Host()));locked.push_back(BulkGraph(owners_[i],Host()));}
 std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());locked.erase(std::unique(locked.begin(),locked.end()),locked.end());std::vector<std::unique_lock<std::mutex>> locks;for(auto g:locked)locks.emplace_back(g->mutex_);
 auto plan=std::make_unique<DiagnosticLayerBulk>();plan->graphs=graphs;plan->chunks=size_t(chunks);const size_t rows=512*size_t(chunks);
 for(size_t i=0;i<32;++i){auto g=BulkGraph(graphs.Get(i),Host()),owner=BulkGraph(owners_[i],Host());
  if(!g->ready_||!owner->ready_||g->verificationTokens_||!g->SharesVerificationWeights(*owner)||g->inputs_.size()!=6||g->inputShapes_[0]!=std::vector<int64_t>{4,128,4096}||g->inputShapes_[1]!=states_[i].shape[0]||g->inputShapes_[2]!=states_[i].shape[1]||!g->CanUpdateKVInPlace(1)||!g->CanUpdateKVInPlace(2))throw std::invalid_argument("Bulk layer owner/cache profile differs");
  DiagnosticBulkLayer layer;layer.graph=g;layer.generation=g->generation_;layer.ffn=g->DescribeDiagnosticFfn();layer.kv=g->DescribeDiagnosticKV();layer.qo=g->DescribeDiagnosticQO(layer.kv,layer.ffn);std::set<size_t> controls;for(size_t j=3;j<6;++j)controls.insert(g->inputs_[j]);for(const auto& v:g->controlSources_)controls.insert(v.first);layer.controlIds.assign(controls.begin(),controls.end());
  plan->layers.push_back(std::move(layer));g->RetireResidentDefaults(i==0);PrepareHiddenPool(*g);
 }
 uint64_t estimate=uint64_t(rows)*(4*4096+2*14336+2*1024)*4+352321536ull+4*128*4096*4+4*4096*4;
 for(const auto& layer:plan->layers)for(auto id:layer.controlIds)estimate+=uint64_t(chunks)*layer.graph->slots_[id].rows*layer.graph->slots_[id].columns*4;
 auto mem=MetalContext::MemoryStats();if(!mem.recommended||estimate>mem.recommended||mem.allocated>mem.recommended-estimate)throw std::runtime_error("Bulk diagnostic workspace exceeds recommended device budget");
 auto owned=[&](size_t n,size_t c){auto m=gpu_->Allocate(n,c);gpu_->Zero(m);return m;};
 for(auto& tape:plan->tape)tape=owned(rows,4096);plan->residual=owned(rows,4096);plan->norm=owned(rows,4096);plan->gate=owned(rows,14336);plan->up=owned(rows,14336);plan->down=plan->norm;plan->hidden=owned(4,128*4096);plan->lastHidden=owned(4,4096);plan->key=owned(rows,1024);plan->value=owned(rows,1024);
 for(size_t c=0;c<size_t(chunks);++c){for(size_t p=0;p<2;++p)plan->tapeViews[p].push_back(gpu_->PrepareDiagnosticDenseView(plan->tape[p],c*512*4096,4,128*4096));plan->residualViews.push_back(gpu_->PrepareDiagnosticDenseView(plan->residual,c*512*4096,512,4096));plan->keyViews.push_back(gpu_->PrepareDiagnosticDenseView(plan->key,c*512*1024,512,1024));plan->valueViews.push_back(gpu_->PrepareDiagnosticDenseView(plan->value,c*512*1024,512,1024));plan->normViews.push_back(gpu_->PrepareDiagnosticDenseView(plan->norm,c*512*4096,512,4096));}
 for(auto& layer:plan->layers){auto g=layer.graph;layer.controls.resize(size_t(chunks));for(auto& bank:layer.controls)for(auto id:layer.controlIds)bank.push_back(owned(g->slots_[id].rows,g->slots_[id].columns));
  const auto& f=layer.ffn;if(!gpu_->PrepareDiagnosticBulkAffine(plan->norm,g->slots_[f.gateWeight].affineInt4,plan->gate)||!gpu_->PrepareDiagnosticBulkAffine(plan->norm,g->slots_[f.upWeight].affineInt4,plan->up)||!gpu_->PrepareDiagnosticBulkAffine(plan->gate,g->slots_[f.downWeight].affineInt4,plan->down))throw std::logic_error("Cannot prepare bulk FFN products");
 }
 // Precompile every actual chunk count up to the profile bound. Views use
 // only the live prefix of each owned root, without maximum-length padding.
 for(size_t count=1;count<=size_t(chunks);++count){DiagnosticBulkGroup group;size_t n=count*512;
  auto view=[&](const MetalMatrix& m,size_t c){return gpu_->PrepareDiagnosticDenseView(m,0,n,c);};
  group.residual=view(plan->residual,4096);group.norm=view(plan->norm,4096);group.gate=view(plan->gate,14336);group.up=view(plan->up,14336);group.down=group.norm;group.key=view(plan->key,1024);group.value=view(plan->value,1024);for(size_t k=0;k<2;++k)group.tape[k]=view(plan->tape[k],4096);
  auto g=plan->layers[0].graph;const auto& f=plan->layers[0].ffn;
  if(!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[f.gateWeight].affineInt4,group.gate)||!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[f.upWeight].affineInt4,group.up)||!gpu_->PrepareDiagnosticBulkAffine(group.gate,g->slots_[f.downWeight].affineInt4,group.down))throw std::logic_error("Cannot prepare bounded live bulk group");
  const auto& kv=plan->layers[0].kv;
  if(!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[kv.keyWeight].affineInt4,group.key)||!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[kv.valueWeight].affineInt4,group.value))throw std::logic_error("Cannot prepare bounded K/V group");
  const auto& qo=plan->layers[0].qo;
  if(!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[qo.queryWeight].affineInt4,group.residual)||!gpu_->PrepareDiagnosticBulkAffine(group.norm,g->slots_[qo.outputWeight].affineInt4,group.residual))throw std::logic_error("Cannot prepare bounded Q/O group");
  plan->groups.push_back(std::move(group));
 }
 std::set<uintptr_t> identities;auto charge=[&](const MetalMatrix& v){auto b=MetalContext::MatrixBufferInfo(v);if(b.identity&&identities.insert(b.identity).second)plan->workspaceBytes+=b.lengthBytes;};
 for(auto t:plan->tape)charge(t);for(auto t:{plan->residual,plan->norm,plan->gate,plan->up,plan->silu,plan->down,plan->hidden,plan->lastHidden,plan->key,plan->value})charge(t);for(auto t:gpu_->DiagnosticBulkScratch())charge(t);for(const auto& layer:plan->layers)for(const auto& bank:layer.controls)for(auto t:bank)charge(t);
 if(plan->workspaceBytes!=estimate)throw std::logic_error("Bulk workspace inventory differs from preflight");diagnosticLayerBulk_=std::move(plan);return true;
}
bool MetalDecoderSession::SetDiagnosticBulkKV(bool enabled) {
 std::lock_guard<std::mutex> lock(mutex_);if(!gpu_||poisoned_||!diagnosticLayerBulk_)return false;
#ifdef GARNET_METAL_CONTINUOUS_ROWS
 if(requestRows_)for(size_t row=0;row<4;++row)if(requestRows_->Owned(row))return false;
#endif
 if(!enabled&&diagnosticBulkQOEnabled_)return false;diagnosticBulkKVEnabled_=enabled;return true;
}
bool MetalDecoderSession::SetDiagnosticBulkQO(bool enabled) {
 std::lock_guard<std::mutex> lock(mutex_);if(!gpu_||poisoned_||!diagnosticLayerBulk_||(enabled&&!diagnosticBulkKVEnabled_))return false;
#ifdef GARNET_METAL_CONTINUOUS_ROWS
 if(requestRows_)for(size_t row=0;row<4;++row)if(requestRows_->Owned(row))return false;
#endif
 diagnosticBulkQOEnabled_=enabled;return true;
}
X::Value MetalDecoderSession::RunDiagnosticLayerBulk(X::Value hiddenChunks,X::Value controls,bool enabled) {
 std::lock_guard<std::mutex> lock(mutex_);auto* p=diagnosticLayerBulk_.get();if(!p||!gpu_||poisoned_||!hiddenChunks.IsList()||!controls.IsList()||hiddenChunks.Size()!=p->chunks||controls.Size()!=p->chunks)throw std::invalid_argument("Bulk input sequence differs from prepared profile");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
 if(managedRows_)throw std::logic_error("Bulk diagnostic refuses leased request rows");
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
 if(tokenCache_)throw std::logic_error("Bulk diagnostic refuses prefix-cache state");
#endif
 std::vector<MetalCompiledGraph*> locked{head_};for(size_t i=0;i<32;++i){locked.push_back(p->layers[i].graph);locked.push_back(BulkGraph(owners_[i],Host()));}std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());locked.erase(std::unique(locked.begin(),locked.end()),locked.end());std::vector<std::unique_lock<std::mutex>> locks;for(auto g:locked)locks.emplace_back(g->mutex_);
 if(!head_->ready_||head_->generation_!=headGeneration_)throw std::logic_error("Bulk head changed");for(size_t i=0;i<32;++i)if(!p->layers[i].graph->ready_||p->layers[i].graph->generation_!=p->layers[i].generation||!p->layers[i].graph->SharesVerificationWeights(*BulkGraph(owners_[i],Host())))throw std::logic_error("Prepared bulk graph changed");
 std::vector<X::Value> h,cs;for(size_t c=0;c<p->chunks;++c){h.push_back(CopyInput(Host(),hiddenChunks.Get(c),{4,128,4096}));auto entry=controls.Get(c);if(!entry.IsList()||entry.Size()!=3)throw std::invalid_argument("Bulk control arity differs");auto copied=X::Value::List(Host());for(size_t v=0;v<3;++v)copied.Append(CopyInput(Host(),entry.Get(v),v?std::vector<int64_t>{4,128}:std::vector<int64_t>{4,128,3}));
  X::Tensor position(copied.Get(uint64_t(0))),slot(copied.Get(1)),active(copied.Get(2));auto pl=position.Acquire(),sl=slot.Acquire(),al=active.Acquire();auto pd=static_cast<const float*>(position.Info().data),sd=static_cast<const float*>(slot.Info().data),ad=static_cast<const float*>(active.Info().data);
  for(size_t b=0;b<4;++b)for(size_t t=0;t<128;++t){size_t at=b*128+t;float expected=float(c*128+t);if(ad[at]!=1||sd[at]!=expected||pd[at*3]!=expected||pd[at*3+1]!=expected||pd[at*3+2]!=expected)throw std::invalid_argument("Bulk diagnostic requires dense contiguous fresh text prefixes");}cs.push_back(copied);
 }
 if(diagnosticBulkKVEnabled_||diagnosticBulkQOEnabled_)throw std::logic_error("Bulk K/V and Q/O require the leased tracked route");
 // Complete validation precedes reset. This isolated diagnostic owns the fresh
 // unmanaged session; it never resets an active serving/request/cache context.
 ResetUnlocked();hidden_=p->hidden;hiddenShape_={4,128,4096};rowFinalHidden_=p->lastHidden;rowFinalWidth_=4096;finalHiddenReady_.assign(4,0);gpu_->Zero(rowFinalHidden_);
 double totalGpu=0;size_t packedCalls=0,bulkCalls=0,halfCalls=0;auto start=std::chrono::steady_clock::now();X::Value result=X::Value::List(Host());
 if(!enabled){locks.clear();for(size_t c=0;c<p->chunks;++c){result=RunUnlocked(p->graphs,h[c],cs[c].Get(uint64_t(0)),cs[c].Get(1),cs[c].Get(2),c+1==p->chunks?127:-1);totalGpu+=gpu_->LastGPUMs();packedCalls+=gpu_->LastPackedAffineDispatches();} }
 else {
  std::vector<MetalPrefillScratch*> scratch;for(auto g:locked)if(g->prefillScratch_)scratch.push_back(g->prefillScratch_.get());std::sort(scratch.begin(),scratch.end(),std::less<MetalPrefillScratch*>());scratch.erase(std::unique(scratch.begin(),scratch.end()),scratch.end());std::vector<std::unique_lock<std::mutex>> scratchLocks;for(auto s:scratch)scratchLocks.emplace_back(s->mutex);
  for(size_t c=0;c<p->chunks;++c){X::Tensor input(h[c]);auto lease=input.Acquire();gpu_->Update(p->tapeViews[0][c],static_cast<const float*>(input.Info().data),4*128*4096);}
  poisoned_=true;
  try {for(size_t layerIndex=0;layerIndex<32;++layerIndex){auto& layer=p->layers[layerIndex];auto g=layer.graph;const auto& f=layer.ffn;
    // Snapshot each graph's complete validated control/derived-control bank
    // before opening the session command. Per-chunk buffers never get
    // overwritten on the CPU while their GPU consumers remain pending.
    for(size_t c=0;c<p->chunks;++c){auto args=X::Value::List(Host());args.Append(X::Value());args.Append(X::Value());args.Append(X::Value());for(size_t v=0;v<3;++v)args.Append(cs[c].Get(v));g->PrepareInputs(args,{0,1,2});for(size_t v=0;v<layer.controlIds.size();++v)gpu_->Update(layer.controls[c][v],g->gpu_->Download(g->slots_[layer.controlIds[v]].matrix));}
    gpu_->Begin();for(size_t c=0;c<p->chunks;++c){std::map<size_t,MetalMatrix> saved;auto bind=[&](size_t id,const MetalMatrix& matrix){saved.emplace(id,g->slots_[id].matrix);g->slots_[id].matrix=matrix;};bind(g->inputs_[0],p->tapeViews[layerIndex%2][c]);for(size_t k=0;k<2;++k){bind(g->inputs_[k+1],states_[layerIndex].matrix[k]);for(auto id:g->outputStorageGroups_[k+1])bind(id,states_[layerIndex].matrix[k]);}for(size_t v=0;v<layer.controlIds.size();++v)bind(layer.controlIds[v],layer.controls[c][v]);
      auto restore=[&](){for(auto& v:saved)g->slots_[v.first].matrix=v.second;};try {g->EncodeDiagnosticPrefix(*gpu_,f.begin);auto residual=g->slots_[f.residual].matrix;gpu_->EncodeLayout(residual,residual,p->residualViews[c],0);}catch(...){restore();throw;}restore();}
    gpu_->EncodeRMSNorm(p->residual,g->slots_[f.normWeight].matrix,p->norm,f.epsilon,f.offset);gpu_->EncodeDiagnosticBulkAffine(p->norm,g->slots_[f.gateWeight].affineInt4,p->gate);gpu_->EncodeDiagnosticBulkAffine(p->norm,g->slots_[f.upWeight].affineInt4,p->up);gpu_->EncodeDiagnosticBulkGatedDown(p->gate,p->up,g->slots_[f.downWeight].affineInt4,p->down);gpu_->EncodeAdd(p->residual,p->down,p->tape[(layerIndex+1)%2]);gpu_->SubmitAndWait();totalGpu+=gpu_->LastGPUMs();packedCalls+=gpu_->LastPackedAffineDispatches();bulkCalls+=gpu_->LastDiagnosticBulkDispatches();halfCalls+=gpu_->LastDiagnosticSiluHalfDispatches();
   }
   auto last=p->tapeViews[0].back();gpu_->Begin();gpu_->EncodeLayout(last,last,hidden_,0);gpu_->EncodeLastActiveHidden(hidden_,p->layers[0].controls.back().at(std::find(p->layers[0].controlIds.begin(),p->layers[0].controlIds.end(),p->layers[0].graph->inputs_[5])-p->layers[0].controlIds.begin()),rowFinalHidden_,128);auto& headInput=head_->slots_[head_->inputs_[0]];gpu_->EncodeSliceGroups(hidden_,headInput.matrix,1,128*4096,127*4096,4096);head_->EncodeGreedy(*gpu_,prediction_);gpu_->SubmitAndWait();totalGpu+=gpu_->LastGPUMs();auto ids=gpu_->Download(prediction_);for(size_t row=0;row<4;++row){if(ids[row*2+1]!=0)throw std::runtime_error("Nonfinite bulk logits");result.Append(X::Value(int64_t(ids[row*2])));}std::fill(finalHiddenReady_.begin(),finalHiddenReady_.end(),1);poisoned_=false;
  }catch(...){gpu_->Abort();throw;}
 }
 auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();gpuMs_=totalGpu;inferenceMs_=wall;std::ostringstream stats;stats<<std::setprecision(17);stats<<"{\"enabled\":"<<(enabled?"true":"false")<<",\"chunks\":"<<p->chunks<<",\"layers\":32,\"rows\":"<<p->chunks*512<<",\"workspace_bytes\":"<<p->workspaceBytes<<",\"gpu_ms\":"<<totalGpu<<",\"body_wall_ms\":"<<wall<<",\"packed_calls\":"<<packedCalls<<",\"bulk_calls\":"<<bulkCalls<<",\"silu_half_calls\":"<<halfCalls<<"}";diagnosticLayerBulkStats_=stats.str();return result;
}
}
