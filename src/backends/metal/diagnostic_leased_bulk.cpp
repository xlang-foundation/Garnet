// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#include "lowering_context.h"
#include "bulk_append_policy.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* LeasedBulkGraph(X::Value value,X3PackageHost* host){if(value.runtime()!=host->runtime)throw std::invalid_argument("Leased bulk graph runtime differs");auto g=value.NativeData<MetalCompiledGraph>();if(!g)throw std::invalid_argument("Leased bulk requires compiled graphs");return g;}
X::Value LeasedBulkCopy(X3PackageHost* host,X::Value value,const std::vector<int64_t>& shape){if(value.runtime()!=host->runtime)throw std::invalid_argument("Leased bulk input runtime differs");X::Tensor t(value);auto lease=t.Acquire();auto info=t.Info();size_t count=1;for(auto n:shape)count*=size_t(n);if(info.dtype!=X3_TENSOR_FLOAT32||info.symbolic||info.device_type||!info.data||info.rank!=shape.size()||!std::equal(shape.begin(),shape.end(),info.shape)||info.byte_size!=count*4||!IsContiguousTensor(t))throw std::invalid_argument("Leased bulk host input profile differs");return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,info.data,count*4);}
}
X::Value MetalDecoderSession::RunDiagnosticLeasedBulk(X::Value hiddenChunks,X::Value controls,X::Value epochs,X::Value tokenChunks,bool enabled) {
 std::lock_guard<std::mutex> lock(mutex_);auto* p=diagnosticLayerBulk_.get();
 if(!p||!gpu_||poisoned_||!requestRows_||!managedRows_||!tokenCache_||states_.size()!=32||states_[0].rows!=4||rowTokens_.size()!=4||rowTokensValid_.size()!=4||hiddenChunks.runtime()!=Host()->runtime||controls.runtime()!=Host()->runtime||epochs.runtime()!=Host()->runtime||tokenChunks.runtime()!=Host()->runtime||!hiddenChunks.IsList()||!controls.IsList()||!epochs.IsList()||!tokenChunks.IsList()||epochs.Size()!=4||!hiddenChunks.Size()||hiddenChunks.Size()>p->chunks||controls.Size()!=hiddenChunks.Size()||tokenChunks.Size()!=hiddenChunks.Size())throw std::invalid_argument("Leased bulk requires bounded native tracked request rows");
#ifdef GARNET_METAL_MTP_BRIDGE
 if(mtp_)throw std::logic_error("Leased bulk MTP capture is not supported");
#endif
 const size_t chunks=size_t(hiddenChunks.Size());const auto& group=p->groups.at(chunks-1);
 std::array<uint64_t,4> expectedArray{};std::vector<uint64_t> expected;
 std::array<size_t,4> starts{};std::array<bool,4> valid{};
 for(size_t row=0;row<4;++row){auto value=epochs.Get(row);if(!value.IsInt64()||value.ToInt64()<0)throw std::invalid_argument("Leased bulk epochs must be nonnegative integers");auto e=uint64_t(value.ToInt64());if(e&&!requestRows_->Matches(row,e))throw std::invalid_argument("Stale leased bulk epoch");expectedArray[row]=e;expected.push_back(e);starts[row]=rowTokens_[row].size();valid[row]=rowTokensValid_[row]!=0;
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
 tokenPrecisionHistory_.at(row).Validate(starts[row]);
#endif
}
 BulkAppendValidator validator(starts,valid,expectedArray,tokenCapacity_,tokenVocabulary_);
 std::vector<X::Value> h,cs;std::vector<std::vector<int64_t>> ids;std::array<std::vector<int64_t>,4> appended;bool lastChunkActive=false;
 for(size_t c=0;c<chunks;++c){h.push_back(LeasedBulkCopy(Host(),hiddenChunks.Get(c),{4,128,4096}));auto entry=controls.Get(c),tokens=tokenChunks.Get(c);if(entry.runtime()!=Host()->runtime||!entry.IsList()||entry.Size()!=3||tokens.runtime()!=Host()->runtime||!tokens.IsList()||tokens.Size()!=512)throw std::invalid_argument("Leased bulk control/token arity differs");auto copied=X::Value::List(Host());for(size_t v=0;v<3;++v)copied.Append(LeasedBulkCopy(Host(),entry.Get(v),v?std::vector<int64_t>{4,128}:std::vector<int64_t>{4,128,3}));
  X::Tensor position(copied.Get(uint64_t(0))),slot(copied.Get(1)),active(copied.Get(2));auto pl=position.Acquire(),sl=slot.Acquire(),al=active.Acquire();auto pd=static_cast<const float*>(position.Info().data),sd=static_cast<const float*>(slot.Info().data),ad=static_cast<const float*>(active.Info().data);std::vector<int64_t> chunkIds;chunkIds.reserve(512);lastChunkActive=false;
  for(size_t row=0;row<4;++row)for(size_t t=0;t<128;++t){size_t at=row*128+t;auto token=tokens.Get(at);if(!token.IsInt64())throw std::invalid_argument("Leased bulk token must be integer");auto id=token.ToInt64();validator.Step(row,ad[at],sd[at],pd+at*3,id);chunkIds.push_back(id);if(ad[at]!=0){appended[row].push_back(id);lastChunkActive=true;}}
  cs.push_back(copied);ids.push_back(std::move(chunkIds));
 }
 validator.Finish(lastChunkActive);const auto counts=validator.Counts();const bool bulkKV=enabled&&diagnosticBulkKVEnabled_&&std::all_of(counts.begin(),counts.end(),[&](size_t n){return n==chunks*128;});const bool bulkQO=bulkKV&&diagnosticBulkQOEnabled_;auto selectedEpochs=X::Value::List(Host());
 for(size_t row=0;row<4;++row){selectedEpochs.Append(X::Value(counts[row]?int64_t(expected[row]):int64_t(0)));if(counts[row])rowTokens_[row].reserve(starts[row]+counts[row]);}
 // Lock immutable graph/weight identities through candidate execution. The
 // control reuses RunUnlocked's own graph locks under this same session lock.
 std::vector<MetalCompiledGraph*> locked{head_};for(size_t i=0;i<32;++i){locked.push_back(p->layers[i].graph);locked.push_back(LeasedBulkGraph(owners_[i],Host()));}std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());locked.erase(std::unique(locked.begin(),locked.end()),locked.end());std::vector<std::unique_lock<std::mutex>> locks;for(auto g:locked)locks.emplace_back(g->mutex_);
 if(!head_||!head_->ready_||head_->generation_!=headGeneration_)throw std::logic_error("Leased bulk head changed");for(size_t i=0;i<32;++i){auto& layer=p->layers[i];auto owner=LeasedBulkGraph(owners_[i],Host());if(!layer.graph->ready_||layer.graph->generation_!=layer.generation||!owner->ready_||owner->generation_!=states_[i].tokenOwnerGeneration||!layer.graph->SharesVerificationWeights(*owner))throw std::logic_error("Leased bulk graph/weight owner changed");}
 if(rowFinalHidden_.Valid()&&(rowFinalWidth_!=4096||finalHiddenReady_.size()!=4))throw std::logic_error("Leased bulk final hidden bank differs");
 // Every externally supplied value and all native lease/history/owner checks
 // above precede mutation. There is deliberately no session or row reset.
 if(!rowFinalHidden_.Valid()){rowFinalHidden_=p->lastHidden;rowFinalWidth_=4096;finalHiddenReady_.assign(4,0);}
 double totalGpu=0;size_t packedCalls=0,bulkCalls=0,halfCalls=0,inputOpsSkipped=0;auto start=std::chrono::steady_clock::now();
 try {
  if(!enabled){locks.clear();for(size_t c=0;c<chunks;++c){RunUnlocked(p->graphs,h[c],cs[c].Get(uint64_t(0)),cs[c].Get(1),cs[c].Get(2),-1,&expected,&ids[c]);totalGpu+=gpu_->LastGPUMs();packedCalls+=gpu_->LastPackedAffineDispatches();}}
  else {
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
   for(auto& state:states_)for(size_t k=0;k<2;++k)state.zeroOutsideDirty[k]=false;
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
   prefixFreshReset_=false;
#endif
   std::vector<MetalPrefillScratch*> scratch;for(auto g:locked)if(g->prefillScratch_)scratch.push_back(g->prefillScratch_.get());std::sort(scratch.begin(),scratch.end(),std::less<MetalPrefillScratch*>());scratch.erase(std::unique(scratch.begin(),scratch.end()),scratch.end());std::vector<std::unique_lock<std::mutex>> scratchLocks;for(auto v:scratch)scratchLocks.emplace_back(v->mutex);
   for(size_t c=0;c<chunks;++c){X::Tensor input(h[c]);auto lease=input.Acquire();gpu_->Update(p->tapeViews[0][c],static_cast<const float*>(input.Info().data),4*128*4096);}
   hidden_=p->hidden;hiddenShape_={4,128,4096};poisoned_=true;
   for(size_t layerIndex=0;layerIndex<32;++layerIndex){auto& layer=p->layers[layerIndex];auto g=layer.graph;const auto& f=layer.ffn;
    for(size_t c=0;c<chunks;++c){auto args=X::Value::List(Host());args.Append(X::Value());args.Append(X::Value());args.Append(X::Value());for(size_t v=0;v<3;++v)args.Append(cs[c].Get(v));g->PrepareInputs(args,{0,1,2});for(size_t v=0;v<layer.controlIds.size();++v)gpu_->Update(layer.controls[c][v],g->gpu_->Download(g->slots_[layer.controlIds[v]].matrix));}
    gpu_->Begin();
    // The complete actual group is dense and active. Future K/V stay private;
    // the original per-chunk rope, attention and cache commits still run.
    if(bulkKV){const auto& kv=layer.kv;gpu_->EncodeRMSNorm(group.tape[layerIndex%2],g->slots_[kv.normWeight].matrix,group.norm,kv.epsilon,kv.offset);gpu_->EncodeDiagnosticBulkAffine(group.norm,g->slots_[kv.keyWeight].affineInt4,group.key);gpu_->EncodeDiagnosticBulkAffine(group.norm,g->slots_[kv.valueWeight].affineInt4,group.value);if(bulkQO)gpu_->EncodeDiagnosticBulkAffine(group.norm,g->slots_[layer.qo.queryWeight].affineInt4,group.residual);}
    for(size_t c=0;c<chunks;++c){std::map<size_t,MetalMatrix> saved;auto bind=[&](size_t id,const MetalMatrix& m){saved.emplace(id,g->slots_[id].matrix);g->slots_[id].matrix=m;};bind(g->inputs_[0],p->tapeViews[layerIndex%2][c]);for(size_t k=0;k<2;++k){bind(g->inputs_[k+1],states_[layerIndex].matrix[k]);for(auto id:g->outputStorageGroups_[k+1])bind(id,states_[layerIndex].matrix[k]);}for(size_t v=0;v<layer.controlIds.size();++v)bind(layer.controlIds[v],layer.controls[c][v]);if(bulkKV){bind(layer.kv.keyOut,p->keyViews[c]);bind(layer.kv.valueOut,p->valueViews[c]);}if(bulkQO){bind(layer.qo.queryOut,p->residualViews[c]);inputOpsSkipped+=layer.qo.inputOps.size();}auto restore=[&](){for(auto& v:saved)g->slots_[v.first].matrix=v.second;};try{if(bulkQO){g->EncodeDiagnosticAttentionPrefix(*gpu_,layer.kv,layer.qo);auto context=g->slots_[layer.qo.context].matrix;gpu_->EncodeLayout(context,context,p->normViews[c],0);}else{g->EncodeDiagnosticPrefix(*gpu_,f.begin,bulkKV?&layer.kv:nullptr);auto residual=g->slots_[f.residual].matrix;gpu_->EncodeLayout(residual,residual,p->residualViews[c],0);}}catch(...){restore();throw;}restore();}
    if(bulkQO){gpu_->EncodeDiagnosticBulkAffine(group.norm,g->slots_[layer.qo.outputWeight].affineInt4,group.residual);gpu_->EncodeAdd(group.tape[layerIndex%2],group.residual,group.norm);}
    // Add keeps its distinct-output contract. Once O consumed all contexts,
    // norm becomes the complete residual, and the dead Q/O root becomes FFN
    // norm/down storage. This rotates roles without another GPU allocation.
    const auto& ffnResidual=bulkQO?group.norm:group.residual;const auto& ffnNorm=bulkQO?group.residual:group.norm;
    gpu_->EncodeRMSNorm(ffnResidual,g->slots_[f.normWeight].matrix,ffnNorm,f.epsilon,f.offset);gpu_->EncodeDiagnosticBulkAffine(ffnNorm,g->slots_[f.gateWeight].affineInt4,group.gate);gpu_->EncodeDiagnosticBulkAffine(ffnNorm,g->slots_[f.upWeight].affineInt4,group.up);gpu_->EncodeDiagnosticBulkGatedDown(group.gate,group.up,g->slots_[f.downWeight].affineInt4,ffnNorm);gpu_->EncodeAdd(ffnResidual,ffnNorm,group.tape[(layerIndex+1)%2]);gpu_->SubmitAndWait();totalGpu+=gpu_->LastGPUMs();packedCalls+=gpu_->LastPackedAffineDispatches();bulkCalls+=gpu_->LastDiagnosticBulkDispatches();halfCalls+=gpu_->LastDiagnosticSiluHalfDispatches();
   }
   // Each row can end in a different chunk. Preserve unselected bank rows and
   // select the last active token across the actual group, including ragged tails.
   auto& first=p->layers[0];auto found=std::find(first.controlIds.begin(),first.controlIds.end(),first.graph->inputs_[5]);if(found==first.controlIds.end())throw std::logic_error("Missing prepared active controls");const size_t activeIndex=size_t(found-first.controlIds.begin());
   for(size_t c=0;c<chunks;++c){gpu_->Begin();auto last=p->tapeViews[0][c];gpu_->EncodeLayout(last,last,p->hidden,0);gpu_->EncodeLastActiveHidden(p->hidden,first.controls[c].at(activeIndex),rowFinalHidden_,128);gpu_->SubmitAndWait();totalGpu+=gpu_->LastGPUMs();}
   for(size_t row=0;row<4;++row)if(counts[row]){
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    tokenPrecisionHistory_.at(row).Append(starts[row],counts[row],true);
#endif
    rowTokens_[row].insert(rowTokens_[row].end(),appended[row].begin(),appended[row].end());finalHiddenReady_[row]=1;}
   reconcileHiddenPending_=false;poisoned_=false;scratchLocks.clear();locks.clear();
  }
  auto result=PredictRowsUnlocked(selectedEpochs);totalGpu+=gpu_->LastGPUMs();gpuMs_=totalGpu;inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::ostringstream stats;stats<<std::setprecision(17)<<"{\"leased\":true,\"enabled\":"<<(enabled?"true":"false")<<",\"chunks\":"<<chunks<<",\"bound_chunks\":"<<p->chunks<<",\"rows\":"<<chunks*512<<",\"workspace_bytes\":"<<p->workspaceBytes<<",\"gpu_ms\":"<<totalGpu<<",\"body_wall_ms\":"<<inferenceMs_<<",\"packed_calls\":"<<packedCalls<<",\"bulk_calls\":"<<bulkCalls<<",\"kv_bulk_calls\":"<<(bulkKV?64:0)<<",\"kv_bulk_enabled\":"<<(diagnosticBulkKVEnabled_?"true":"false")<<",\"qo_bulk_enabled\":"<<(diagnosticBulkQOEnabled_?"true":"false")<<",\"qo_bulk_calls\":"<<(bulkQO?64:0)<<",\"qo_input_ops_skipped\":"<<inputOpsSkipped<<",\"silu_half_calls\":"<<halfCalls<<",\"appended\":[";for(size_t row=0;row<4;++row){if(row)stats<<',';stats<<counts[row];}stats<<"]}";diagnosticLayerBulkStats_=stats.str();return result;
 }catch(...){gpu_->Abort();poisoned_=true;throw;}
}
}
