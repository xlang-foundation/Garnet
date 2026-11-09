#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
size_t MetalContext::LastFusedMistralMlpBlocks() const {return impl_->fusedMlpBlocks;}
MetalMatrix MetalContext::FusedMistralMlpScratch() const {return impl_->fusedMlpHalf;}
bool MetalContext::TryEncodeFusedMistralMlp(const MetalMatrix& input,const MetalAffineInt4Weight& gate,
 const MetalAffineInt4Weight& up,const MetalAffineInt4Weight& down,const MetalMatrix& gateLayout,
 const MetalMatrix& upLayout,const MetalMatrix& activatedLayout,const MetalMatrix& output,
 const std::vector<size_t>* selectedRows) {
 @autoreleasepool {
  if(!impl_->packedEnabled)return false;
  if(!impl_->command||!input.impl_||!gate.impl_||!up.impl_||!down.impl_||!gateLayout.impl_||!upLayout.impl_||!activatedLayout.impl_||!output.impl_)
   throw std::invalid_argument("Fused MLP requires an open command and complete operands");
  auto a=input.impl_->matrix,o=output.impl_->matrix,g=gateLayout.impl_->matrix,u=upLayout.impl_->matrix,s=activatedLayout.impl_->matrix;
  const auto& gw=gate.impl_;const auto& uw=up.impl_;const auto& dw=down.impl_;
  for(auto matrix:{a,o,g,u,s})if(matrix.device!=impl_->device||matrix.dataType!=MPSDataTypeFloat32)throw std::invalid_argument("Fused MLP device/type mismatch");
  for(const auto& w:{gw,uw,dw})if(w->packed.device!=impl_->device||w->scales.device!=impl_->device||w->biases.device!=impl_->device)throw std::invalid_argument("Fused MLP weight device mismatch");
  if(!a.rows||a.rows>512||a.columns!=4096||o.rows!=a.rows||o.columns!=4096||
   g.rows!=a.rows||u.rows!=a.rows||s.rows!=a.rows||g.columns!=14336||u.columns!=14336||s.columns!=14336||
   gw->rows!=14336||gw->columns!=4096||uw->rows!=14336||uw->columns!=4096||dw->rows!=4096||dw->columns!=14336)return false;
  std::vector<uint32_t> indices;std::array<bool,512> seen{};
  if(selectedRows){if(selectedRows->size()>a.rows)throw std::invalid_argument("Fused MLP selection exceeds rows");for(auto row:*selectedRows){if(row>=a.rows||seen[row])throw std::invalid_argument("Fused MLP selection must be unique/in bounds");seen[row]=true;indices.push_back(uint32_t(row));}}
  else for(size_t row=0;row<a.rows;++row)indices.push_back(uint32_t(row));
  if(indices.size()<128)return false;
  auto gp=impl_->packedPlans.find({4096,14336,a.rowBytes/4,g.rowBytes/4});
  auto dp=impl_->packedPlans.find({14336,4096,s.rowBytes/4,o.rowBytes/4});
  if(gp==impl_->packedPlans.end()||dp==impl_->packedPlans.end()||!impl_->fusedMlpHalf.Valid())return false;
  if(a.rows>gp->second.maximumRows||a.rows>dp->second.maximumRows)throw std::logic_error("Fused MLP exceeds prepared upper bound");
  size_t gateTile=indices.size()>=256?256:128,downTile=indices.size()>=384?256:128;
  auto gatePipeline=gp->second.fused[gateTile==256],downPipeline=dp->second.pipelines[downTile==256];
  if(!gatePipeline||!downPipeline)return false;
  if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_->PendingKey())))throw std::logic_error("Fused MLP input not ready in current command");
  if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_->PendingKey()))throw std::logic_error("Fused MLP output belongs to another command");
  output.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(output.impl_->PendingKey()));
  auto compact=impl_->packedInput.impl_->matrix,half=impl_->fusedMlpHalf.impl_->matrix;
  for(const auto& scratch:{impl_->packedInput,impl_->fusedMlpHalf})if(scratch.impl_->pending&&!impl_->pendingOutputs.count(scratch.impl_->PendingKey()))throw std::logic_error("Fused MLP private scratch belongs to another command");
  impl_->fusedMlpHalf.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(impl_->fusedMlpHalf.impl_->PendingKey()));
  auto span=[](MPSMatrix* m){if(m.rowBytes<m.columns*4||m.offset%4||(m.rows-1)>(SIZE_MAX-m.columns*4)/m.rowBytes)throw std::invalid_argument("Invalid fused MLP span");size_t bytes=(m.rows-1)*m.rowBytes+m.columns*4;if(m.offset>m.data.length||bytes>m.data.length-m.offset)throw std::invalid_argument("Fused MLP span exceeds allocation");return bytes;};
  size_t outBytes=0;for(auto m:{a,o,g,u,s}){auto bytes=span(m);if(m==o)outBytes=bytes;if(m.data==compact.data||m.data==half.data)throw std::invalid_argument("Fused MLP operand aliases private scratch");}
  for(const auto& w:{gw,uw,dw})for(auto m:{a,o,g,u,s})if(m.data==w->packed||m.data==w->scales||m.data==w->biases)throw std::invalid_argument("Fused MLP matrix aliases checkpoint bytes");
  if(a.rowBytes%4||g.rowBytes%16||s.rowBytes%4||o.rowBytes%16||o.offset%16||
    !AffineProfileIndex32Fits(a.rows,4096,14336,a.rowBytes/4,g.rowBytes/4)||!AffineProfileIndex32Fits(a.rows,14336,4096,s.rowBytes/4,o.rowBytes/4))throw std::invalid_argument("Fused MLP index profile mismatch");
  for(const auto& value:{output,impl_->fusedMlpHalf}){value.impl_->ready=false;value.impl_->pending=true;if(impl_->pendingOutputs.insert(value.impl_->PendingKey()).second)impl_->outputs.push_back(value.impl_);}
  // All validation is complete. Initialize the full private arena so Abort
  // recovery and inactive tails never expose uninitialized carrier bytes.
  auto blit=[impl_->command blitCommandEncoder];if(!blit)throw std::runtime_error("Cannot initialize fused half arena");
  [blit fillBuffer:half.data range:NSMakeRange(half.offset,512*14336*2) value:0];
  [blit endEncoding];
  auto encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot gather fused MLP input");
  uint32_t shape[]={uint32_t(indices.size()),4096,uint32_t(a.rowBytes/4),0};
  [encoder setComputePipelineState:impl_->packedGather];[encoder setBuffer:a.data offset:a.offset atIndex:0];[encoder setBuffer:compact.data offset:0 atIndex:1];[encoder setBytes:indices.data() length:indices.size()*4 atIndex:2];[encoder setBytes:shape length:sizeof(shape) atIndex:3];[encoder dispatchThreads:MTLSizeMake(indices.size()*4096,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
  // Input may share its original arena with the final down output.
  // Gather first; clearing inactive output rows must not erase that input.
  if(indices.size()<a.rows){auto clear=[impl_->command blitCommandEncoder];if(!clear)throw std::runtime_error("Cannot clear fused inactive rows");[clear fillBuffer:o.data range:NSMakeRange(o.offset,outBytes) value:0];[clear endEncoding];}
  struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params gateParams{indices.size(),4096,14336,a.rowBytes/4,g.rowBytes/4};
  encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot encode fused gate/up SiLU");
  [encoder setComputePipelineState:gatePipeline];[encoder setBuffer:compact.data offset:0 atIndex:0];[encoder setBuffer:gw->packed offset:0 atIndex:1];[encoder setBuffer:gw->scales offset:0 atIndex:2];[encoder setBuffer:gw->biases offset:0 atIndex:3];[encoder setBuffer:g.data offset:g.offset atIndex:4];[encoder setBytes:&gateParams length:sizeof(gateParams) atIndex:5];[encoder setBytes:indices.data() length:indices.size()*4 atIndex:6];[encoder setBuffer:uw->packed offset:0 atIndex:8];[encoder setBuffer:uw->scales offset:0 atIndex:9];[encoder setBuffer:uw->biases offset:0 atIndex:10];[encoder setBuffer:u.data offset:u.offset atIndex:11];[encoder setBuffer:half.data offset:half.offset atIndex:13];[encoder setBuffer:g.data offset:g.offset atIndex:14];[encoder dispatchThreadgroups:MTLSizeMake(448,(indices.size()+gateTile-1)/gateTile,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
  Params downParams{indices.size(),14336,4096,s.rowBytes/4,o.rowBytes/4};
  encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot encode fused MLP down projection");
  [encoder setComputePipelineState:downPipeline];[encoder setBuffer:half.data offset:half.offset atIndex:0];[encoder setBuffer:dw->packed offset:0 atIndex:1];[encoder setBuffer:dw->scales offset:0 atIndex:2];[encoder setBuffer:dw->biases offset:0 atIndex:3];[encoder setBuffer:o.data offset:o.offset atIndex:4];[encoder setBytes:&downParams length:sizeof(downParams) atIndex:5];[encoder setBytes:indices.data() length:indices.size()*4 atIndex:6];[encoder dispatchThreadgroups:MTLSizeMake(64,(indices.size()+downTile-1)/downTile,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
  ++impl_->fusedMlpBlocks;impl_->packedDispatches+=3;impl_->affineProfileDispatches+=3;if(selectedRows)impl_->indexedAffineDispatches+=3;
  impl_->affineProfileMaxThreads=std::max(gatePipeline.maxTotalThreadsPerThreadgroup,downPipeline.maxTotalThreadsPerThreadgroup);
  return true;
 }
}
#endif
