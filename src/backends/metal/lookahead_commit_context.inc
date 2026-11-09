// SPDX-License-Identifier: Apache-2.0
size_t MetalContext::LastLookaheadKVCommitDispatches() const {return impl_->lookaheadKVCommitDispatches;}
void MetalContext::EncodeLookaheadKVCommit(const MetalMatrix& local,const MetalMatrix& canonical,
    const LookaheadLayout& layout,const std::vector<LookaheadLayout::View>& views,
    const std::vector<LookaheadLayout::CommitSelection>& paths) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin lookahead KV commit first");
    if(!local.impl_||!canonical.impl_||views.size()!=paths.size()||views.size()>layout.Batch())
        throw std::invalid_argument("Invalid lookahead KV commit owners");
    auto t=local.impl_->matrix,c=canonical.impl_->matrix;
    LookaheadProjectionMatrix(t,impl_->device);LookaheadProjectionMatrix(c,impl_->device);
    if(t.rows!=layout.Batch()*16||t.columns!=1024||c.rows!=layout.Batch()||c.columns!=8*layout.Capacity()*128||
       t.data==c.data||t.offset%16||c.offset%16||t.rowBytes%16||c.rowBytes%16||
       t.rowBytes/4>UINT32_MAX/t.rows||c.rowBytes/4>UINT32_MAX/c.rows||!impl_->kernels->kvChunkScatter)
        throw std::invalid_argument("Lookahead KV commit differs from prepared vector layout");
    for(const auto* matrix:{&local,&canonical})if(!matrix->impl_->ready&&
        !(matrix->impl_->pending&&impl_->pendingOutputs.count(matrix->impl_->PendingKey())))
        throw std::logic_error("Lookahead KV commit operand uninitialized or foreign pending");
    if(canonical.impl_->pending&&!impl_->pendingOutputs.count(canonical.impl_->PendingKey()))
        throw std::logic_error("Foreign canonical KV writer");
    canonical.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(canonical.impl_->PendingKey()));
    std::array<float,64> slots{},active{};std::array<bool,4> assigned{};
    for(size_t i=0;i<views.size();++i) {
        const auto& view=views[i];layout.Position(view,0);const auto& path=paths[i];const size_t row=view.Row();
        if(row>=layout.Batch()||assigned[row]||!path.count||path.count>layout.Levels()||path.local[0])
            throw std::invalid_argument("Invalid/duplicate lookahead KV commit row/path");
        assigned[row]=true;size_t branch=0;
        if(path.count>1) {
            if(path.local[1]<layout.LookaheadRows())throw std::invalid_argument("Lookahead window cannot become canonical");
            const size_t relative=path.local[1]-layout.LookaheadRows();
            if(relative%(layout.Levels()-1))throw std::invalid_argument("Lookahead commit branch skips its first node");
            branch=relative/(layout.Levels()-1);
        }
        const auto expected=layout.SelectCommit(view,branch,path.count);
        if(expected.local!=path.local)throw std::invalid_argument("Lookahead commit path/padding differs from selected branch");
        for(size_t depth=0;depth<path.count;++depth) {
            const size_t node=path.local[depth];
            if(layout.Position(view,node)!=view.Slot()+depth)throw std::invalid_argument("Lookahead commit positions are not contiguous");
            slots[row*16+node]=float(view.Slot()+depth);active[row*16+node]=1;
        }
    }
    if(views.empty())return;
    // The existing scatter kernel only needs unique destinations. The stronger
    // host proof here establishes one contiguous logical accepted path even
    // when its physical nodes are not a contiguous prefix of the local tile.
    struct Params {uint64_t batch,heads,dim,capacity,chunk,tStride,cStride,sStride,aStride,oStride,contiguous;};
    Params p{layout.Batch(),8,128,layout.Capacity(),16,t.rowBytes/4,c.rowBytes/4,16,16,c.rowBytes/4,1};
    auto encoder=impl_->ComputeEncoder(__func__);if(!encoder)throw std::runtime_error("Cannot encode accepted lookahead KV scatter");
    [encoder setComputePipelineState:impl_->kernels->kvChunkScatter];
    [encoder setBuffer:t.data offset:t.offset atIndex:0];
    [encoder setBytes:slots.data() length:layout.Batch()*16*sizeof(float) atIndex:2];
    [encoder setBytes:active.data() length:layout.Batch()*16*sizeof(float) atIndex:3];
    [encoder setBuffer:c.data offset:c.offset atIndex:4];[encoder setBytes:&p length:sizeof(p) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(layout.Batch()*16*1024/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding];canonical.impl_->ready=false;canonical.impl_->pending=true;
    if(impl_->pendingOutputs.insert(canonical.impl_->PendingKey()).second)impl_->outputs.push_back(canonical.impl_);
    ++impl_->lookaheadKVCommitDispatches;
    }
}
