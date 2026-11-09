#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
void MetalContext::EncodeDiagnosticMlpCopy(const MetalMatrix& source,const MetalMatrix& destination) {
    if(!impl_->command||!source.impl_||!destination.impl_)throw std::invalid_argument("MLP snapshot requires an open command and two matrices");
    auto a=source.impl_->matrix,o=destination.impl_->matrix;
    if(a.device!=impl_->device||o.device!=impl_->device||a.dataType!=MPSDataTypeFloat32||o.dataType!=MPSDataTypeFloat32||
       a.rows!=512||o.rows!=a.rows||o.columns!=a.columns||(a.columns!=4096&&a.columns!=14336)||a.data==o.data||
       a.rowBytes<a.columns*4||o.rowBytes<o.columns*4||a.offset%4||o.offset%4)
        throw std::invalid_argument("MLP snapshot exact shape/device/alias mismatch");
    if(!source.impl_->ready&&!(source.impl_->pending&&impl_->pendingOutputs.count(source.impl_->PendingKey())))
        throw std::logic_error("MLP snapshot source is not ready in this command");
    if(destination.impl_->pending&&!impl_->pendingOutputs.count(destination.impl_->PendingKey()))
        throw std::logic_error("MLP snapshot destination belongs to another command");
    destination.impl_->ValidateSpanWrite(impl_->pendingOutputs.count(destination.impl_->PendingKey()));
    auto span=[](MPSMatrix* matrix){
        if(matrix.rows-1>(SIZE_MAX-matrix.columns*4)/matrix.rowBytes)throw std::overflow_error("MLP snapshot span overflow");
        size_t bytes=(matrix.rows-1)*matrix.rowBytes+matrix.columns*4;
        if(matrix.offset>matrix.data.length||bytes>matrix.data.length-matrix.offset)throw std::invalid_argument("MLP snapshot span exceeds allocation");
    };span(a);span(o);
    auto encoder=[impl_->command blitCommandEncoder];if(!encoder)throw std::runtime_error("Cannot encode MLP snapshot blit");
    if(a.rowBytes==a.columns*4&&o.rowBytes==o.columns*4)[encoder copyFromBuffer:a.data sourceOffset:a.offset toBuffer:o.data destinationOffset:o.offset size:a.rows*a.columns*4];
    else for(size_t row=0;row<a.rows;++row)[encoder copyFromBuffer:a.data sourceOffset:a.offset+row*a.rowBytes toBuffer:o.data destinationOffset:o.offset+row*o.rowBytes size:a.columns*4];
    [encoder endEncoding];destination.impl_->ready=false;destination.impl_->pending=true;
    if(impl_->pendingOutputs.insert(destination.impl_->PendingKey()).second)impl_->outputs.push_back(destination.impl_);
}
#endif
