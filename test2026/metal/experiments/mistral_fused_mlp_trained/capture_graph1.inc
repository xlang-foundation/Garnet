// Research-only observer. It copies existing values, never replaces arithmetic.
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
bool MetalCompiledGraph::PrepareDiagnosticMlpCapture(int64_t invocation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!ready_||!gpu_||diagnosticMlpCapture_||invocation!=16||
       inputShapes_.empty()||inputShapes_[0]!=std::vector<int64_t>{4,128,4096})
        throw std::invalid_argument("Prepare one exact B4/T128 Mistral call16 MLP observation");
    auto plan=std::make_unique<DiagnosticMlpCapture>();
    plan->ffn=DescribeDiagnosticFfn();plan->generation=generation_;plan->target=size_t(invocation);
    size_t matched=0;
    for(size_t layer:{size_t(0),size_t(15),size_t(31)}) {
        auto expected="model.layers."+std::to_string(layer)+".mlp.gate_proj.weight";
        auto found=weights_.find(expected);
        if(found!=weights_.end()&&found->second==plan->ffn.gateWeight){plan->layer=layer;++matched;}
    }
    if(matched!=1)throw std::invalid_argument("Capture only exact trained Mistral layers0/15/31");
    const std::array<size_t,5> widths{4096,14336,14336,14336,4096};
    const auto memory=MetalContext::MemoryStats();const uint64_t bytes=uint64_t(512)*(2*4096+3*14336)*4;
    if(memory.recommended<=bytes+256*1024*1024||memory.allocated>memory.recommended-bytes-256*1024*1024)
        throw std::runtime_error("MLP observation exceeds device budget");
    for(size_t i=0;i<5;++i){plan->snapshots[i]=gpu_->Allocate(512,widths[i]);gpu_->Zero(plan->snapshots[i]);}
    diagnosticMlpCapture_=std::move(plan);return true;
}
void MetalCompiledGraph::ObserveDiagnosticMlp(MetalContext& gpu,size_t operation) {
    auto* p=diagnosticMlpCapture_.get();if(!p||p->calls!=p->target)return;
    if(p->generation!=generation_)throw std::logic_error("Stale MLP observation generation");
    auto copy=[&](size_t part,size_t slot){
        if(p->mask&(1u<<part))throw std::logic_error("Duplicate MLP observation boundary");
        const auto source=MetalContext::MatrixBufferInfo(slots_.at(slot).matrix);
        const auto destination=MetalContext::MatrixBufferInfo(p->snapshots[part]);
        if(!source.identity||source.identity==destination.identity)throw std::logic_error("MLP snapshot aliases its producer");
        p->sources[part]=source.identity;gpu.EncodeDiagnosticMlpCopy(slots_[slot].matrix,p->snapshots[part]);p->mask|=1u<<part;
    };
    const size_t begin=p->ffn.begin;
    if(operation==begin+1)copy(0,ops_.at(operation).a);
    if(operation==begin+3){copy(1,ops_[operation].a);copy(2,ops_[operation].b);}
    if(operation==begin+4)copy(3,ops_[operation].a);
    if(operation==begin+5)copy(4,ops_[operation].b);
}
std::string MetalCompiledGraph::SaveDiagnosticMlpCapture(const std::string& destination) {
    std::lock_guard<std::mutex> lock(mutex_);auto* p=diagnosticMlpCapture_.get();
    if(!ready_||!gpu_||!p||p->generation!=generation_||p->calls<p->target||p->mask!=31||p->saved)
        throw std::logic_error("MLP observation is incomplete, stale or already saved");
    std::filesystem::path folder(destination);
    if(!folder.is_absolute()||std::filesystem::exists(folder)||!std::filesystem::is_directory(folder.parent_path()))
        throw std::invalid_argument("MLP observation requires a fresh absolute output folder");
    // Download checks actual readiness after the session command completed.
    // Nothing is published if an asynchronous command/range failure occurred.
    std::array<std::vector<float>,5> values;
    const std::array<size_t,5> widths{4096,14336,14336,14336,4096};
    const std::array<const char*,5> names{"norm-input","gate","up","silu","down"};
    for(size_t i=0;i<5;++i){values[i]=gpu_->Download(p->snapshots[i]);if(values[i].size()!=512*widths[i]||!std::all_of(values[i].begin(),values[i].end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Invalid trained MLP observation values");}
    if(!std::filesystem::create_directory(folder))throw std::runtime_error("Cannot create MLP observation folder");
    for(size_t i=0;i<5;++i){std::ofstream stream(folder/(std::string(names[i])+".f32"),std::ios::binary);stream.write(reinterpret_cast<const char*>(values[i].data()),values[i].size()*4);if(!stream)throw std::runtime_error("Cannot save complete MLP observation");}
    std::ostringstream metadata;metadata<<"{\"layer\":"<<p->layer<<",\"invocation\":"<<p->target<<",\"rows\":512,\"chunk\":128,\"offset\":1920,\"mask\":"<<p->mask<<",\"generation\":"<<p->generation<<",\"snapshot_bytes\":"<<uint64_t(512)*(2*4096+3*14336)*4<<",\"source_identities\":[";
    for(size_t i=0;i<5;++i){if(i)metadata<<",";metadata<<p->sources[i];}metadata<<"],\"arithmetic_changed\":false,\"diagnostic_only\":true}";
    std::ofstream stream(folder/"metadata.json");stream<<metadata.str()<<'\n';if(!stream)throw std::runtime_error("Cannot save MLP observation metadata");p->saved=true;return metadata.str();
}
#endif
