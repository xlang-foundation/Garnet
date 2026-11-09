"""Isolate a native scheduler verification bridge without xModel operator changes."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;N=M/'mistral-verification-split-score-native1';S=R/'source1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(M/'mistral-split-score-ownership1/audit2.json')['passed'];assert not S.exists()
subprocess.run(['/bin/cp','-cR',str(N/'source2'),str(S)],check=True)
for p in [S]+list(S.rglob('*')):
 if not p.is_symlink():p.chmod(p.stat().st_mode|0o200)
changes={}
def edit(name,a,b):
 p=S/name;s=p.read_text();assert s.count(a)==1,(name,s.count(a),a[:80]);p.write_text(s.replace(a,b));changes[name]=sha(p)
edit('cmake/GarnetMetal.cmake','option(GARNET_METAL_MISTRAL_VERIFY_LAYOUT ','''option(GARNET_METAL_DENSE_VERIFY_SCHEDULER "Opt-in native leased dense verification and exact scheduler span commit" OFF)
if(GARNET_METAL_DENSE_VERIFY_SCHEDULER)
    if(NOT GARNET_METAL_VERIFY_ATTENTION OR NOT GARNET_METAL_CONTINUOUS_ROWS OR NOT GARNET_METAL_TOKEN_PREFIX_CACHE)
        message(FATAL_ERROR "Dense verification scheduler requires verification, continuous rows and native token-prefix tracking")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_VERIFY_SCHEDULER=1)
endif()
option(GARNET_METAL_MISTRAL_VERIFY_LAYOUT ''')
base='src/backends/metal/'
edit(base+'decoder_session.h','    std::string VerificationStatsJson();','''    std::string VerificationStatsJson();
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    std::array<size_t,2> VerificationGeometry();
#endif''')
edit(base+'verification_session.cpp','std::string MetalDecoderSession::VerificationStatsJson(){','''#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
std::array<size_t,2> MetalDecoderSession::VerificationGeometry() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!verification_||!gpu_||poisoned_)throw std::logic_error("Prepare native verification before scheduled decoding");
    return {verification_->batch,verification_->chunk};
}
#endif
std::string MetalDecoderSession::VerificationStatsJson(){''')
edit(base+'continuous_scheduler.h','    int64_t CompleteDecode(int64_t request,int64_t token,bool finished);','''    int64_t CompleteDecode(int64_t request,int64_t token,bool finished);
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    std::string VerifyDecode(X::Value hidden,X::Value positions,X::Value slots,X::Value active,X::Value tokens,X::Value stops);
#endif''')
edit(base+'continuous_scheduler.h','        APISET().AddFunc<3>("complete_decode",&MetalContinuousScheduler::CompleteDecode);','''        APISET().AddFunc<3>("complete_decode",&MetalContinuousScheduler::CompleteDecode);
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
        APISET().AddFunc<6>("verify_decode_json",&MetalContinuousScheduler::VerifyDecode);
#endif''')
edit(base+'continuous_scheduler.cpp','#include <climits>','#include <climits>\n#include <cmath>')
method=r'''
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
std::string MetalContinuousScheduler::VerifyDecode(X::Value hidden,X::Value positions,X::Value slots,X::Value active,X::Value tokens,X::Value stops) {
    std::lock_guard<std::mutex> lock(mutex_);Ready();
    auto session=sessionOwner_.NativeData<MetalDecoderSession>();
    const auto geometry=session->VerificationGeometry();const size_t chunk=geometry[1];
    if(geometry[0]!=rows_||!chunk||chunk>4||rows_*chunk>4)
        throw std::invalid_argument("Scheduled verification exceeds prepared physical token rows");
    // Own immutable control snapshots before validating the native schedule.
    auto control=[&](X::Value source,const std::vector<int64_t>& shape,std::vector<float>& values){
        if(source.runtime()!=Host()->runtime)throw std::invalid_argument("Scheduled control runtime differs");
        X::Tensor tensor(source);auto lease=tensor.Acquire();const auto info=tensor.Info();size_t count=1;for(auto dim:shape)count*=size_t(dim);
        if(info.symbolic||info.device_type||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=shape.size()||!std::equal(shape.begin(),shape.end(),info.shape)||!info.data||info.byte_size<count*4||!IsContiguousTensor(tensor))
            throw std::invalid_argument("Scheduled verification control shape/storage differs");
        const auto data=static_cast<const float*>(info.data);values.assign(data,data+count);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),count*4);
    };
    std::vector<float> positionValues,slotValues,flags;
    positions=control(positions,{int64_t(rows_),int64_t(chunk),3},positionValues);
    slots=control(slots,{int64_t(rows_),int64_t(chunk)},slotValues);
    active=control(active,{int64_t(rows_),int64_t(chunk)},flags);
    if(tokens.runtime()!=Host()->runtime||!tokens.IsList()||tokens.Size()!=rows_*chunk)
        throw std::invalid_argument("Scheduled verification token count/runtime differs");
    auto immutableTokens=X::Value::List(Host());std::vector<int64_t> tokenValues;
    for(uint64_t i=0;i<tokens.Size();++i){auto value=tokens.Get(i);if(!value.IsInt64()||value.ToInt64()<0||value.ToInt64()>INT_MAX)throw std::invalid_argument("Scheduled verification tokens must be bounded integers");tokenValues.push_back(value.ToInt64());immutableTokens.Append(X::Value(value.ToInt64()));}
    struct Item{uint64_t id;size_t row;uint64_t epoch;int known,budget,slot,depth;std::vector<int> emitted;bool eos=false;};
    std::vector<Item> items;std::vector<int> budgets(rows_,0);std::vector<uint64_t> leases(rows_,0);
    for(const auto& entry:inflight_)if(entry.second==RequestStage::Decode){
        int known=0,budget=0;if(!scheduler_->DecodeProgress(entry.first,known,budget))throw std::logic_error("Scheduled verification decode progress invalid");
        const auto placement=allocator_->Placement(entry.first);const auto expected=trackedExpected_.find(entry.first);const int slot=allocator_->LogicalLength(entry.first);
        if(!allocator_->Tracked(entry.first)||expected==trackedExpected_.end()||expected->second!=slot+1||slot<0||placement.requestRow<0||size_t(placement.requestRow)>=rows_||!placement.rowEpoch||leases[size_t(placement.requestRow)])
            throw std::logic_error("Scheduled verification requires native tracked row placement");
        const auto row=size_t(placement.requestRow);bool ended=false;int depth=0;
        for(size_t t=0;t<chunk;++t){const auto i=row*chunk+t;const float flag=flags[i];if(flag!=0.0f&&flag!=1.0f)throw std::invalid_argument("Scheduled verification masks must be binary");if(!flag){ended=true;continue;}
            if(ended||slotValues[i]!=float(slot+int(t))||slot+int(t)>=int(capacity_))throw std::invalid_argument("Scheduled verification mask/slot differs from native request");
            for(size_t axis=0;axis<3;++axis)if(positionValues[i*3+axis]!=float(slot+int(t)))throw std::invalid_argument("Scheduled text verification position differs from zero-based native slot");++depth;
        }
        if(depth<1||depth>budget||tokenValues[row*chunk]!=known)throw std::invalid_argument("Scheduled verification known token or depth exceeds native output budget");
        budgets[row]=budget;leases[row]=placement.rowEpoch;items.push_back({entry.first,row,placement.rowEpoch,known,budget,slot,depth,{}});
    }
    if(items.empty())throw std::invalid_argument("Schedule a native dense decode batch first");
    for(size_t row=0;row<rows_;++row)if(!leases[row])for(size_t t=0;t<chunk;++t)if(flags[row*chunk+t]!=0.0f)throw std::invalid_argument("Unscheduled verification row is active");
    auto remaining=X::Value::List(Host()),epochs=X::Value::List(Host());for(size_t row=0;row<rows_;++row){remaining.Append(X::Value(int64_t(budgets[row])));epochs.Append(X::Value(int64_t(leases[row])));}
    try {
        auto result=session->VerifyGreedy(hidden,positions,slots,active,immutableTokens,stops,remaining,epochs);
        const auto statistics=nlohmann::json::parse(session->VerificationStatsJson());
        // Prove every accepted target span and token-ledger length before any
        // scheduler state is published or a terminal request lease released.
        for(auto& item:items){auto row=result.Get(uint64_t(item.row)),ids=row.Get(uint64_t(0));const auto count=row.Get(uint64_t(1)).ToInt64();auto eos=row.Get(uint64_t(2));
            if(!ids.IsList()||count<1||count>item.depth||count>item.budget||uint64_t(count)!=ids.Size()||eos.raw().tag!=X3_TAG_BOOL)throw std::logic_error("Scheduled target result exceeds accepted budget");
            for(uint64_t i=0;i<ids.Size();++i){auto value=ids.Get(i);if(!value.IsInt64()||value.ToInt64()<0||value.ToInt64()>INT_MAX)throw std::logic_error("Scheduled target result token invalid");item.emitted.push_back(int(value.ToInt64()));}item.eos=eos.ToInt64()!=0;
            if(!allocator_->ProgressValid(item.id,item.slot+int(count)))throw std::logic_error("Scheduled verification accepted token ledger differs");
        }
        nlohmann::json completed=nlohmann::json::array();
        for(const auto& item:items){
            if(item.eos||item.emitted.size()==size_t(item.budget))allocator_->Capture(item.id,item.slot+int(item.emitted.size()));
            if(!scheduler_->CompleteDecodeTokens(item.id,item.emitted,item.eos,*allocator_))throw std::logic_error("Scheduled dense target span commit failed");
            const auto stage=scheduler_->Stage(item.id);inflight_.erase(item.id);trackedExpected_.erase(item.id);
            completed.push_back({{"id",item.id},{"row",item.row},{"epoch",item.epoch},{"known",item.known},{"slot_before",item.slot},{"remaining_before",item.budget},{"submitted",item.depth},{"emitted",item.emitted},{"consumed",item.emitted.size()},{"eos",item.eos},{"finished",stage==RequestStage::Finished}});
            if(stage==RequestStage::Finished)scheduler_->ForgetFinished(item.id);
        }
        return nlohmann::json{{"requests",completed},{"verification",statistics},{"scope","Native leased dense target verification and exact scheduler span commit; no serving speed admission"}}.dump();
    }catch(...){failed_=true;throw;}
}
#endif
'''
a='#ifdef GARNET_METAL_MTP_BRIDGE\nstd::string MetalContinuousScheduler::GenerateMTP'
edit(base+'continuous_scheduler.cpp',a,method+'\n'+a)
edit('src/entry/metal_main.cpp','#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES\n    info+=','''#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    info+="; native_scheduled_dense_verify_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    info+=''')
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
(R/'source-bindings1.json').write_text(json.dumps(dict(files_sha256=files,changed_sha256=changes,base_source=str(N/'source2'),default_option_on=False,goal_complete=False),indent=2)+'\n');print(json.dumps(dict(files=len(files),changed=list(changes))))
