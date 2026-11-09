// SPDX-License-Identifier: Apache-2.0
#include "streamed_bf16_projection.h"
#include "cuda_device_scope.h"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <set>

namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunStreamedFP8Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
int RunBF16Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
int RunBF16ProjectionFloat32(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
namespace {
constexpr uint64_t SlotBytes=256ULL<<20;
std::atomic<uint64_t> mappedBytes{0},mappedFiles{0},activePools{0},uploadedBytes{0};
std::atomic<uint64_t> registeredBytes{0};
std::atomic<uint64_t> deviceCachedBytes{0},deviceCacheHits{0};
void Check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct MappedAsset {
    SafeTensorsMappedFile mapping;
    uint64_t bytes=0;
    std::string file,identity;
    std::mutex mutex;
    void* registered=nullptr;
    int registrationDevice=-1;
    std::map<int,cudaEvent_t> copies;
    ~MappedAsset(){
        int previous=-1;cudaGetDevice(&previous);
        // Source mappings can disappear independently of the shared GPU pool.
        // Wait only for DMA reading this asset before unregistering/unmapping.
        for(const auto& entry:copies){cudaSetDevice(entry.first);cudaEventSynchronize(entry.second);cudaEventDestroy(entry.second);}
        if(registered){cudaSetDevice(registrationDevice);cudaHostUnregister(registered);registeredBytes-=bytes;}
        if(previous>=0)cudaSetDevice(previous);
        if(bytes){mappedBytes-=bytes;--mappedFiles;}
    }
    void Register(){
        std::lock_guard<std::mutex> lock(mutex);
        if(registered)return;
        const char* configured=std::getenv("GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB");
        if(!configured||!*configured)throw std::runtime_error("registered_bf16 requires an explicit GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB host-memory budget");
        const std::string value(configured);size_t consumed=0;
        const auto mib=std::stoull(value,&consumed);
        if(consumed!=value.size()||value[0]=='-'||mib>(std::numeric_limits<uint64_t>::max()>>20))
            throw std::runtime_error("invalid registered checkpoint host-memory budget");
        const uint64_t budget=mib<<20;
        auto before=registeredBytes.load();
        do{if(before>budget||bytes>budget-before)throw std::runtime_error("registered checkpoint host-memory budget exceeded");}
        while(!registeredBytes.compare_exchange_weak(before,before+bytes));
        try {
            Check(cudaGetDevice(&registrationDevice));int supported=0;
            Check(cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterReadOnlySupported,registrationDevice));
            if(!supported)throw std::runtime_error("CUDA device does not support registering read-only checkpoints");
            void* base=const_cast<void*>(mapping.DataAt(0,bytes));
            if(!base)throw std::runtime_error("checkpoint mapping unavailable for registration");
            Check(cudaHostRegister(base,size_t(bytes),cudaHostRegisterPortable|cudaHostRegisterReadOnly));registered=base;
        }catch(...){registeredBytes-=bytes;throw;}
    }
    void RecordCopy(int device,cudaStream_t upload){
        std::lock_guard<std::mutex> lock(mutex);
        auto& event=copies[device];
        if(!event)Check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
        Check(cudaEventRecord(event,upload));
    }
};
std::mutex assetsMutex;
std::map<std::string,std::weak_ptr<MappedAsset>> assets;
std::shared_ptr<MappedAsset> AcquireAsset(const StreamedBF16ProjectionSpec& spec){
    std::lock_guard<std::mutex> lock(assetsMutex);
    const auto path=std::filesystem::u8path(spec.file);
    if(std::filesystem::file_size(path)!=spec.fileBytes||
        std::filesystem::last_write_time(path).time_since_epoch().count()!=spec.modified)
        throw std::runtime_error("streamed projection checkpoint source changed; rebuild the model cache");
    const auto key=spec.file+":"+std::to_string(spec.fileBytes)+":"+std::to_string(spec.modified);
    if(auto found=assets[key].lock())return found;
    auto asset=std::make_shared<MappedAsset>();std::string error;
    if(!asset->mapping.Open(path,error))throw std::runtime_error(error);
    asset->bytes=spec.fileBytes;asset->file=spec.file;asset->identity=key;mappedBytes+=asset->bytes;++mappedFiles;
    assets[key]=asset;return asset;
}
struct StagingPool {
    struct Slot {void* host=nullptr;void* device=nullptr;cudaEvent_t copied=nullptr,consumed=nullptr;bool used=false;};
    struct CachedMatrix {
        void* data=nullptr;cudaEvent_t ready=nullptr;uint64_t bytes=0;std::string file;
        std::map<cudaStream_t,cudaEvent_t> readers;
        ~CachedMatrix(){
            if(ready)cudaEventSynchronize(ready);
            for(const auto& reader:readers){cudaEventSynchronize(reader.second);cudaEventDestroy(reader.second);}
            if(ready)cudaEventDestroy(ready);
            if(data){cudaFree(data);deviceCachedBytes-=bytes;}
        }
        void RecordUse(cudaStream_t stream){
            auto& event=readers[stream];if(!event)Check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            Check(cudaEventRecord(event,stream));
        }
    };
    int device=-1;cudaStream_t upload=nullptr;std::array<Slot,2> slots{};
    std::shared_ptr<void> gemm;std::mutex mutex;size_t next=0;bool counted=false,failed=false;
    uint64_t cacheBudget=0,cacheBytes=0;
    std::map<std::string,std::unique_ptr<CachedMatrix>> cached;
    explicit StagingPool(int id):device(id){}
    ~StagingPool(){
        int previous=-1;cudaGetDevice(&previous);cudaSetDevice(device);
        if(upload)cudaStreamSynchronize(upload);
        cached.clear();
        for(auto& slot:slots){
            if(slot.used&&slot.consumed)cudaEventSynchronize(slot.consumed);
            if(slot.copied)cudaEventDestroy(slot.copied);
            if(slot.consumed)cudaEventDestroy(slot.consumed);
            if(slot.device)cudaFree(slot.device);
            if(slot.host)cudaFreeHost(slot.host);
        }
        if(upload)cudaStreamDestroy(upload);
        gemm.reset();if(counted)--activePools;
        if(previous>=0)cudaSetDevice(previous);
    }
    void Initialize(){
        const char* cache=std::getenv("GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB");
        if(cache&&*cache){
            const std::string value(cache);size_t consumed=0;const auto mib=std::stoull(value,&consumed);
            if(consumed!=value.size()||value[0]=='-'||mib>(std::numeric_limits<uint64_t>::max()>>20))
                throw std::runtime_error("invalid projection device-cache budget");
            cacheBudget=mib<<20;
        }
        Check(cudaStreamCreateWithFlags(&upload,cudaStreamNonBlocking));
        for(auto& slot:slots){
            Check(cudaHostAlloc(&slot.host,SlotBytes,cudaHostAllocPortable));
            Check(cudaMalloc(&slot.device,SlotBytes));
            Check(cudaEventCreateWithFlags(&slot.copied,cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&slot.consumed,cudaEventDisableTiming));
        }
        gemm=CreateBF16ProjectionContext();if(!gemm)throw std::runtime_error("streamed projection GEMM context unavailable");
        counted=true;++activePools;
    }
    void ClearCached(const std::set<std::string>& files){
        std::lock_guard<std::mutex> lock(mutex);CUDADeviceScope scope(device);
        for(auto iterator=cached.begin();iterator!=cached.end();){
            if(files.count(iterator->second->file)){cacheBytes-=iterator->second->bytes;iterator=cached.erase(iterator);}
            else ++iterator;
        }
    }
    void Project(const std::shared_ptr<MappedAsset>& asset,const void* source,size_t bytes,const void* input,void* output,int rows,int inner,int columns,
        cudaStream_t stream,void* workspace,size_t workspaceBytes,bool direct,bool floatOutput,bool cacheWeights,bool fp8Compute){
        std::lock_guard<std::mutex> lock(mutex);
        if(failed)throw std::runtime_error("streamed projection staging pool is in failed state");
        auto& slot=slots[next];next=(next+1)%slots.size();
        auto multiply=fp8Compute?RunStreamedFP8Projection:(floatOutput?RunBF16ProjectionFloat32:RunBF16Projection);
        try {
            // A scan-resistant, bounded cache keeps the first admitted immutable
            // matrices rather than thrashing when a layer sweep exceeds VRAM.
            // It is disabled by default, and model retirement explicitly trims
            // entries for that model's checkpoints before the next pipeline stage.
            if(direct&&cacheBudget&&cacheWeights){
                const auto offset=static_cast<const unsigned char*>(source)-static_cast<const unsigned char*>(asset->mapping.DataAt(0,asset->bytes));
                const auto key=asset->identity+":"+std::to_string(offset)+":"+std::to_string(bytes);
                auto found=cached.find(key);
                if(found==cached.end()&&cacheBytes<=cacheBudget&&bytes<=cacheBudget-cacheBytes){
                    auto value=std::make_unique<CachedMatrix>();value->file=asset->file;value->bytes=bytes;
                    Check(cudaEventCreateWithFlags(&value->ready,cudaEventDisableTiming));
                    Check(cudaMalloc(&value->data,bytes));deviceCachedBytes+=bytes;cacheBytes+=bytes;
                    found=cached.emplace(key,std::move(value)).first;
                    Check(cudaMemcpyAsync(found->second->data,source,bytes,cudaMemcpyHostToDevice,upload));
                    Check(cudaEventRecord(found->second->ready,upload));asset->RecordCopy(device,upload);uploadedBytes+=bytes;
                }else if(found!=cached.end())++deviceCacheHits;
                if(found!=cached.end()){
                    Check(cudaStreamWaitEvent(stream,found->second->ready,0));
                    if(multiply(gemm,input,found->second->data,output,rows,inner,columns,stream,workspace,workspaceBytes))
                        throw std::runtime_error("cached projection GEMM failed");
                    found->second->RecordUse(stream);return;
                }
            }
            if(slot.used){if(!direct)Check(cudaEventSynchronize(slot.copied));Check(cudaStreamWaitEvent(upload,slot.consumed,0));}
            if(!direct)std::memcpy(slot.host,source,bytes);
            Check(cudaMemcpyAsync(slot.device,direct?source:slot.host,bytes,cudaMemcpyHostToDevice,upload));
            Check(cudaEventRecord(slot.copied,upload));
            if(direct)asset->RecordCopy(device,upload);
            Check(cudaStreamWaitEvent(stream,slot.copied,0));
            if(multiply(gemm,input,slot.device,output,rows,inner,columns,stream,workspace,workspaceBytes))
                throw std::runtime_error("streamed projection GEMM failed");
            Check(cudaEventRecord(slot.consumed,stream));slot.used=true;uploadedBytes+=bytes;
        } catch(...) {
            // A failed event publication must not leave DMA using a host slot
            // or a GEMM using its device slot after the error is returned.
            failed=true;cudaStreamSynchronize(upload);cudaStreamSynchronize(stream);throw;
        }
    }
};
std::mutex poolsMutex;
std::map<int,std::weak_ptr<StagingPool>> pools;
std::shared_ptr<StagingPool> AcquirePool(int device){
    std::lock_guard<std::mutex> lock(poolsMutex);
    if(auto found=pools[device].lock())return found;
    auto pool=std::make_shared<StagingPool>(device);pool->Initialize();pools[device]=pool;return pool;
}
}
void StreamedBF16ProjectionSpec::Validate() const {
    if(fp8Compute&&(floatOutput||inner%16||columns%16))
        throw std::invalid_argument("FP8 projection requires 16-aligned matrices and BF16 output");
    if(file.empty()||file.size()>32768||file.find('\0')!=std::string::npos||!std::filesystem::u8path(file).is_absolute()||
        columns<1||columns>32768||inner<1||inner>16384||
        bytes!=uint64_t(columns)*uint64_t(inner)*2||bytes>SlotBytes||
        offset>fileBytes||bytes>fileBytes-offset)
        throw std::invalid_argument("streamed projection requires a valid bounded BF16 matrix/checkpoint range (<=256 MiB)");
}
StreamedBF16ProjectionSpec StreamedBF16ProjectionSpec::FromMetadata(const SafeTensorMetadata& meta){
    if(meta.dataType!="BF16"||meta.shape.size()!=2||meta.shape[0]>32768||meta.shape[1]>16384||meta.shape[0]<1||meta.shape[1]<1)
        throw std::invalid_argument("streamed projection requires rank-two BF16 checkpoint weights");
    const auto file=std::filesystem::weakly_canonical(meta.filePath);
    StreamedBF16ProjectionSpec spec;spec.file=file.u8string();spec.offset=meta.dataOffset;spec.bytes=meta.dataSize;
    spec.fileBytes=std::filesystem::file_size(file);spec.modified=std::filesystem::last_write_time(file).time_since_epoch().count();
    spec.columns=int(meta.shape[0]);spec.inner=int(meta.shape[1]);spec.Validate();return spec;
}
std::string StreamedBF16ProjectionSpec::Serialize() const {
    Validate();nlohmann::json value{{"schema",1},{"file",file},{"offset",offset},{"bytes",bytes},
        {"file_bytes",fileBytes},{"modified",modified},{"columns",columns},{"inner",inner},{"registered_source",registeredSource}};
    if(floatOutput)value["float_output"]=true;
    if(fp8Compute)value["fp8_compute"]=true;
    if(!cacheWeights)value["cache_weights"]=false;
    return value.dump();
}
StreamedBF16ProjectionSpec StreamedBF16ProjectionSpec::Parse(const std::string& value){
    if(value.size()>65536)throw std::invalid_argument("streamed projection metadata is too large");
    const auto json=nlohmann::json::parse(value);if(json.at("schema")!=1)throw std::invalid_argument("unsupported streamed projection schema");
    StreamedBF16ProjectionSpec spec;spec.file=json.at("file").get<std::string>();spec.offset=json.at("offset").get<uint64_t>();
    spec.bytes=json.at("bytes").get<uint64_t>();spec.fileBytes=json.at("file_bytes").get<uint64_t>();
    spec.modified=json.at("modified").get<int64_t>();spec.columns=json.at("columns").get<int>();spec.inner=json.at("inner").get<int>();
    spec.registeredSource=json.value("registered_source",false);
    spec.floatOutput=json.value("float_output",false);
    spec.fp8Compute=json.value("fp8_compute",false);
    spec.cacheWeights=json.value("cache_weights",true);
    spec.Validate();return spec;
}
struct StreamedBF16Projection::State {
    StreamedBF16ProjectionSpec spec;std::mutex mutex;
    std::shared_ptr<MappedAsset> asset;std::shared_ptr<StagingPool> pool;int device=-1;
    explicit State(StreamedBF16ProjectionSpec value):spec(std::move(value)){spec.Validate();}
};
StreamedBF16Projection::StreamedBF16Projection(StreamedBF16ProjectionSpec value):state_(new State(std::move(value))){}
StreamedBF16Projection::~StreamedBF16Projection()=default;
void StreamedBF16Projection::Run(const void* input,void* output,int rows,cudaStream_t stream,void* workspace,size_t workspaceBytes){
    if(!input||!output||rows<1||!workspace||workspaceBytes<16384)throw std::invalid_argument("invalid streamed projection execution buffers");
    int device=-1;Check(cudaGetDevice(&device));std::shared_ptr<MappedAsset> asset;std::shared_ptr<StagingPool> pool;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if(state_->device>=0&&state_->device!=device)throw std::runtime_error("streamed projection instance changed CUDA devices");
        cudaStreamCaptureStatus capture;Check(cudaStreamIsCapturing(stream,&capture));
        if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("streamed checkpoint projections do not support CUDA graph capture");
        if(!state_->asset)state_->asset=AcquireAsset(state_->spec);
        if(state_->spec.registeredSource)state_->asset->Register();
        if(!state_->pool)state_->pool=AcquirePool(device);
        state_->device=device;asset=state_->asset;pool=state_->pool;
    }
    const auto* source=asset->mapping.DataAt(state_->spec.offset,state_->spec.bytes);
    if(!source)throw std::runtime_error("streamed projection checkpoint span unavailable");
    pool->Project(asset,source,size_t(state_->spec.bytes),input,output,rows,state_->spec.inner,state_->spec.columns,stream,workspace,workspaceBytes,state_->spec.registeredSource,state_->spec.floatOutput,state_->spec.cacheWeights,state_->spec.fp8Compute);
}
void ClearStreamedProjectionDeviceCache(const std::vector<std::string>& checkpointFiles){
    const std::set<std::string> files(checkpointFiles.begin(),checkpointFiles.end());
    std::vector<std::shared_ptr<StagingPool>> owners;
    {std::lock_guard<std::mutex> lock(poolsMutex);for(const auto& entry:pools)if(auto owner=entry.second.lock())owners.push_back(std::move(owner));}
    for(const auto& owner:owners)owner->ClearCached(files);
}
std::string StreamedBF16ProjectionStats(){
    const auto count=activePools.load();
    return nlohmann::json{{"active_device_pools",count},{"mapped_files",mappedFiles.load()},
        {"mapped_virtual_bytes",mappedBytes.load()},{"pinned_staging_bytes",count*2*SlotBytes},
        {"device_staging_bytes",count*2*SlotBytes},{"uploaded_weight_bytes",uploadedBytes.load()},{"registered_checkpoint_bytes",registeredBytes.load()},
        {"device_cached_weight_bytes",deviceCachedBytes.load()},{"device_cache_hits",deviceCacheHits.load()},
        {"scope","Native projection staging and mapped address ranges only; not total process RAM, resident physical pages or total VRAM"}}.dump();
}
}
