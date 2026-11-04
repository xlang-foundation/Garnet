// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "metal_context.h"
#include "safetensors_index.h"
#ifdef GARNET_METAL_MLX_AFFINE_INT4
#include "mlx_qwen35_checkpoint.h"
#endif
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <map>
namespace Garnet {
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
// Graphs retain this owner independently of the preparing session. Every
// execution path locks it through GPU completion, including standalone Run.
struct MetalPrefillScratch {
    std::mutex mutex;
    std::map<std::pair<size_t,size_t>,std::vector<MetalMatrix>> buffers;
};
#endif
class MetalCompiledGraph {
    friend class MetalDecoderSession;
    struct Slot {MetalMatrix matrix;size_t rows=0,columns=0;std::vector<int64_t> shape;MetalBF16Weight bf16;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
        MetalAffineInt4Weight affineInt4;
#endif
    };
    enum class Kind {MatMul,BF16MatVec,Norm,Silu,Add,Bias,Mask,Conv,Delta,Slice,Prepare,GateNorm,Concat,NormRope,KVUpdate,KVChunk,AttentionChunk,Scores,AttentionContext,Vision,Layout,DeltaSplit,ConvChunk,DeltaChunk};
    struct Op {Kind kind;size_t a,b,out;float epsilon=0,offset=0;bool transpose=false;
        std::vector<size_t> arguments={},dimensions={};bool contiguousSlots=false;size_t auxiliaryOutput=SIZE_MAX;
        bool indexedTokenProjection=false,indexedDecodeProjection=false;};
    std::unique_ptr<MetalContext> gpu_;
    std::vector<Slot> slots_;
    std::vector<Op> ops_;
    std::vector<size_t> inputs_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::unordered_map<uint64_t,size_t> tensors_;
    std::unordered_map<std::string,size_t> weights_;
    struct CheckpointDependency {SafeTensorMetadata metadata;std::filesystem::file_time_type modified;uintmax_t bytes;};
    struct SharedWeight {Slot slot;SafeTensorMetadata metadata;
        std::filesystem::file_time_type modified;uintmax_t bytes;
        std::vector<CheckpointDependency> dependencies;
    };
    std::unordered_map<std::string,SharedWeight> sharedWeights_,weightRecords_;
    std::unordered_map<uint64_t,std::unordered_map<std::string,size_t>> bindings_;
    std::unordered_map<size_t,std::pair<size_t,size_t>> controlSources_;
    size_t output_=0;
    std::vector<size_t> outputParts_;
    std::vector<std::vector<size_t>> outputStorageGroups_;
    std::vector<uint8_t> residentOps_;
    uint64_t generation_=0;
    double compileMs_=0,inferenceMs_=0;
    size_t visionGlobalAlphaDispatches_=0;
    size_t visionParallelDispatches_=0;
    size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;
    bool resetKVPrefix_[2]={false,false};
    bool ready_=false,bf16Projections_=false;
    mutable std::mutex mutex_;
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::shared_ptr<MetalPrefillScratch> prefillScratch_;
    size_t sharedPrefillScratchBytes_=0;
    void SharePrefillScratch(const std::shared_ptr<MetalPrefillScratch>&);
#endif
    SafeTensorsIndex index_;
    const SafeTensorMetadata* FindWeightMetadata(const std::string&) const;
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    MlxQwen35Checkpoint mlxCheckpoint_;
    size_t AffineInt4Weight(const std::string&);
#endif
    size_t Weight(const std::string&);
    size_t BF16Weight(const std::string&,bool visionPrefill=false);
    size_t Allocate(size_t,size_t);
    size_t LowerVision(size_t input,const X::Value& attributes);
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    bool lazyDefaultPlanReady_=false;std::vector<size_t> lazyDefaultSlots_;
    size_t lazyRetiredBytes_=0,lazyRehydratedBytes_=0;
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
    size_t nonresidentPackSlots_=0;
#endif
#endif
    void RetireResidentDefaults(bool preserveHiddenInput=false);
    void MaterializeDefault(size_t slot);
    void Clear();
    void PrepareInputs(X::Value,const std::unordered_set<size_t>& skip={});
    void Encode(MetalContext&,bool packedOutput=true,const MetalMatrix* greedyOutput=nullptr,
        const std::vector<size_t>* selectedTokenRows=nullptr);
    void EncodeGreedy(MetalContext&,const MetalMatrix& prediction,const std::vector<size_t>* selectedRows=nullptr);
    bool CanUpdateKVInPlace(size_t part) const;
    bool ComputeResetKVPrefix(size_t part) const;
    bool CanResetKVPrefix(size_t part) const {return part>=1&&part<=2&&resetKVPrefix_[part-1];}
    bool SwapOutputStorage(size_t part,MetalMatrix& replacement);
    bool BuildInternal(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false);
public:
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source);
    X::Value Run(X::Value inputs);
    void Close();
    std::vector<MetalResourceReference> MemoryReferences();
    size_t LazyRetiredBytes() const;
    size_t RetiredPrefillHiddenInputSlots() const;
    size_t LazyRehydratedBytes() const;
    size_t LazyPlannedSlots() const;
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
    size_t NonresidentPackSlots() const {std::lock_guard<std::mutex> lock(mutex_);return nonresidentPackSlots_;}
#endif
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    size_t SharedPrefillScratchBytes() const {return sharedPrefillScratchBytes_;}
#endif
    bool Ready() const {return ready_;}
    double CompileMs() const {return compileMs_;}
    double LastInferenceMs() const {return inferenceMs_;}
    size_t OperationCount() const {return ops_.size();}
    size_t LastVisionGlobalAlphaDispatches() const {return visionGlobalAlphaDispatches_;}
    size_t LastVisionParallelDispatches() const {return visionParallelDispatches_;}
    size_t LastVisionPrefetchDispatches() const {return visionPrefetchDispatches_;}
    size_t LastVisionVectorDispatches() const {return visionVectorDispatches_;}
    size_t LastMlpGraphDispatches() const {return mlpGraphDispatches_;}
    BEGIN_PACKAGE(MetalCompiledGraph)
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
        APISET().AddProp("shared_prefill_scratch_bytes",&MetalCompiledGraph::SharedPrefillScratchBytes);
#endif
        APISET().AddFunc<3>("build",&MetalCompiledGraph::Build);
        APISET().AddFunc<3>("build_bf16",&MetalCompiledGraph::BuildBF16);
        APISET().AddFunc<4>("build_shared",&MetalCompiledGraph::BuildShared);
        APISET().AddFunc<1>("run",&MetalCompiledGraph::Run);
        APISET().AddFunc<0>("close",&MetalCompiledGraph::Close);
        APISET().AddProp("lazy_default_retired_bytes",&MetalCompiledGraph::LazyRetiredBytes);
        APISET().AddProp("retired_prefill_hidden_input_slots",&MetalCompiledGraph::RetiredPrefillHiddenInputSlots);
        APISET().AddProp("lazy_default_rehydrated_bytes",&MetalCompiledGraph::LazyRehydratedBytes);
        APISET().AddProp("lazy_default_planned_slots",&MetalCompiledGraph::LazyPlannedSlots);
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
        APISET().AddProp("nonresident_pack_slots",&MetalCompiledGraph::NonresidentPackSlots);
#endif
        APISET().AddProp("ready",&MetalCompiledGraph::Ready);
        APISET().AddProp("compile_ms",&MetalCompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&MetalCompiledGraph::LastInferenceMs);
        APISET().AddProp("operation_count",&MetalCompiledGraph::OperationCount);
        APISET().AddProp("vision_global_alpha_dispatches",&MetalCompiledGraph::LastVisionGlobalAlphaDispatches);
        APISET().AddProp("vision_parallel_dispatches",&MetalCompiledGraph::LastVisionParallelDispatches);
        APISET().AddProp("vision_prefetch_dispatches",&MetalCompiledGraph::LastVisionPrefetchDispatches);
        APISET().AddProp("vision_vector_dispatches",&MetalCompiledGraph::LastVisionVectorDispatches);
        APISET().AddProp("mlp_prefill_graph_dispatches",&MetalCompiledGraph::LastMlpGraphDispatches);
    END_PACKAGE
};
}
