// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "compiled_graph.h"
#include <array>
#include <map>
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
#include "lookahead_row_owner.h"
#include <optional>
#endif
#ifdef GARNET_METAL_TARGET_JACOBI
#include "target_jacobi_candidates.h"
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
#include "../../runtime/kv/prefix_cache.h"
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
#include "prefix_precision_history.h"
#endif
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
#include "../../runtime/scheduler/request_row_leases.h"
#endif
namespace Garnet {
// One synchronous request context. Graph owners are retained; request state is
// private to this session, including when graphs are reused by another session.
class MetalDecoderSession {
    struct VerificationRun;
    struct State {
        std::array<MetalMatrix,2> matrix;
        std::array<std::vector<int64_t>,2> shape;
        std::array<size_t,2> columns;
        size_t rows;
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        uint64_t tokenOwnerGeneration=0;
        std::array<bool,2> tokenCompact{false,false};
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
        bool prefixEligible=false;uint64_t prefixOwnerGeneration=0;
#endif
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        std::array<bool,2> prefixLayout{false,false},zeroOutsideDirty{false,false};
        std::array<size_t,2> dirtyExtent{0,0};
#endif
    };
    std::unique_ptr<MetalContext> gpu_;
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    struct LookaheadInstruction {MetalCompiledGraph::Op op;bool attention=false;};
    struct LookaheadLayer {
        MetalCompiledGraph* source=nullptr;uint64_t generation=0;
        std::vector<MetalMatrix> matrices;
        std::vector<LookaheadInstruction> program;
        size_t input=0,output=0,key=SIZE_MAX,value=SIZE_MAX;
    };
    struct LookaheadWorkspace {
        explicit LookaheadWorkspace(const LookaheadLayout& p):layout(p){}
        LookaheadLayout layout;
        std::vector<LookaheadLayer> layers;LookaheadLayer head;
        std::array<std::unique_ptr<LookaheadRowOwner>,4> owners;
        MetalMatrix input,positions,active,probabilities,predictions,packedHidden,packedPredictions,accepted,finalHidden;
        std::array<MetalMatrix,2> hidden;
        AffineInt4Mapping embedding;
        struct Dependency {std::filesystem::path path;std::filesystem::file_time_type modified;uintmax_t bytes;};
        std::vector<Dependency> dependencies;
        std::vector<float> hostHidden,hostPositions,hostActive,hostAccepted;
        size_t preparedBytes=0;std::string stats="{}";
    };
    std::unique_ptr<LookaheadWorkspace> lookahead_;
    LookaheadLayer CompileLookaheadLayer(MetalCompiledGraph&,LookaheadWorkspace&,size_t,bool);
    void EncodeLookaheadLayer(LookaheadLayer&,LookaheadWorkspace&,const MetalMatrix&,
        const std::array<MetalMatrix,2>*,const std::vector<LookaheadLayout::View>&);
    void ResetLookaheadRow(size_t row);
#endif
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
    struct DiagnosticBulkLayer {MetalCompiledGraph* graph=nullptr;uint64_t generation=0;MetalCompiledGraph::DiagnosticFfn ffn;MetalCompiledGraph::DiagnosticKV kv;MetalCompiledGraph::DiagnosticQO qo;std::vector<size_t> controlIds;std::vector<std::vector<MetalMatrix>> controls;};
    struct DiagnosticBulkGroup {MetalMatrix residual,norm,gate,up,down,key,value;std::array<MetalMatrix,2> tape;};
    struct DiagnosticLayerBulk {X::Value graphs;size_t chunks=0;std::vector<DiagnosticBulkGroup> groups;std::vector<DiagnosticBulkLayer> layers;std::array<MetalMatrix,2> tape;std::array<std::vector<MetalMatrix>,2> tapeViews;MetalMatrix residual,norm,gate,up,silu,down,hidden,lastHidden,key,value;std::vector<MetalMatrix> residualViews,keyViews,valueViews,normViews;uint64_t workspaceBytes=0;};
    std::unique_ptr<DiagnosticLayerBulk> diagnosticLayerBulk_;
    std::string diagnosticLayerBulkStats_="{}";
    bool diagnosticBulkKVEnabled_=false;
    bool diagnosticBulkQOEnabled_=false;
#endif

#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::shared_ptr<MetalPrefillScratch> prefillScratch_;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    std::unique_ptr<PrefixCache> tokenCache_;
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
    uint64_t tokenPrefixComparedBytes_=0,tokenPrefixCopiedBytes_=0;
    uint64_t tokenPrefixReusedBlocks_=0,tokenPrefixNewBlocks_=0;
#endif
    PrefixCacheIdentity tokenIdentity_;
    size_t tokenCacheBudget_=0,tokenCapacity_=0,tokenVocabulary_=0;
    std::vector<std::vector<int64_t>> rowTokens_;
    std::vector<uint8_t> rowTokensValid_;
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    std::vector<PrefixPrecisionHistory> tokenPrecisionHistory_;
    uint64_t tokenPrefillSkippedMixed_=0;
#endif
    std::unordered_map<MetalCompiledGraph*,std::pair<size_t,uint64_t>> tokenGraphs_;
    std::vector<X::Value> tokenGraphOwners_;
    void ValidateTokenOwners(std::vector<std::unique_lock<std::mutex>>&);
    size_t reconcileCalls_=0,reconcileOld_=0,reconcileLCP_=0,reconcileKept_=0,reconcileZeroBytes_=0;
    bool reconcileReplay_=false,reconcileHiddenPending_=false;
    double reconcileGpuMs_=0;
#endif
    std::vector<X::Value> owners_;
    std::vector<State> states_;
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
    struct CompactSpan {std::vector<State> states;MetalMatrix finalHidden,output;};
    struct CompactFamily {
        X::Value graphs,fallback;
        size_t rows=0,chunk=0,width=0,capacity=0,fallbackChunk=0;
        std::vector<uint64_t> generations,fallbackGenerations,baseGenerations;
        std::vector<CompactSpan> spans;
        MetalMatrix zero,hidden;
        std::vector<int64_t> shape,physicalShape;
    };
    std::map<std::pair<size_t,size_t>,CompactFamily> compactFamilies_;
    MetalMatrix compactDecodeHidden_;
    CompactFamily* compactRunning_=nullptr;
    size_t compactFirst_=0,compactLastRows_=0,compactLastFirst_=4;
    bool compactSealed_=false;
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
    bool CompactVerificationPreparationSupported() const;
#endif
    bool TryCompactPrefill(X::Value,X::Value,X::Value,X::Value,X::Value,int64_t,
        const std::vector<uint64_t>*,const std::vector<int64_t>*,VerificationRun*,X::Value&);
#endif
    size_t RunPhysicalRow(size_t row) const {
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
        return compactRunning_?compactFirst_+row:row;
#else
        return row;
#endif
    }
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
    std::map<std::pair<size_t,size_t>,std::array<MetalMatrix,2>> prefillHiddenPool_;
    size_t hiddenPoolBindings_=0;
    void PrepareHiddenPool(const MetalCompiledGraph&);
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    struct MTPEmbedding {
        SafeTensorsIndex index;
        MlxQwen35Checkpoint adapter;
        AffineInt4Mapping mapping;
        struct Dependency {std::filesystem::path path;std::filesystem::file_time_type modified;uintmax_t bytes;};
        std::vector<Dependency> dependencies;
    };
    struct MTPCapture {
        MetalMatrix flat,normalized,packed,active;
        std::vector<float> flags;
    };
    struct MTPProfile {
        X::Value owner;
        MetalCompiledGraph* graph=nullptr;
        uint64_t generation=0;
        std::array<MetalMatrix,7> inputs;
    };
    struct MTPWorkspace {
        X::Value graphOwner,headOwner;
        MetalCompiledGraph *graph=nullptr,*head=nullptr;
        uint64_t generation=0,headGeneration=0;
        size_t batch=0,chunk=0,width=0,heads=0,dim=0,capacity=0,currentCapture=0;
        MetalMatrix normWeight,lastTarget,lastHidden,prediction,baseHidden,basePrediction,backupSlots,backupActive;
        std::array<MetalMatrix,7> inputs;
        std::array<MetalMatrix,2> cache,backup;
        std::map<size_t,MTPCapture> captures;
        std::map<size_t,MTPProfile> profiles;
        std::map<size_t,MetalMatrix> slices,zeros;
        std::vector<size_t> committed,tentative;
        std::vector<size_t> targetLengths;
        std::vector<uint8_t> targetPredictionReady;
        std::vector<float> backupFlags;
        std::vector<float> nextPositions;
        std::vector<uint8_t> predictionReady,baseReady;
        std::unique_ptr<MTPEmbedding> embedding;
        MetalMatrix validation;
        float epsilon=0,offset=0;
        bool pending=false;
        size_t teacherCalls=0,draftCalls=0,restoreCalls=0,headCalls=0,validationCalls=0;
        std::string stats="{}";
        std::string generationStats="{}";
    };
    std::unique_ptr<MTPWorkspace> mtp_;
    void ResetMTP();
    void ResetMTPRow(size_t);
    void ValidateMTPTarget(size_t);
    void EncodeMTPTarget(const MetalMatrix&,size_t,bool verification);
    void CommitMTPTarget(const MetalMatrix&,const std::vector<float>&,size_t);
    X::Value MTPStep(X::Value,X::Value,X::Value,X::Value,int64_t,X::Value,bool,bool predict=true);
    X::Value MTPStepUnlocked(X::Value,X::Value,X::Value,X::Value,int64_t,X::Value,bool,bool);
    X::Value MTPEmbed(const std::vector<int64_t>&,size_t);
    X::Value MTPGenerateUnlocked(X::Value,X::Value,X::Value,X::Value,bool);
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
    struct VerificationWorkspace {
        X::Value graphs,headOwner;
        MetalCompiledGraph* head=nullptr;
        uint64_t headGeneration=0;
        size_t batch=0,chunk=0,width=0,vocab=0;
        std::vector<uint64_t> generations,sourceGenerations;
        std::vector<MetalCompiledGraph::VerificationBindings> layers;
        MetalMatrix allHidden,flatHidden,predictions,packedPredictions,acceptedMask,replayMask,finalHidden;
        size_t preparedMetalBytes=0;
    };
    struct VerificationRun {
        VerificationWorkspace* plan=nullptr;
        std::vector<int64_t> tokens,stops,remaining,consumed;
        std::vector<float> active,accepted,replay;
        std::vector<std::vector<int64_t>> emitted;
        std::vector<bool> eos;
        std::vector<MetalCompiledGraph::VerificationBindings> bindings;
#ifdef GARNET_METAL_TARGET_JACOBI
        std::vector<float> jacobiSlots;
        std::vector<uint64_t> jacobiEpochs;
#endif
    };
    std::unique_ptr<VerificationWorkspace> verification_;
    std::string verificationStats_="{}";
#ifdef GARNET_METAL_TARGET_JACOBI
    TargetJacobiCandidates targetJacobi_;
    bool targetJacobiEnabled_=false;
#endif
    void ValidateVerificationRun(VerificationRun&,const std::vector<MetalCompiledGraph*>&);
    void EncodeVerificationBackups(VerificationRun&,const std::vector<MetalCompiledGraph*>&);
    void EncodeVerificationHead(VerificationRun&,const MetalMatrix&);
    void CommitVerification(VerificationRun&,const std::vector<MetalCompiledGraph*>&);
    X::Value VerifyGreedyUnlocked(X::Value,X::Value,X::Value,X::Value,X::Value,X::Value,X::Value,X::Value);
#endif
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    std::vector<float> resetZeroSource_;
#endif
    size_t reusedSourceBytes_=0,reusedCopyBytes_=0,reusedCopyStates_=0;
    MetalCompiledGraph* head_=nullptr;
    uint64_t headGeneration_=0;
    MetalMatrix hidden_;
    MetalMatrix prediction_;
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
    MetalMatrix confidence_;
    void EncodePredictionConfidence(const std::vector<size_t>* selectedRows);
    X::Value PredictionConfidenceValues(const std::vector<float>&,const std::vector<uint8_t>&);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    MetalMatrix rowFinalHidden_;
    size_t rowFinalWidth_=0;
    std::vector<uint8_t> finalHiddenReady_;
#endif
    std::vector<int64_t> hiddenShape_;
    size_t kvInplaceUpdates_=0;
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
    size_t linearStateInplaceUpdates_=0;
#endif
    size_t joinedPrefillPairs_=0;
    size_t constantHeadDispatches_=0;
    size_t singleCoderHeadDispatches_=0;
    size_t residualNormDispatches_=0;
    size_t packedPrefillNormDispatches_=0;
    size_t globalHalfAlphaDispatches_=0,wideBatchPrefillDispatches_=0,interleavedBodyDispatches_=0,coalescedBlockDispatches_=0,mlpGraphDispatches_=0;
    size_t joinedGlobalMlpBlocks_=0;
    size_t prefillPartialSubmissions_=0;
    size_t fusedHeadArgmaxDispatches_=0;
    size_t screenedHeadDispatches_=0;
    size_t screenedHeadSeedRows_=0;
    size_t interleavedScreenDispatches_=0;
    size_t bodyShapeDispatches_=0,bodyShapeMisses_=0;
    size_t deferredZeroBytes_=0,deferredZeroStates_=0;
    size_t dirtyPrefixResetBytes_=0,dirtyPrefixResetStates_=0,dirtyPrefixResetFallbackStates_=0;
    bool poisoned_=false;
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    // Reset/release must not reopen compilation of this initialized engine.
    bool wideVerificationPreparationClosed_=false;
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
    bool predictionInitialized_=false;
    size_t rowResetBytes_=0;
#endif
    double inferenceMs_=0,prepareMs_=0,encodeMs_=0,gpuMs_=0,waitMs_=0;
    std::mutex mutex_;
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    std::unique_ptr<RequestRowLeases> requestRows_;
    bool managedRows_=false;
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    struct PrefixSnapshot {size_t tokens=0,bytes=0;uint64_t lastUse=0;std::vector<std::array<std::vector<float>,2>> layers;};
    std::map<int64_t,PrefixSnapshot> prefixCache_;
    size_t prefixBytes_=0,prefixRestoreBytes_=0,prefixCaptureBytes_=0;
    uint64_t prefixClock_=0;bool prefixFreshReset_=false;
    void ValidatePrefixOwners(std::vector<std::unique_lock<std::mutex>>& locks);
#endif
#ifdef GARNET_METAL_FUSED_Q4_DECODE
    void PrepareFusedDecodeProfiles(const MetalCompiledGraph&);
#endif
    void Clear();
    void ResetUnlocked();
    X::Value RunUnlocked(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken,
        const std::vector<uint64_t>* epochs=nullptr,const std::vector<int64_t>* tokenIds=nullptr,
        VerificationRun* verification=nullptr
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
        ,bool confidence=false
#endif
        );
#ifdef GARNET_METAL_REQUEST_ROWS
    void ResetRowUnlocked(int64_t row);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    X::Value PredictRowsUnlocked(X::Value
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
        ,bool confidence=false
#endif
        );
#endif
public:
    bool PrepareGpuCounters();
#ifdef GARNET_METAL_SILU_PROFILE32
    bool SetSiluProfile32Enabled(bool enabled);
    size_t LastSiluProfile32Dispatches(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastSiluProfile32Dispatches():0;}
#endif
#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL
    bool SetPairedPackedPrefillEnabled(bool enabled);
    size_t LastPairedPackedPrefillDispatches(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastPairedPackedPrefillDispatches():0;}
#endif
#ifdef GARNET_METAL_VALUE_PROFILE32
    bool SetValueProfile32Enabled(bool enabled);
    size_t LastValueProfile32Dispatches(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastValueProfile32Dispatches():0;}
#endif
#ifdef GARNET_METAL_SPLIT_DECODE_SCORES
    bool SetSplitScoresEnabled(bool enabled);
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
    bool SetProfile32ScoresEnabled(bool enabled);
    size_t LastProfile32ScoreDispatches(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastProfile32ScoreDispatches():0;}
#endif
    size_t LastSplitScoreDispatches(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastSplitScoreDispatches():0;}
#endif
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
    bool ConfigurePredictionConfidence();
    X::Value PredictRowsConfidence(X::Value epochs);
    X::Value RunTokensConfidence(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,
        X::Value active,int64_t predictionToken,X::Value epochs,X::Value tokens);
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
    bool PrepareMTP(X::Value graph,X::Value head,X::Value sourceChunks);
    bool PrepareMTPEmbedding(const std::string& checkpoint);
    bool PrepareMTPProfiles(X::Value);
    X::Value MTPTeacher(X::Value embedding,X::Value positions,X::Value slots,X::Value active,int64_t sourceOffset,X::Value epochs);
    X::Value MTPDraft(X::Value embedding,X::Value positions,X::Value slots,X::Value active,X::Value epochs);
    X::Value MTPTeacherCache(X::Value embedding,X::Value positions,X::Value slots,X::Value active,int64_t sourceOffset,X::Value epochs);
    X::Value MTPGenerate(X::Value known,X::Value remaining,X::Value stops,X::Value epochs);
    X::Value MTPGenerateScheduled(X::Value known,X::Value remaining,X::Value stops,X::Value epochs);
    std::string MTPGenerationStatsJson();
    X::Value MTPSnapshot();
    std::string MTPStatsJson();
#endif
#ifdef GARNET_METAL_AFFINE_WEIGHT_RESIDENCY
    bool PrepareAffineResidency(int64_t budget);
    std::string AffineResidencyStatsJson();
#endif
    bool Build(X::Value graphs,X::Value head);
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
    bool PrepareCompactPrefill(X::Value graphs,X::Value fallback);
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    size_t SelectPreparedPrefillChunk(const std::vector<size_t>& rows,const std::vector<uint64_t>& epochs,
        const std::vector<size_t>& remaining,size_t maximum,size_t budget);
    size_t MaximumPreparedPrefillChunk();
#endif
    size_t CompactPrefillFamilyCount(){std::lock_guard<std::mutex> lock(mutex_);return compactFamilies_.size();}
    size_t LastCompactPrefillRows(){std::lock_guard<std::mutex> lock(mutex_);return compactLastRows_;}
    size_t LastCompactPrefillFirst(){std::lock_guard<std::mutex> lock(mutex_);return compactLastFirst_;}
#endif
    X::Value Run(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken);
    void Reset();
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
    bool PrepareDiagnosticLayerBulk(X::Value graphs,int64_t chunks);
    bool SetDiagnosticBulkKV(bool enabled);
    bool SetDiagnosticBulkQO(bool enabled);
    X::Value RunDiagnosticLayerBulk(X::Value hiddenChunks,X::Value controls,bool enabled);
#if defined(GARNET_METAL_TOKEN_PREFIX_CACHE) && defined(GARNET_METAL_CONTINUOUS_ROWS)
    bool ConfigureTokenTracking(X::Value identity){return ConfigureTokenPrefix(identity,0,0);}
    int64_t DiagnosticBulkChunkBound(){std::lock_guard<std::mutex> lock(mutex_);return gpu_&&!poisoned_&&tokenCache_&&diagnosticLayerBulk_?int64_t(diagnosticLayerBulk_->chunks):0;}
    X::Value RunDiagnosticLeasedBulk(X::Value hiddenChunks,X::Value controls,X::Value epochs,X::Value tokenChunks,bool enabled);
    X::Value DiagnosticFinalHidden();
#endif
    std::string DiagnosticLayerBulkStatsJson(){std::lock_guard<std::mutex> lock(mutex_);return diagnosticLayerBulkStats_;}
#endif

#ifdef GARNET_METAL_VERIFY_ATTENTION
    bool PrepareVerification(X::Value graphs,X::Value head);
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    bool PrepareLookahead(int64_t window,int64_t levels,int64_t guesses);
    X::Value LookaheadGenerate(X::Value known,X::Value slots,X::Value remaining,X::Value stops,X::Value epochs);
    std::string LookaheadStatsJson(){std::lock_guard<std::mutex> lock(mutex_);return lookahead_?lookahead_->stats:"{}";}
#endif
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    bool WideVerificationPreparationOpen(){std::lock_guard<std::mutex> lock(mutex_);return gpu_&&!poisoned_&&!wideVerificationPreparationClosed_;}
#endif
    X::Value VerifyGreedy(X::Value hidden,X::Value positions,X::Value slots,X::Value active,
        X::Value tokens,X::Value stops,X::Value remaining,X::Value epochs);
    std::string VerificationStatsJson();
#ifdef GARNET_METAL_TARGET_JACOBI
    bool SetTargetJacobiEnabled(bool enabled);
    std::string TargetJacobiDraftsJson(X::Value known,X::Value slots,X::Value budgets,X::Value epochs);
    std::string TargetJacobiStatsJson();
#endif
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    std::array<size_t,2> VerificationGeometry();
#endif
#endif
#if defined(GARNET_METAL_QWEN38_WIDE_PREFILL)||defined(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS)
    bool PreparePrefill(X::Value graph,int64_t layer);
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
    void ResetRow(int64_t row);
    size_t LastRowResetBytes() const {return rowResetBytes_;}
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
    X::Value ClaimRequestRow(int64_t request);
    bool ReleaseRequestRow(int64_t row,int64_t epoch);
    std::pair<size_t,size_t> RequestRowLayout();
    bool RequestRowMatches(size_t row,uint64_t epoch);
    X::Value RunLeased(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken,X::Value epochs);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    X::Value PredictRows(X::Value epochs);
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    bool ConfigureTokenPrefix(X::Value identity,int64_t maxEntries,int64_t maxBytes);
    X::Value RunTokens(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,
        X::Value active,int64_t predictionToken,X::Value epochs,X::Value tokens);
    bool CacheTokenPrefix(int64_t row,int64_t epoch);
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
    int64_t TrackedPrefillPrefixTokens(int64_t row,int64_t epoch);
#endif
    // Internal scheduler maintenance; no Xlang/APISET binding.
    bool TouchTokenPrefix(int64_t row,int64_t epoch,X::Value prompt);
    int64_t RestoreTokenPrefix(int64_t row,int64_t epoch,X::Value tokens);
    int64_t TrackedTokenCount(int64_t row,int64_t epoch);
    int64_t ReconcileDenseTokens(int64_t row,int64_t epoch,X::Value tokens);
    std::string DenseReconcileStatsJson();
    std::string TokenPrefixStatsJson();
    bool ClearTokenPrefixes();
#endif
    int64_t CapturePrefix(int64_t tokens);
    int64_t RestorePrefix(int64_t id);
    bool ClearPrefixCache();
    size_t PrefixCacheBytes();
    size_t PrefixCacheEntries();
    size_t PrefixRestoreBytes();
    size_t PrefixCaptureBytes();
    std::vector<MetalResourceReference> MemoryReferences();
    X::Value Snapshot(int64_t layer);
    X::Value Hidden();
    void Close();
#ifdef GARNET_METAL_MPP_AFFINE
    size_t LastMPPAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastMPPAffineDispatches():0;}
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
    size_t LastIndexedAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastIndexedAffineDispatches():0;}
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    size_t LastReuse64AffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastReuse64AffineDispatches():0;}
#endif
#ifdef GARNET_METAL_MISTRAL_PACKED_PREFILL
    size_t LastPackedAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastPackedAffineDispatches():0;}
    size_t PackedAffinePlanCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->PackedAffinePlanCount():0;}
    bool PrefillSetPackedAffine(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_||poisoned_)throw std::logic_error("Ready session required for packed prefill selection");
        gpu_->SetPackedAffineEnabled(enabled);return true;
    }
#endif
#ifdef GARNET_METAL_MISTRAL_STREAMED_PREFILL
    size_t LastStreamedAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastStreamedAffineDispatches():0;}
    size_t StreamedAffinePlanCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->StreamedAffinePlanCount():0;}
    bool PrefillSetStreamedAffine(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_||poisoned_)throw std::logic_error("Ready session required for streamed prefill selection");
        gpu_->SetStreamedAffineEnabled(enabled);return true;
    }
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
    size_t LastReuse128AffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastReuse128AffineDispatches():0;}
#ifdef GARNET_METAL_VERIFY_ATTENTION
    bool VerificationSetAffineReuse128(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_)throw std::logic_error("Closed decoder session has no affine verification route");
        gpu_->VerificationSetAffineReuse128(enabled);return true;
    }
#endif
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
    bool VerificationSetMistralLayout(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_||poisoned_)throw std::logic_error("Ready decoder session required for verification layout selection");
        gpu_->VerificationSetMistralLayout(enabled);return true;
    }
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
    bool VerificationSetValueVector4(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!gpu_||poisoned_)throw std::logic_error("Live decoder session required for value vector selection");
        gpu_->VerificationSetValueVector4(enabled);return true;
    }
#endif
    size_t LastChunkProbabilityReuseDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkProbabilityReuseDispatches():0;}
    size_t LastChunkAttentionIndex32Dispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionIndex32Dispatches():0;}
    size_t LastChunkKVScatterDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkKVScatterDispatches():0;}
    size_t LastChunkKVHoistedCommits() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkKVHoistedCommits():0;}
    size_t LastHalfAttentionDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastHalfAttentionDispatches():0;}
    size_t HalfAttentionScratchBytes() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->HalfAttentionScratchBytes():0;}
    size_t LastCommittedKVAttentionDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastCommittedKVAttentionDispatches():0;}
    size_t LastLayoutVectorDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastLayoutVectorDispatches():0;}
    size_t LastChunkAttentionMPSDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionMPSDispatches():0;}
    size_t LastChunkFlashAttentionDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkFlashAttentionDispatches():0;}
    size_t LastChunkAttentionMPSMatrices() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionMPSMatrices():0;}
    size_t LastChunkAttentionMPSSkippedMatrices() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionMPSSkippedMatrices():0;}
    size_t ChunkAttentionMPSPlanCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->ChunkAttentionMPSPlanCount():0;}
    size_t ChunkAttentionMPSScratchBytes() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->ChunkAttentionMPSScratchBytes():0;}
    size_t ChunkAttentionMPSBucketCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->ChunkAttentionMPSBucketCount():0;}
    size_t LastChunkAttentionBucketDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionBucketDispatches():0;}
    size_t LastChunkAttentionTailDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionTailDispatches():0;}
    size_t LastChunkAttentionBucketCapacity() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionBucketCapacity():0;}
    size_t LastChunkAttentionUnknownControls() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastChunkAttentionUnknownControls():0;}
    size_t LastKVInplaceUpdates() const {return kvInplaceUpdates_;}
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
    size_t LastLinearStateInplaceUpdates() const {return linearStateInplaceUpdates_;}
#endif
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
    size_t LastHiddenPoolBindings() const {return hiddenPoolBindings_;}
#endif
#ifdef GARNET_METAL_INDEXED_DECODE
    size_t LastIndexedDecodeAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastIndexedDecodeAffineDispatches():0;}
    size_t LastIndexedDecodeFusedClearDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastIndexedDecodeFusedClearDispatches():0;}
    size_t LastIndexedDecodeBlitClears() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastIndexedDecodeBlitClears():0;}
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
    bool SetIndexedDecodeFusedClearEnabled(bool enabled){std::lock_guard<std::mutex> lock(mutex_);if(!gpu_||poisoned_)throw std::logic_error("Ready native session required");gpu_->SetIndexedDecodeFusedClearEnabled(enabled);return true;}
#endif
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
    size_t LastAffineProfileDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineProfileDispatches():0;}
    size_t LastAffineProfileMisses() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineProfileMisses():0;}
    size_t LastAffineDecodeBlockDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineDecodeBlockDispatches():0;}
    size_t AffineProfileIndexBits() {return MetalContext::AffineProfileIndexBits();}
    size_t LastAffineProfileMaxThreads() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineProfileMaxThreads():0;}
    size_t AffineProfilePipelineCount() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->AffineProfilePipelineCount():0;}
#endif
    size_t LastJoinedPrefillPairs() const {return joinedPrefillPairs_;}
    size_t LastConstantHeadDispatches() const {return constantHeadDispatches_;}
    size_t LastSingleCoderHeadDispatches() const {return singleCoderHeadDispatches_;}
    size_t LastFusedHeadArgmaxDispatches() const {return fusedHeadArgmaxDispatches_;}
    size_t LastScreenedHeadDispatches() const {return screenedHeadDispatches_;}
    size_t LastScreenedHeadSeedRows() const {return screenedHeadSeedRows_;}
    size_t LastInterleavedScreenDispatches() const {return interleavedScreenDispatches_;}
    size_t LastBodyShapeDispatches() const {return bodyShapeDispatches_;}
    size_t LastBodyShapeMisses() const {return bodyShapeMisses_;}
    size_t BodyShapePipelineCount() const {return gpu_?gpu_->BodyShapePipelineCount():0;}
    size_t LastDeferredZeroBytes() const {return deferredZeroBytes_;}
    size_t LastDeferredZeroStates() const {return deferredZeroStates_;}
    size_t ReusedZeroSourceBytes() const {return reusedSourceBytes_;}
    size_t LastReusedZeroCopyBytes() const {return reusedCopyBytes_;}
    size_t LastReusedZeroCopyStates() const {return reusedCopyStates_;}
    size_t LastResidualNormDispatches() const {return residualNormDispatches_;}
    size_t LastPackedPrefillNormDispatches() const {return packedPrefillNormDispatches_;}
    size_t LastInterleavedBodyDispatches() const {return interleavedBodyDispatches_;}
    size_t LastMlpGraphDispatches() const {return mlpGraphDispatches_;}
    size_t LastCoalescedBlockDispatches() const {return coalescedBlockDispatches_;}
    size_t LastWideBatchPrefillDispatches() const {return wideBatchPrefillDispatches_;}
    size_t LastGlobalHalfAlphaDispatches() const {return globalHalfAlphaDispatches_;}
    size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}
    size_t LastPrefillPartialSubmissions() const {return prefillPartialSubmissions_;}
    size_t LastDirtyPrefixResetBytes() const {return dirtyPrefixResetBytes_;}
    size_t LastDirtyPrefixResetStates() const {return dirtyPrefixResetStates_;}
    size_t LastDirtyPrefixResetFallbackStates() const {return dirtyPrefixResetFallbackStates_;}
    double LastInferenceMs() const {return inferenceMs_;}
    double PrepareMs() const {return prepareMs_;}
    double EncodeMs() const {return encodeMs_;}
    double GPUMs() const {return gpuMs_;}
    double WaitMs() const {return waitMs_;}
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
    size_t FusedMlpBlocks(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastFusedMistralMlpBlocks():0;}
#endif
#ifdef GARNET_METAL_FUSED_Q4_DECODE
    size_t FusedDecodeBlocks(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastFusedMistralDecodeBlocks():0;}
    size_t FusedDecodeInputCopies(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastFusedMistralDecodeInputCopies():0;}
    size_t FusedDecodeProfiles(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->FusedMistralDecodeProfileCount():0;}
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    size_t FusedWideDecodeBlocks(){std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastFusedMistralWideDecodeBlocks():0;}
    bool SetFusedWideDecodeEnabled(bool enabled){std::lock_guard<std::mutex> lock(mutex_);if(!gpu_||poisoned_)throw std::logic_error("Ready native session required");gpu_->SetFusedMistralWideDecodeEnabled(enabled);return true;}
#endif
#endif
    BEGIN_PACKAGE(MetalDecoderSession)
#ifdef GARNET_METAL_FUSED_Q4_DECODE
        APISET().AddProp("fused_decode_blocks",&MetalDecoderSession::FusedDecodeBlocks);
        APISET().AddProp("fused_decode_input_copies",&MetalDecoderSession::FusedDecodeInputCopies);
        APISET().AddProp("fused_decode_profiles",&MetalDecoderSession::FusedDecodeProfiles);
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
        APISET().AddProp("fused_wide_decode_blocks",&MetalDecoderSession::FusedWideDecodeBlocks);
        APISET().AddFunc<1>("fused_wide_decode_set_enabled",&MetalDecoderSession::SetFusedWideDecodeEnabled);
#endif
#endif
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
        APISET().AddProp("fused_mlp_blocks",&MetalDecoderSession::FusedMlpBlocks);
#endif
        APISET().AddFunc<0>("prepare_gpu_counters",&MetalDecoderSession::PrepareGpuCounters);
#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL
        APISET().AddFunc<1>("paired_prefill_set_enabled",&MetalDecoderSession::SetPairedPackedPrefillEnabled);
        APISET().AddProp("paired_prefill_dispatches",&MetalDecoderSession::LastPairedPackedPrefillDispatches);
#endif
#ifdef GARNET_METAL_SILU_PROFILE32
        APISET().AddFunc<1>("silu_profile32_set_enabled",&MetalDecoderSession::SetSiluProfile32Enabled);
        APISET().AddProp("silu_profile32_dispatches",&MetalDecoderSession::LastSiluProfile32Dispatches);
#endif
#ifdef GARNET_METAL_VALUE_PROFILE32
        APISET().AddFunc<1>("value_profile32_set_enabled",&MetalDecoderSession::SetValueProfile32Enabled);
        APISET().AddProp("value_profile32_dispatches",&MetalDecoderSession::LastValueProfile32Dispatches);
#endif
#ifdef GARNET_METAL_SPLIT_DECODE_SCORES
        APISET().AddFunc<1>("split_score_set_enabled",&MetalDecoderSession::SetSplitScoresEnabled);
        APISET().AddProp("split_score_dispatches",&MetalDecoderSession::LastSplitScoreDispatches);
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
        APISET().AddFunc<1>("split_score_profile32_set_enabled",&MetalDecoderSession::SetProfile32ScoresEnabled);
        APISET().AddProp("split_score_profile32_dispatches",&MetalDecoderSession::LastProfile32ScoreDispatches);
#endif
#endif
#ifdef GARNET_METAL_AFFINE_WEIGHT_RESIDENCY
        APISET().AddFunc<1>("prepare_affine_residency",&MetalDecoderSession::PrepareAffineResidency);
        APISET().AddFunc<0>("affine_residency_stats_json",&MetalDecoderSession::AffineResidencyStatsJson);
#endif
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
        APISET().AddFunc<0>("configure_prediction_confidence",&MetalDecoderSession::ConfigurePredictionConfidence);
        APISET().AddFunc<1>("predict_rows_confidence",&MetalDecoderSession::PredictRowsConfidence);
        APISET().AddFunc<8>("run_tokens_confidence",&MetalDecoderSession::RunTokensConfidence);
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
        APISET().AddFunc<3>("prepare_mtp",&MetalDecoderSession::PrepareMTP);
        APISET().AddFunc<1>("prepare_mtp_embedding",&MetalDecoderSession::PrepareMTPEmbedding);
        APISET().AddFunc<1>("prepare_mtp_profiles",&MetalDecoderSession::PrepareMTPProfiles);
        APISET().AddFunc<6>("mtp_teacher",&MetalDecoderSession::MTPTeacher);
        APISET().AddFunc<5>("mtp_draft",&MetalDecoderSession::MTPDraft);
        APISET().AddFunc<6>("mtp_teacher_cache",&MetalDecoderSession::MTPTeacherCache);
        APISET().AddFunc<4>("mtp_generate",&MetalDecoderSession::MTPGenerate);
        APISET().AddFunc<0>("mtp_generation_stats_json",&MetalDecoderSession::MTPGenerationStatsJson);
        APISET().AddFunc<0>("mtp_snapshot",&MetalDecoderSession::MTPSnapshot);
        APISET().AddFunc<0>("mtp_stats_json",&MetalDecoderSession::MTPStatsJson);
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
        APISET().AddFunc<2>("prepare_verification",&MetalDecoderSession::PrepareVerification);
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
        APISET().AddFunc<3>("prepare_lookahead",&MetalDecoderSession::PrepareLookahead);
        APISET().AddFunc<5>("lookahead_generate",&MetalDecoderSession::LookaheadGenerate);
        APISET().AddFunc<0>("lookahead_stats_json",&MetalDecoderSession::LookaheadStatsJson);
#endif
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
        APISET().AddProp("wide_verification_preparation_open",&MetalDecoderSession::WideVerificationPreparationOpen);
#endif
        APISET().AddFunc<8>("verify_greedy",&MetalDecoderSession::VerifyGreedy);
        APISET().AddFunc<0>("verification_stats_json",&MetalDecoderSession::VerificationStatsJson);
#ifdef GARNET_METAL_TARGET_JACOBI
        APISET().AddFunc<1>("target_jacobi_set_enabled",&MetalDecoderSession::SetTargetJacobiEnabled);
        APISET().AddFunc<4>("target_jacobi_drafts_json",&MetalDecoderSession::TargetJacobiDraftsJson);
        APISET().AddFunc<0>("target_jacobi_stats_json",&MetalDecoderSession::TargetJacobiStatsJson);
#endif
#endif
        APISET().AddFunc<2>("build",&MetalDecoderSession::Build);
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
        APISET().AddFunc<2>("prepare_compact_prefill",&MetalDecoderSession::PrepareCompactPrefill);
        APISET().AddProp("compact_prefill_family_count",&MetalDecoderSession::CompactPrefillFamilyCount);
        APISET().AddProp("compact_prefill_last_rows",&MetalDecoderSession::LastCompactPrefillRows);
        APISET().AddProp("compact_prefill_last_first",&MetalDecoderSession::LastCompactPrefillFirst);
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
        APISET().AddProp("compact_prefill_max_chunk",&MetalDecoderSession::MaximumPreparedPrefillChunk);
#endif
#endif
        APISET().AddFunc<6>("run",&MetalDecoderSession::Run);
        APISET().AddFunc<0>("reset",&MetalDecoderSession::Reset);
#if defined(GARNET_METAL_QWEN38_WIDE_PREFILL)||defined(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS)
        APISET().AddFunc<2>("prepare_prefill",&MetalDecoderSession::PreparePrefill);
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
        APISET().AddFunc<2>("diagnostic_layer_bulk_prepare",&MetalDecoderSession::PrepareDiagnosticLayerBulk);
        APISET().AddFunc<1>("diagnostic_bulk_kv_set_enabled",&MetalDecoderSession::SetDiagnosticBulkKV);
        APISET().AddFunc<1>("diagnostic_bulk_qo_set_enabled",&MetalDecoderSession::SetDiagnosticBulkQO);
        APISET().AddFunc<3>("diagnostic_layer_bulk_run",&MetalDecoderSession::RunDiagnosticLayerBulk);
#if defined(GARNET_METAL_TOKEN_PREFIX_CACHE) && defined(GARNET_METAL_CONTINUOUS_ROWS)
        APISET().AddFunc<1>("configure_token_tracking",&MetalDecoderSession::ConfigureTokenTracking);
        APISET().AddFunc<5>("diagnostic_leased_bulk_run",&MetalDecoderSession::RunDiagnosticLeasedBulk);
        APISET().AddFunc<0>("diagnostic_final_hidden",&MetalDecoderSession::DiagnosticFinalHidden);
        APISET().AddFunc<2>("diagnostic_tracked_token_count",&MetalDecoderSession::TrackedTokenCount);
#endif
        APISET().AddFunc<0>("diagnostic_layer_bulk_stats_json",&MetalDecoderSession::DiagnosticLayerBulkStatsJson);
#endif
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
        APISET().AddFunc<1>("reset_row",&MetalDecoderSession::ResetRow);
        APISET().AddProp("row_reset_bytes",&MetalDecoderSession::LastRowResetBytes);
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        APISET().AddFunc<1>("claim_request_row",&MetalDecoderSession::ClaimRequestRow);
        APISET().AddFunc<2>("release_request_row",&MetalDecoderSession::ReleaseRequestRow);
        APISET().AddFunc<7>("run_leased",&MetalDecoderSession::RunLeased);
#endif
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
        APISET().AddFunc<1>("predict_rows",&MetalDecoderSession::PredictRows);
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        APISET().AddFunc<3>("configure_token_prefix",&MetalDecoderSession::ConfigureTokenPrefix);
        APISET().AddFunc<8>("run_tokens",&MetalDecoderSession::RunTokens);
        APISET().AddFunc<2>("cache_token_prefix",&MetalDecoderSession::CacheTokenPrefix);
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        APISET().AddFunc<2>("tracked_prefill_prefix_tokens",&MetalDecoderSession::TrackedPrefillPrefixTokens);
#endif
        APISET().AddFunc<3>("restore_token_prefix",&MetalDecoderSession::RestoreTokenPrefix);
        APISET().AddFunc<3>("reconcile_dense_tokens",&MetalDecoderSession::ReconcileDenseTokens);
        APISET().AddFunc<0>("dense_reconcile_stats_json",&MetalDecoderSession::DenseReconcileStatsJson);
        APISET().AddFunc<0>("token_prefix_stats_json",&MetalDecoderSession::TokenPrefixStatsJson);
        APISET().AddFunc<0>("clear_token_prefixes",&MetalDecoderSession::ClearTokenPrefixes);
#endif
        APISET().AddFunc<1>("capture_prefix",&MetalDecoderSession::CapturePrefix);
        APISET().AddFunc<1>("restore_prefix",&MetalDecoderSession::RestorePrefix);
        APISET().AddFunc<0>("clear_prefix_cache",&MetalDecoderSession::ClearPrefixCache);
        APISET().AddProp("prefix_cache_bytes",&MetalDecoderSession::PrefixCacheBytes);
        APISET().AddProp("prefix_cache_entries",&MetalDecoderSession::PrefixCacheEntries);
        APISET().AddProp("prefix_restore_bytes",&MetalDecoderSession::PrefixRestoreBytes);
        APISET().AddProp("prefix_capture_bytes",&MetalDecoderSession::PrefixCaptureBytes);
        APISET().AddFunc<1>("snapshot",&MetalDecoderSession::Snapshot);
        APISET().AddFunc<0>("hidden",&MetalDecoderSession::Hidden);
        APISET().AddFunc<0>("close",&MetalDecoderSession::Close);
#ifdef GARNET_METAL_MPP_AFFINE
        APISET().AddProp("mpp_affine_dispatches",&MetalDecoderSession::LastMPPAffineDispatches);
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
        APISET().AddProp("indexed_affine_dispatches",&MetalDecoderSession::LastIndexedAffineDispatches);
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
        APISET().AddProp("mpp_reuse64_dispatches",&MetalDecoderSession::LastReuse64AffineDispatches);
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
        APISET().AddProp("mpp_reuse128_dispatches",&MetalDecoderSession::LastReuse128AffineDispatches);
#ifdef GARNET_METAL_MISTRAL_PACKED_PREFILL
        APISET().AddProp("packed_prefill_dispatches",&MetalDecoderSession::LastPackedAffineDispatches);
        APISET().AddProp("packed_prefill_plans",&MetalDecoderSession::PackedAffinePlanCount);
        APISET().AddFunc<1>("prefill_set_packed_affine",&MetalDecoderSession::PrefillSetPackedAffine);
#endif
#ifdef GARNET_METAL_MISTRAL_STREAMED_PREFILL
        APISET().AddProp("streamed_prefill_dispatches",&MetalDecoderSession::LastStreamedAffineDispatches);
        APISET().AddProp("streamed_prefill_plans",&MetalDecoderSession::StreamedAffinePlanCount);
        APISET().AddFunc<1>("prefill_set_streamed_affine",&MetalDecoderSession::PrefillSetStreamedAffine);
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
        APISET().AddFunc<1>("verification_set_affine_reuse128",&MetalDecoderSession::VerificationSetAffineReuse128);
#endif
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
        APISET().AddFunc<1>("verification_set_mistral_layout",&MetalDecoderSession::VerificationSetMistralLayout);
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
        APISET().AddFunc<1>("verification_set_value_vector4",&MetalDecoderSession::VerificationSetValueVector4);
#endif
        APISET().AddProp("chunk_probability_reuse_dispatches",&MetalDecoderSession::LastChunkProbabilityReuseDispatches);
        APISET().AddProp("chunk_attention_index32_dispatches",&MetalDecoderSession::LastChunkAttentionIndex32Dispatches);
        APISET().AddProp("chunk_kv_scatter_dispatches",&MetalDecoderSession::LastChunkKVScatterDispatches);
        APISET().AddProp("chunk_kv_hoisted_commits",&MetalDecoderSession::LastChunkKVHoistedCommits);
        APISET().AddProp("chunk_committed_kv_attention_dispatches",&MetalDecoderSession::LastCommittedKVAttentionDispatches);
        APISET().AddProp("chunk_half_attention_dispatches",&MetalDecoderSession::LastHalfAttentionDispatches);
        APISET().AddProp("chunk_half_attention_scratch_bytes",&MetalDecoderSession::HalfAttentionScratchBytes);
        APISET().AddProp("layout_vector_dispatches",&MetalDecoderSession::LastLayoutVectorDispatches);
        APISET().AddProp("chunk_attention_mps_dispatches",&MetalDecoderSession::LastChunkAttentionMPSDispatches);
        APISET().AddProp("chunk_flash_attention_dispatches",&MetalDecoderSession::LastChunkFlashAttentionDispatches);
        APISET().AddProp("chunk_attention_mps_matrices",&MetalDecoderSession::LastChunkAttentionMPSMatrices);
        APISET().AddProp("chunk_attention_mps_skipped_matrices",&MetalDecoderSession::LastChunkAttentionMPSSkippedMatrices);
        APISET().AddProp("chunk_attention_mps_plan_count",&MetalDecoderSession::ChunkAttentionMPSPlanCount);
        APISET().AddProp("chunk_attention_mps_scratch_bytes",&MetalDecoderSession::ChunkAttentionMPSScratchBytes);
        APISET().AddProp("chunk_attention_mps_bucket_count",&MetalDecoderSession::ChunkAttentionMPSBucketCount);
        APISET().AddProp("chunk_attention_bucket_dispatches",&MetalDecoderSession::LastChunkAttentionBucketDispatches);
        APISET().AddProp("chunk_attention_tail_dispatches",&MetalDecoderSession::LastChunkAttentionTailDispatches);
        APISET().AddProp("chunk_attention_bucket_capacity",&MetalDecoderSession::LastChunkAttentionBucketCapacity);
        APISET().AddProp("chunk_attention_unknown_controls",&MetalDecoderSession::LastChunkAttentionUnknownControls);
        APISET().AddProp("kv_inplace_updates",&MetalDecoderSession::LastKVInplaceUpdates);
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
        APISET().AddProp("linear_state_inplace_updates",&MetalDecoderSession::LastLinearStateInplaceUpdates);
#endif
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
        APISET().AddProp("prefill_hidden_pool_bindings",&MetalDecoderSession::LastHiddenPoolBindings);
#endif
#ifdef GARNET_METAL_INDEXED_DECODE
        APISET().AddProp("indexed_decode_affine_dispatches",&MetalDecoderSession::LastIndexedDecodeAffineDispatches);
        APISET().AddProp("indexed_decode_fused_clear_dispatches",&MetalDecoderSession::LastIndexedDecodeFusedClearDispatches);
        APISET().AddProp("indexed_decode_blit_clears",&MetalDecoderSession::LastIndexedDecodeBlitClears);
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
        APISET().AddFunc<1>("indexed_decode_fused_clear_set_enabled",&MetalDecoderSession::SetIndexedDecodeFusedClearEnabled);
#endif
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
        APISET().AddProp("affine_profile_dispatches",&MetalDecoderSession::LastAffineProfileDispatches);
        APISET().AddProp("affine_profile_misses",&MetalDecoderSession::LastAffineProfileMisses);
        APISET().AddProp("affine_decode_block_dispatches",&MetalDecoderSession::LastAffineDecodeBlockDispatches);
        APISET().AddProp("affine_profile_index_bits",&MetalDecoderSession::AffineProfileIndexBits);
        APISET().AddProp("affine_profile_max_threads",&MetalDecoderSession::LastAffineProfileMaxThreads);
        APISET().AddProp("affine_profile_pipeline_count",&MetalDecoderSession::AffineProfilePipelineCount);
#endif
        APISET().AddProp("joined_prefill_pairs",&MetalDecoderSession::LastJoinedPrefillPairs);
        APISET().AddProp("constant_head_dispatches",&MetalDecoderSession::LastConstantHeadDispatches);
        APISET().AddProp("single_coder_head_dispatches",&MetalDecoderSession::LastSingleCoderHeadDispatches);
        APISET().AddProp("fused_head_argmax_dispatches",&MetalDecoderSession::LastFusedHeadArgmaxDispatches);
        APISET().AddProp("screened_head_dispatches",&MetalDecoderSession::LastScreenedHeadDispatches);
        APISET().AddProp("screened_head_seed_rows",&MetalDecoderSession::LastScreenedHeadSeedRows);
        APISET().AddProp("interleaved_screen_dispatches",&MetalDecoderSession::LastInterleavedScreenDispatches);
        APISET().AddProp("body_shape_dispatches",&MetalDecoderSession::LastBodyShapeDispatches);
        APISET().AddProp("body_shape_misses",&MetalDecoderSession::LastBodyShapeMisses);
        APISET().AddProp("body_shape_pipeline_count",&MetalDecoderSession::BodyShapePipelineCount);
        APISET().AddProp("deferred_zero_bytes",&MetalDecoderSession::LastDeferredZeroBytes);
        APISET().AddProp("deferred_zero_states",&MetalDecoderSession::LastDeferredZeroStates);
        APISET().AddProp("reused_zero_source_bytes",&MetalDecoderSession::ReusedZeroSourceBytes);
        APISET().AddProp("reused_zero_copy_bytes",&MetalDecoderSession::LastReusedZeroCopyBytes);
        APISET().AddProp("reused_zero_copy_states",&MetalDecoderSession::LastReusedZeroCopyStates);
        APISET().AddProp("residual_norm_dispatches",&MetalDecoderSession::LastResidualNormDispatches);
        APISET().AddProp("packed_prefill_norm_dispatches",&MetalDecoderSession::LastPackedPrefillNormDispatches);
        APISET().AddProp("interleaved_body_dispatches",&MetalDecoderSession::LastInterleavedBodyDispatches);
        APISET().AddProp("coalesced_block_dispatches",&MetalDecoderSession::LastCoalescedBlockDispatches);
        APISET().AddProp("mlp_prefill_graph_dispatches",&MetalDecoderSession::LastMlpGraphDispatches);
        APISET().AddProp("wide_batch_prefill_dispatches",&MetalDecoderSession::LastWideBatchPrefillDispatches);
        APISET().AddProp("global_half_alpha_dispatches",&MetalDecoderSession::LastGlobalHalfAlphaDispatches);
        APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);
        APISET().AddProp("prefill_partial_submissions",&MetalDecoderSession::LastPrefillPartialSubmissions);
        APISET().AddProp("dirty_prefix_reset_bytes",&MetalDecoderSession::LastDirtyPrefixResetBytes);
        APISET().AddProp("dirty_prefix_reset_states",&MetalDecoderSession::LastDirtyPrefixResetStates);
        APISET().AddProp("dirty_prefix_reset_fallback_states",&MetalDecoderSession::LastDirtyPrefixResetFallbackStates);
        APISET().AddProp("prepare_ms",&MetalDecoderSession::PrepareMs);
        APISET().AddProp("encode_ms",&MetalDecoderSession::EncodeMs);
        APISET().AddProp("gpu_ms",&MetalDecoderSession::GPUMs);
        APISET().AddProp("wait_ms",&MetalDecoderSession::WaitMs);
        APISET().AddProp("last_inference_ms",&MetalDecoderSession::LastInferenceMs);
    END_PACKAGE
};
}
