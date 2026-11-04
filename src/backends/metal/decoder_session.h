// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "compiled_graph.h"
#include <array>
#include <map>
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
#include "../../runtime/kv/prefix_cache.h"
#endif
#ifdef GARNET_METAL_CONTINUOUS_ROWS
#include "../../runtime/scheduler/request_row_leases.h"
#endif
namespace Garnet {
// One synchronous request context. Graph owners are retained; request state is
// private to this session, including when graphs are reused by another session.
class MetalDecoderSession {
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
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
    std::shared_ptr<MetalPrefillScratch> prefillScratch_;
#endif
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
    std::unique_ptr<PrefixCache> tokenCache_;
    PrefixCacheIdentity tokenIdentity_;
    size_t tokenCacheBudget_=0,tokenCapacity_=0;
    std::vector<std::vector<int64_t>> rowTokens_;
    std::vector<uint8_t> rowTokensValid_;
    std::unordered_map<MetalCompiledGraph*,std::pair<size_t,uint64_t>> tokenGraphs_;
    std::vector<X::Value> tokenGraphOwners_;
    void ValidateTokenOwners(std::vector<std::unique_lock<std::mutex>>&);
#endif
    std::vector<X::Value> owners_;
    std::vector<State> states_;
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    std::vector<float> resetZeroSource_;
#endif
    size_t reusedSourceBytes_=0,reusedCopyBytes_=0,reusedCopyStates_=0;
    MetalCompiledGraph* head_=nullptr;
    uint64_t headGeneration_=0;
    MetalMatrix hidden_;
    MetalMatrix prediction_;
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    MetalMatrix rowFinalHidden_;
    size_t rowFinalWidth_=0;
    std::vector<uint8_t> finalHiddenReady_;
#endif
    std::vector<int64_t> hiddenShape_;
    size_t kvInplaceUpdates_=0;
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
    void Clear();
    void ResetUnlocked();
    X::Value RunUnlocked(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken,
        const std::vector<uint64_t>* epochs=nullptr,const std::vector<int64_t>* tokenIds=nullptr);
#ifdef GARNET_METAL_REQUEST_ROWS
    void ResetRowUnlocked(int64_t row);
#endif
public:
    bool Build(X::Value graphs,X::Value head);
    X::Value Run(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken);
    void Reset();
#ifdef GARNET_METAL_QWEN38_WIDE_PREFILL
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
    int64_t RestoreTokenPrefix(int64_t row,int64_t epoch,X::Value tokens);
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
    size_t LastKVInplaceUpdates() const {return kvInplaceUpdates_;}
#ifdef GARNET_METAL_INDEXED_DECODE
    size_t LastIndexedDecodeAffineDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastIndexedDecodeAffineDispatches():0;}
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
    size_t LastAffineProfileDispatches() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineProfileDispatches():0;}
    size_t LastAffineProfileMisses() {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->LastAffineProfileMisses():0;}
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
    BEGIN_PACKAGE(MetalDecoderSession)
        APISET().AddFunc<2>("build",&MetalDecoderSession::Build);
        APISET().AddFunc<6>("run",&MetalDecoderSession::Run);
        APISET().AddFunc<0>("reset",&MetalDecoderSession::Reset);
#ifdef GARNET_METAL_QWEN38_WIDE_PREFILL
        APISET().AddFunc<2>("prepare_prefill",&MetalDecoderSession::PreparePrefill);
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
        APISET().AddFunc<3>("restore_token_prefix",&MetalDecoderSession::RestoreTokenPrefix);
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
        APISET().AddProp("kv_inplace_updates",&MetalDecoderSession::LastKVInplaceUpdates);
#ifdef GARNET_METAL_INDEXED_DECODE
        APISET().AddProp("indexed_decode_affine_dispatches",&MetalDecoderSession::LastIndexedDecodeAffineDispatches);
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
        APISET().AddProp("affine_profile_dispatches",&MetalDecoderSession::LastAffineProfileDispatches);
        APISET().AddProp("affine_profile_misses",&MetalDecoderSession::LastAffineProfileMisses);
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
