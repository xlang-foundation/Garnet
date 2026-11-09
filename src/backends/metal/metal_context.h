// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#ifdef GARNET_METAL_LOOKAHEAD_ATTENTION
#include "lookahead_layout.h"
#endif
namespace Garnet {
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
struct DensePrefixBlock;
struct DensePrefixCapture;
#endif
#ifdef GARNET_METAL_COMPACT_ATTENTION_INDEX
inline constexpr size_t kMetalAttentionChunkCapacityLimit=4096;
#if defined(GARNET_METAL_QWEN38_LARGE_PREFILL)||(defined(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS)&&GARNET_METAL_DENSE_PREFILL_MAX_TOKENS>127)
inline constexpr size_t kMetalAttentionIndexBytes=sizeof(int16_t);
#else
inline constexpr size_t kMetalAttentionIndexBytes=1;
#endif
#else
inline constexpr size_t kMetalAttentionChunkCapacityLimit=2048;
inline constexpr size_t kMetalAttentionIndexBytes=sizeof(int32_t);
#endif
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
static_assert(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==16||GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==32||
    GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==64||GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==128||
    GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==256||GARNET_METAL_DENSE_PREFILL_MAX_TOKENS==512,"Invalid dense prefill profile");
inline constexpr size_t kMetalPrefillChunkLimit=GARNET_METAL_DENSE_PREFILL_MAX_TOKENS;
inline constexpr size_t kMetalPrefillPackedHiddenLimit=kMetalPrefillChunkLimit*4096;
#elif defined(GARNET_METAL_QWEN38_WIDE_PREFILL)
#ifdef GARNET_METAL_QWEN38_LARGE_PREFILL
inline constexpr size_t kMetalPrefillChunkLimit=256;
#else
inline constexpr size_t kMetalPrefillChunkLimit=64;
#endif
inline constexpr size_t kMetalPrefillPackedHiddenLimit=kMetalPrefillChunkLimit*5120;
#elif defined(GARNET_METAL_QWEN35_EXTENDED_PREFILL)
inline constexpr size_t kMetalPrefillChunkLimit=80;
inline constexpr size_t kMetalPrefillPackedHiddenLimit=80*1024;
#else
inline constexpr size_t kMetalPrefillChunkLimit=64;
#ifdef GARNET_METAL_QWEN35_4B_WIDE_PREFILL
inline constexpr size_t kMetalPrefillPackedHiddenLimit=64*2560;
#else
inline constexpr size_t kMetalPrefillPackedHiddenLimit=65536;
#endif
#endif
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
inline constexpr size_t kMetalAttentionChunkTokenLimit=GARNET_METAL_DENSE_PREFILL_MAX_TOKENS;
#elif defined(GARNET_METAL_QWEN38_LARGE_PREFILL)
inline constexpr size_t kMetalAttentionChunkTokenLimit=256;
#else
inline constexpr size_t kMetalAttentionChunkTokenLimit=127;
#endif
class MetalMatrix {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit MetalMatrix(std::shared_ptr<Impl> impl): impl_(std::move(impl)) {}
    friend class MetalContext;
public:
    MetalMatrix() = default;
    bool Valid() const {return bool(impl_);}
    bool SharesStorage(const MetalMatrix&) const;
    bool SameView(const MetalMatrix& other) const {return impl_&&impl_==other.impl_;}
    bool CompatibleLayout(const MetalMatrix&) const;
};
#ifdef GARNET_METAL_REQUEST_ROWS
struct MetalRowClearRange {
    MetalMatrix matrix;
    size_t row=0,groups=0,capacity=0,width=0,begin=0,end=0;
};
#endif
class MetalBF16Weight {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit MetalBF16Weight(std::shared_ptr<Impl> impl):impl_(std::move(impl)) {}
    friend class MetalContext;
public:
    MetalBF16Weight()=default;
    bool Valid() const {return bool(impl_);}
    bool PrimaryReady() const;
    bool ExactHalfReady() const;
    bool GlobalHalfReady() const;
    bool InterleavedBodyReady() const;
    bool CoalescedBlockReady() const;
    size_t CoalescedBlockCount() const;
    size_t CoalescedCompactCount() const;
};
#ifdef GARNET_METAL_MLX_AFFINE_INT4
class MetalAffineInt4Weight {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit MetalAffineInt4Weight(std::shared_ptr<Impl> impl):impl_(std::move(impl)) {}
    friend class MetalContext;
public:
    MetalAffineInt4Weight()=default;
    bool Valid() const {return bool(impl_);}
};
#endif
// C++ interface keeps Objective-C ownership out of model-facing headers.
struct MetalScreenedDiagnostics {
    size_t rows=0,columns=0;
    std::vector<float> maskedLogits;
    std::vector<uint32_t> kept;
    std::vector<float> querySafe;
};
struct MetalBufferInfo {
    uintptr_t identity=0;uint64_t allocatedBytes=0,lengthBytes=0;std::string representation;
};
struct MetalResourceReference {
    MetalBufferInfo buffer;size_t slot=0;std::string role,name;
    std::vector<int64_t> shape;bool residentOperand=false,anyWriter=false,residentWriter=false,input=false,outputPart=false,legacyOutput=false,weight=false;
};
struct MetalMemoryStats {
    uint64_t allocated=0,recommended=0,registry=0,physical=0,resident=0,compressed=0,internal=0,virtualBytes=0;
    int taskStatus=-1;
};
struct MetalPrimaryStats {
 uint64_t activeWeights=0,originalBytes=0,payloadBytes=0,metadataBytes=0,allocatedBytes=0,uploads=0,decodeEncoded=0,prefillEncoded=0,headEncoded=0,vectorDecodeEncoded=0,vectorHeadEncoded=0;
};
struct MetalProfileOwnerStats {size_t matrices=0,affineWeights=0,bf16Weights=0,pipelineSets=0;};
class MetalContext {
    friend class MetalCompiledGraph;
    friend struct MetalCommittedAttentionReference;
    friend struct MetalHalfAttentionReference;
    void EncodeAttentionChunkInternal(const std::vector<MetalMatrix>&,const MetalMatrix&,size_t,size_t,size_t,size_t,size_t,bool,bool,bool,bool,bool halfKV=false);
#ifdef GARNET_METAL_COMMITTED_KV_MPS
    void EncodeCommittedAttentionChunk(const std::vector<MetalMatrix>&,const MetalMatrix&,size_t,size_t,size_t,size_t,size_t,bool);
#endif
#ifdef GARNET_METAL_CHUNK_KV_COMMIT_HOIST
    void NoteChunkKVCommitHoist();
#endif
#ifdef GARNET_METAL_CHUNK_HALF_ATTENTION
    void EncodeHalfAttentionChunk(const std::vector<MetalMatrix>&,const MetalMatrix&,size_t,size_t,size_t,size_t,size_t,bool);
    std::array<MetalMatrix,3> HalfAttentionScratchForTest() const;
    uint32_t HalfAttentionPartialStatusForTest() const;
#endif
    struct Impl;
    std::unique_ptr<Impl> impl_;
#ifdef GARNET_METAL_MISTRAL_PACKED_PREFILL
    bool TryEncodePackedAffine(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,const std::vector<uint32_t>&,bool reportIndexed);
#endif
#ifdef GARNET_METAL_MISTRAL_STREAMED_PREFILL
    bool TryEncodeStreamedAffine(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,const std::vector<uint32_t>&,bool reportIndexed);
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
    void EncodeAffineInt4IndexedImpl(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalMatrix&,const std::vector<size_t>&,bool reportIndexed,bool allowReuse128=true);
#endif
    void EncodePointwise(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&,
        unsigned operation, float epsilon, float offset);
    void EncodeStateKernel(unsigned operation,const std::vector<MetalMatrix>&,
        const MetalMatrix&,const void* parameters,size_t parameterBytes,size_t threads,const MetalMatrix* secondOutput=nullptr,
        bool referenceAttention=false,const MetalMatrix* thirdOutput=nullptr,bool scatterKV=false);
#ifdef GARNET_METAL_VERIFY_ATTENTION
    void ValidateVerificationRecurrent(const std::vector<MetalMatrix>&,
        const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,size_t stateInput=SIZE_MAX) const;
    struct VerificationKVSpanParams {
        uint64_t batch,chunk,heads,dim,capacity,cacheStride,slotStride,
            submittedStride,acceptedStride,backupStride;
    };
    VerificationKVSpanParams ValidateVerificationKVSpan(const MetalMatrix& cache,
        const MetalMatrix& slots,const MetalMatrix& submitted,const MetalMatrix* accepted,
        const MetalMatrix& backup,size_t heads,size_t dim,size_t capacity,size_t chunk) const;
#endif
    std::vector<uint32_t> PrimaryReadWordsInternal(const MetalBF16Weight&,size_t start,size_t count,bool vector);
#ifdef GARNET_METAL_PREPARED_KV_ROW_SPANS
    friend class MetalDecoderSession;
    friend struct MetalPreparedKVRowsTestAccess;
    // Native preparation only. No XLang view API: scheduler/session integration
    // must retain physical ownership/epoch validation around these cache views.
    MetalMatrix PrepareKVRowSpan(const MetalMatrix&,size_t first,size_t rows,
        size_t heads,size_t dim,size_t capacity) const;
#endif
public:
    static MetalPrimaryStats PrimaryStats();
    std::vector<uint32_t> PrimaryReadWords(const MetalBF16Weight&,size_t start,size_t count);
    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
    static void PrimaryTestFailAfter(int stage);
#endif
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    std::string DeviceName() const;
    static std::string AvailableDeviceName();
    static MetalMemoryStats MemoryStats();
    static MetalProfileOwnerStats ProfileOwnerStats();
    // Correctness diagnostic only; deliberately queries live MPS properties.
    static bool MatrixMetadataMatchesLive(const MetalMatrix&);
    static MetalBufferInfo MatrixBufferInfo(const MetalMatrix&);
    bool IsDefaultLayout(const MetalMatrix&,size_t rows,size_t columns) const;
    static std::vector<MetalBufferInfo> WeightBufferInfo(const MetalBF16Weight&);
#ifdef GARNET_METAL_MLX_AFFINE_INT4
    static std::vector<MetalBufferInfo> AffineInt4BufferInfo(const MetalAffineInt4Weight&);
    MetalAffineInt4Weight UploadAffineInt4(const void* packed,size_t packedBytes,
        const void* scales,size_t scaleBytes,const void* biases,size_t biasBytes,
        size_t rows,size_t columns,bool coefficientsFP16=false);
    void EncodeAffineInt4MatMul(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&);
    // Preparation only: compile model dimensions/strides for every supported
    // route up to maximumRows (1..512). Actual rows remain dynamic. Encoding
    // never compiles; missing/unsupported profiles use the original pipelines.
#ifdef GARNET_METAL_MISTRAL_STREAMED_PREFILL
    // Prepared up-to512 profile: currently only512 active rows use MPS; all
    // shorter counts retain existing affine kernels. One scratch set/context.
    bool PrepareStreamedAffineProfile(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,size_t maximumRows);
    void SetStreamedAffineEnabled(bool enabled);
    size_t LastStreamedAffineDispatches() const;
    size_t StreamedAffinePlanCount() const;
    std::array<MetalMatrix,3> StreamedAffineScratch() const;
#endif
#ifdef GARNET_METAL_MISTRAL_PACKED_PREFILL
    bool PreparePackedAffineProfile(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,size_t maximumRows);
    void SetPackedAffineEnabled(bool enabled);
    // Research-only projection amortization. Prepared dense up-to8192 rows;
    // not an inference/session/serving interface.
    bool PrepareDiagnosticBulkAffine(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&);
    void EncodeDiagnosticBulkAffine(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&);
    size_t LastDiagnosticBulkDispatches() const;
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
    void EncodeDiagnosticSiluHalf(const MetalMatrix&,const MetalMatrix&,const MetalMatrix& halfCarrier);
    void EncodeDiagnosticHalfCastReference(const MetalMatrix&,const MetalMatrix& halfCarrier);
    void EncodeDiagnosticBulkGatedDown(const MetalMatrix& gate,const MetalMatrix& up,const MetalAffineInt4Weight&,const MetalMatrix&);
    size_t LastDiagnosticSiluHalfDispatches() const;
    MetalMatrix PrepareDiagnosticDenseView(const MetalMatrix&,size_t firstElement,size_t rows,size_t columns) const;
#endif
    std::array<MetalMatrix,2> DiagnosticBulkScratch() const;

    size_t LastPackedAffineDispatches() const;
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
    size_t LastFusedMistralMlpBlocks() const;
    MetalMatrix FusedMistralMlpScratch() const;
    bool TryEncodeFusedMistralMlp(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalAffineInt4Weight&,
        const MetalAffineInt4Weight&,const MetalMatrix& gateLayout,const MetalMatrix& upLayout,
        const MetalMatrix& activatedLayout,const MetalMatrix& output,const std::vector<size_t>* selectedRows=nullptr);
#endif

#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL
    void SetPairedPackedPrefillEnabled(bool enabled);
    size_t LastPairedPackedPrefillDispatches() const;
    // Returns false without encoding for unsupported profiles/layouts. Both
    // outputs remain materialized; no graph operator or SiLU semantics change.
    bool TryEncodePairedPackedPrefill(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalAffineInt4Weight&,const MetalMatrix&,const MetalMatrix&,
        const std::vector<size_t>* selectedRows=nullptr);
#endif
    size_t PackedAffinePlanCount() const;
    MetalMatrix PackedAffineScratch() const;
#endif
#ifdef GARNET_METAL_FUSED_Q4_DECODE
    bool PrepareFusedMistralDecode(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalAffineInt4Weight&,const MetalMatrix&);
    bool TryEncodeFusedMistralDecode(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalAffineInt4Weight&,const MetalMatrix&,const std::vector<size_t>* selectedRows=nullptr);
    size_t FusedMistralDecodeProfileCount() const;
    size_t LastFusedMistralDecodeBlocks() const;
    size_t LastFusedMistralDecodeInputCopies() const;
    MetalMatrix FusedMistralDecodeScratch() const;
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    void SetFusedMistralWideDecodeEnabled(bool enabled);
    bool FusedMistralWideDecodeEnabled() const;
    size_t LastFusedMistralWideDecodeBlocks() const;
    MetalMatrix FusedMistralWideDecodeScratch() const;
#endif
#endif
    bool PrepareAffineInt4Profile(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalMatrix&,size_t maximumRows);
    size_t AffineProfilePipelineCount() const;
    size_t LastAffineProfileDispatches() const;
    size_t LastAffineProfileMisses() const;
    size_t LastAffineDecodeBlockDispatches() const;
    size_t LastAffineProfileMaxThreads() const;
    static size_t AffineProfileIndexBits();
#ifdef GARNET_METAL_INDEXED_DECODE
    void EncodeAffineInt4IndexedDecode(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalMatrix&,const std::vector<size_t>& selectedRows);
    bool HasIndexedDecodeAffine() const;
    size_t LastIndexedDecodeAffineDispatches() const;
    // Optional prepared physical-B4 pipeline; runtime toggles require a drained
    // context. Generic/padded/empty/full selections retain established routes.
    void SetIndexedDecodeFusedClearEnabled(bool enabled);
    size_t LastIndexedDecodeFusedClearDispatches() const;
    size_t LastIndexedDecodeBlitClears() const;
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
    // Experimental strict BF16 indexed prefill. Does not change graph dispatch.
    // Selected rows must be unique. Empty selection explicitly zeros output.
    void EncodeAffineInt4IndexedMatMul(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalMatrix&,const std::vector<size_t>& selectedRows);
    bool HasIndexedAffine() const;
    size_t LastIndexedAffineDispatches() const;
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    bool HasReuse64Affine() const;
    size_t LastReuse64AffineDispatches() const;
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
    bool HasReuse128Affine() const;
    size_t LastReuse128AffineDispatches() const;
#ifdef GARNET_METAL_VERIFY_ATTENTION
    // Diagnostic selection between two already-prepared routes; closed batches only.
    void VerificationSetAffineReuse128(bool enabled);
#endif
    // Operator-control entry point; preserves the existing64-row policy.
    void EncodeAffineInt4IndexedMatMulControl(const MetalMatrix&,const MetalAffineInt4Weight&,
        const MetalMatrix&,const std::vector<size_t>& selectedRows);
#endif
#endif
    MetalMatrix Upload(const std::vector<float>& values, size_t rows, size_t columns);
    MetalMatrix Allocate(size_t rows, size_t columns);
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
    void EncodeLastActiveHidden(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,size_t chunk);
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
    void ValidateCompletedRow(const MetalMatrix&,size_t row) const;
    void ZeroRow(const MetalMatrix&,size_t row);
    // Validate every range, then clear in one synchronous GPU blit batch.
    size_t ZeroRowRanges(const std::vector<MetalRowClearRange>&);
    std::vector<float> ReadRow(const MetalMatrix&,size_t row) const;
    void WriteRow(const MetalMatrix&,size_t row,const std::vector<float>&);
    // Compact one request's dense KV prefix. Recurrent/convolution state must
    // use complete ReadRow snapshots at the same boundary. Caller clears the
    // destination row before installing a prefix, so future slots stay empty.
    std::vector<float> ReadRowPrefixGroups(const MetalMatrix&,size_t row,size_t groups,size_t capacity,size_t width,size_t tokens) const;
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
    DensePrefixCapture ReadRowPrefixBlocks(const MetalMatrix&,size_t row,size_t groups,size_t capacity,size_t width,size_t tokens,
        size_t commonTokens,const std::vector<std::shared_ptr<const DensePrefixBlock>>& candidates) const;
#endif
    void WriteRowPrefixGroups(const MetalMatrix&,size_t row,size_t groups,size_t capacity,size_t width,size_t tokens,const std::vector<float>&);
#endif
    MetalBF16Weight UploadBF16(const void* data,size_t bytes,size_t rows,size_t columns,bool floatPrefill=false,bool visionPrefill=false);
#ifdef GARNET_METAL_MPP_AFFINE
    bool HasMPPAffine() const;
    size_t LastMPPAffineDispatches() const;
#endif
    bool PrepareScreenedHead(const MetalBF16Weight&);
    bool EncodeScreenedArgmax(const MetalMatrix&,const MetalBF16Weight&,const MetalMatrix&);
    size_t LastScreenedHeadDispatches() const;
    size_t LastScreenedHeadSeedRows() const;
    size_t LastInterleavedScreenDispatches() const;
    // Diagnostics contain private masked logits, never public complete-logit results.
    MetalScreenedDiagnostics ScreenedDiagnostics() const;
    // Preserve BF16 checkpoint values; use SIMD matvec for decode and FP32 tiled GEMM for up to 128 prompt rows.
    void EncodeBF16MatVec(const MetalMatrix&,const MetalBF16Weight&,const MetalMatrix&);
    // Compile bounded body-only shape pipelines during graph initialization.
    // Unsupported profiles retain the original dispatch path.
    bool PrepareBF16BodyShape(const MetalMatrix&,const MetalBF16Weight&,const MetalMatrix&);
    size_t BodyShapePipelineCount() const;
    size_t LastBodyShapeDispatches() const;
    size_t LastBodyShapeMisses() const;
    // Try the bounded exact-half prefill pair; false leaves both outputs untouched.
    bool EncodeJoinedBF16Prefill(const MetalMatrix&,const MetalBF16Weight&,
        const MetalBF16Weight&,const MetalMatrix&,const MetalMatrix&);
    size_t LastJoinedPrefillPairs() const;
    size_t LastConstantHeadDispatches() const;
    size_t LastSingleCoderHeadDispatches() const;
    size_t LastResidualNormDispatches() const;
    size_t LastPackedPrefillNormDispatches() const;
    size_t LastGlobalHalfAlphaDispatches() const;
    size_t LastWideBatchPrefillDispatches() const;
    size_t LastInterleavedBodyDispatches() const;
    size_t LastCoalescedBlockDispatches() const;
    size_t LastVisionGlobalAlphaDispatches() const;
    size_t LastVisionParallelDispatches() const;
    size_t LastVisionPrefetchDispatches() const;
    size_t LastVisionVectorDispatches() const;
    // False preserves all three independent outputs. Decode never takes this route.
    bool EncodeJoinedGlobalMlp(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,
        const MetalMatrix& gate,const MetalMatrix& up,const MetalMatrix& activated);
    size_t LastJoinedGlobalMlpBlocks() const;
    // Optional prefill-only graph. May flush prior producers on the same queue.
    // False leaves the command unchanged; output stays pending until final submit.
    bool EncodeMlpPrefillGraph(const MetalMatrix&,const MetalBF16Weight&,
        const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&);
    size_t LastMlpGraphDispatches() const;
    // Output is batch x [token index, nonfinite flag]; lowest index wins ties.
    void EncodeArgmax(const MetalMatrix&,const MetalMatrix&);
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
    void EncodePredictionConfidence(const MetalMatrix&,const MetalMatrix&);
#endif
    // Greedy-only BF16 projection. False encodes nothing; generic logits remain separate.
    // Optional full logits are emitted only for numerical validation.
    bool EncodeBF16Argmax(const MetalMatrix&,const MetalBF16Weight&,const MetalMatrix&,
        const MetalMatrix* debugLogits=nullptr);
    size_t LastFusedHeadArgmaxDispatches() const;
    // Synchronous host update into an existing allocation, between command batches.
    void Update(const MetalMatrix&,const std::vector<float>& values);
    void Update(const MetalMatrix&,const float* values,size_t count);
    // Clear logical elements of an existing shared allocation between batches.
    void Zero(const MetalMatrix&);
    // Owner snapshots are logically zero immediately; physical zeroing precedes
    // all consumers in the owner's next Begin. Host Update/Zero cancels the queue.
    void DeferZero(const MetalMatrix&);
    size_t LastDeferredZeroBytes() const;
    size_t LastDeferredZeroStates() const;
    // Caller proves the remaining logical elements are already positive zero.
    std::vector<float> ReadPrefixGroups(const MetalMatrix&,size_t groups,size_t capacity,size_t width,size_t tokens);
    void WritePrefixGroups(const MetalMatrix&,size_t groups,size_t capacity,size_t width,size_t tokens,const std::vector<float>&);
    void ZeroPrefixGroups(const MetalMatrix&,size_t groups,size_t capacity,size_t width,size_t dirtyTokens);
    void Begin();
#ifdef GARNET_METAL_LOOKAHEAD_ATTENTION
    void PrepareLookaheadAttention(const LookaheadLayout&);
    size_t LookaheadAttentionPipelineCount() const;
    size_t LastLookaheadAttentionDispatches() const;
    // Read-only canonical prefix and staged local K/V. Session callers must
    // validate actual live row leases before constructing these request views.
    void EncodeLookaheadAttention(const std::vector<MetalMatrix>& inputs,
        const MetalMatrix& probabilities,const MetalMatrix& output,
        const LookaheadLayout&,const std::vector<LookaheadLayout::View>&);
#endif
#ifdef GARNET_METAL_LOOKAHEAD_PROJECTIONS
    void PrepareLookaheadProjection(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,const LookaheadLayout&);
    void EncodeLookaheadProjection(const MetalMatrix&,const MetalAffineInt4Weight&,const MetalMatrix&,const LookaheadLayout&,const std::vector<LookaheadLayout::View>&);
    size_t LookaheadProjectionPipelineCount() const;
    size_t LastLookaheadProjectionDispatches() const;
#endif
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    void EncodeLookaheadKVCommit(const MetalMatrix& local,const MetalMatrix& canonical,const LookaheadLayout&,
        const std::vector<LookaheadLayout::View>&,const std::vector<LookaheadLayout::CommitSelection>&);
    size_t LastLookaheadKVCommitDispatches() const;
#endif
#if defined(GARNET_METAL_TARGET_JACOBI)||defined(GARNET_METAL_LOOKAHEAD_RUNTIME)
    void RequireDrainedProfileControl() const;
#endif
    // Commit an ordered partial batch and continue encoding without CPU wait.
    void Flush();
    size_t LastPartialSubmissions() const;
    void EncodeMatMul(const MetalMatrix& left, const MetalMatrix& right,
        const MetalMatrix& output, bool transposeRight = false);
    void EncodeRMSNorm(const MetalMatrix& input, const MetalMatrix& weight,
        const MetalMatrix& output, float epsilon, float weightOffset = 1.0f);
#ifdef GARNET_METAL_SILU_PROFILE32
    void SetSiluProfile32Enabled(bool enabled);
    size_t LastSiluProfile32Dispatches() const;
#endif
    void EncodeSiluMul(const MetalMatrix& gate, const MetalMatrix& up,
        const MetalMatrix& output);
    void EncodeAdd(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    void EncodeResidualRMSNorm(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix& residual,const MetalMatrix& normalized,float epsilon,float offset);
    void EncodePackedPrefillNorm(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix& packed,const MetalMatrix& reshaped,const MetalMatrix& normalized,float epsilon,float offset);
    void EncodeBias(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    void EncodeMaskRows(const MetalMatrix&, const MetalMatrix&, const MetalMatrix&);
    // Packed rows retain explicit caller-owned states after the output segment.
    void EncodeCausalConv(const MetalMatrix& token,const MetalMatrix& weight,
        const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& packed,size_t kernel);
    void EncodeCausalConvChunk(const MetalMatrix& tokens,const MetalMatrix& weight,
        const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& output,
        const MetalMatrix& nextState,size_t chunk,size_t kernel);
    void EncodeGatedDelta(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& value,
        const MetalMatrix& decay,const MetalMatrix& beta,const MetalMatrix& state,
        const MetalMatrix& active,const MetalMatrix& packed,size_t heads,size_t keyDim,size_t valueDim);
    void EncodeGatedDeltaSplit(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& value,
        const MetalMatrix& decay,const MetalMatrix& beta,const MetalMatrix& state,const MetalMatrix& active,
        const MetalMatrix& result,const MetalMatrix& nextState,size_t heads,size_t keyDim,size_t valueDim);
    void EncodeGatedDeltaChunk(const MetalMatrix& prepared,const MetalMatrix& state,const MetalMatrix& active,
        const MetalMatrix& output,const MetalMatrix& nextState,size_t chunk,size_t heads,size_t keyDim,size_t valueDim);
#ifdef GARNET_METAL_VERIFY_ATTENTION
    // Explicit caller-owned outputs include S1 after the first known token.
    // Controls are nonpending host-readable prefix masks; no allocation/readback
    // occurs during encoding. An entirely inactive row retains its input state.
    void EncodeVerificationConvCheckpoint(const MetalMatrix& tokens,const MetalMatrix& weight,
        const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& output,
        const MetalMatrix& nextState,const MetalMatrix& firstState,size_t chunk,size_t kernel);
    void EncodeVerificationDeltaCheckpoint(const MetalMatrix& prepared,const MetalMatrix& state,
        const MetalMatrix& active,const MetalMatrix& output,const MetalMatrix& nextState,
        const MetalMatrix& firstState,size_t chunk,size_t heads,size_t keyDim,size_t valueDim);
    size_t LastVerificationConvCheckpoints() const;
    size_t LastVerificationDeltaCheckpoints() const;
    // Backup before speculative writes. Restore only rejected submitted slots
    // into an initialized cache; all other cache words remain untouched.
    void EncodeVerificationKVBackup(const MetalMatrix& cache,const MetalMatrix& slots,
        const MetalMatrix& submitted,const MetalMatrix& backup,
        size_t heads,size_t dim,size_t capacity,size_t chunk);
    void EncodeVerificationKVRestore(const MetalMatrix& backup,const MetalMatrix& slots,
        const MetalMatrix& submitted,const MetalMatrix& accepted,const MetalMatrix& cache,
        size_t heads,size_t dim,size_t capacity,size_t chunk);
    size_t LastVerificationKVBackups() const;
    size_t LastVerificationKVRestores() const;
#endif
    void EncodeSliceGroups(const MetalMatrix&,const MetalMatrix&,size_t groups,size_t sourceWidth,size_t start,size_t count);
    // kernel=0 accepts convolution values without interleaved state fields.
    void EncodeLinearPrepare(const MetalMatrix& conv,const MetalMatrix& a,const MetalMatrix& b,
        const MetalMatrix& aLog,const MetalMatrix& dtBias,const MetalMatrix& output,
        size_t keyHeads,size_t valueHeads,size_t keyDim,size_t valueDim,size_t kernel);
    // Offset writes fill disjoint sequence segments; callers must fill every
    // segment before reading or consuming the combined output.
    void EncodeGatedNorm(const MetalMatrix& core,const MetalMatrix& z,const MetalMatrix& norm,
        const MetalMatrix& output,size_t heads,size_t valueDim,float epsilon,size_t columnOffset=0);
    void EncodeConcat(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,bool includeThird=true);
    void EncodeNormRope(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t heads,size_t dim,size_t rotary,float epsilon,float weightOffset=1,bool normalize=true);
    void EncodeKVUpdate(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t kvHeads,size_t dim,size_t capacity);
    void EncodeKVChunk(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t kvHeads,size_t dim,size_t capacity,size_t chunk,bool contiguousSlots=false);
    void EncodeAttentionChunk(const std::vector<MetalMatrix>&,const MetalMatrix&,size_t heads,size_t kvHeads,
        size_t dim,size_t capacity,size_t chunk,bool gated,bool reference=false,bool fullRequestRows=false);
#ifdef GARNET_METAL_VERIFY_ATTENTION
    // Explicitly prepared probability/output matrices; no GPU workspace allocation.
    // Inputs use chunk attention order: Q,K,V,oldK,oldV,slots,active,gate.
    // Host-readable, nonpending controls must describe a contiguous active
    // prefix in each row. The cache remains unchanged; commit KV separately.
    void EncodeVerificationAttention(const std::vector<MetalMatrix>&,const MetalMatrix& probabilities,
        const MetalMatrix& output,size_t heads,size_t kvHeads,size_t dim,size_t capacity,size_t chunk,bool gated);
    size_t LastVerificationAttentionDispatches() const;
    size_t LastVerificationLayoutDispatches() const;
    size_t LastVerificationSplitScoreDispatches() const;
    size_t LastVerificationGqaValueDispatches() const;
    void VerificationSetGqaValuesEnabled(bool enabled);
    size_t LastVerificationGqaScoreDispatches() const;
    void VerificationSetGqaScoresEnabled(bool enabled);
    void VerificationSetSplitScoresEnabled(bool enabled);
    size_t LastVerificationValueVectorDispatches() const;
    void VerificationSetValueVector4(bool enabled);
    void VerificationSetMistralLayout(bool enabled);
#endif
    size_t LastChunkProbabilityReuseDispatches() const;
    size_t LastChunkAttentionIndex32Dispatches() const;
    size_t LastChunkKVScatterDispatches() const;
    size_t LastChunkKVHoistedCommits() const;
    size_t LastCommittedKVAttentionDispatches() const;
    size_t LastHalfAttentionDispatches() const;
    size_t HalfAttentionScratchBytes() const;
#ifdef GARNET_METAL_CHUNK_KV_COMMIT_HOIST
    bool CanHoistKVControls(const MetalMatrix&,const MetalMatrix&,size_t batch,size_t chunk,size_t capacity) const;
#endif
    size_t LastLayoutVectorDispatches() const;
    size_t LastChunkAttentionOptimizedDispatches() const;
    size_t LastChunkAttentionMPSDispatches() const;
    size_t LastChunkFlashAttentionDispatches() const;
    size_t LastChunkAttentionMPSMatrices() const;
    size_t LastChunkAttentionMPSSkippedMatrices() const;
    size_t ChunkAttentionMPSPlanCount() const;
    size_t ChunkAttentionMPSScratchBytes() const;
    size_t ChunkAttentionMPSBucketCount() const;
    size_t LastChunkAttentionBucketDispatches() const;
    size_t LastChunkAttentionTailDispatches() const;
    size_t LastChunkAttentionBucketCapacity() const;
    size_t LastChunkAttentionUnknownControls() const;
    void PrepareAttentionChunk(const std::vector<MetalMatrix>&,size_t heads,size_t kvHeads,
        size_t dim,size_t capacity,size_t chunk,bool halfOnly=false);
    size_t HalfOnlyAttentionPlanCount() const;
#ifdef GARNET_METAL_SPLIT_DECODE_SCORES
    void SetSplitScoresEnabled(bool enabled);
    size_t LastSplitScoreDispatches() const;
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
    void SetProfile32ScoresEnabled(bool enabled);
    size_t LastProfile32ScoreDispatches() const;
#endif
#endif
    void EncodeAttentionScores(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        const MetalMatrix&,size_t heads,size_t kvHeads,size_t dim,size_t capacity);
#ifdef GARNET_METAL_VALUE_PROFILE32
    void SetValueProfile32Enabled(bool enabled);
    size_t LastValueProfile32Dispatches() const;
#endif
    void EncodeAttentionContext(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,
        size_t heads,size_t kvHeads,size_t dim,size_t capacity,bool gated=true);
    void EncodeLayout(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,size_t tokenWidth=0);
    void SubmitAndWait();
    double LastGPUMs() const;
    // Explicit diagnostic preparation. Does not split batches or disable graph
    // fusions. MPS-owned encoders remain unobserved.
#ifdef GARNET_METAL_AFFINE_WEIGHT_RESIDENCY
    void PrepareAffineResidency(const std::vector<MetalAffineInt4Weight>& weights,size_t budget);
    std::string AffineResidencyStatsJson() const;
#endif
    void PrepareGpuCounters();
    void SetGpuCounterRegion(const char* kind,size_t op,size_t rows,size_t columns);
    void ClearGpuCounterRegion();
    // Fixed-grid image operators; projections continue to use MPS matmul.
    void EncodeVision(unsigned operation,const std::vector<MetalMatrix>& inputs,
        const MetalMatrix& output,size_t tokens,size_t hidden,size_t heads,size_t mergeUnit);
    void Abort();
    std::vector<float> Download(const MetalMatrix& matrix) const;
    // Initial synchronous FP32 correctness primitive. Not the final scheduler:
    // Qwen execution must retain buffers and encode multiple ops per submission.
    std::vector<float> MatMul(const std::vector<float>& left,
        const std::vector<float>& right, size_t rows, size_t inner,
        size_t columns, bool transposeRight = false);
};
}
