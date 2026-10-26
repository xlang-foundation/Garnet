// CPU-only stubs for wire/descriptor checks; never link these into Garnet.
#include "qwen38_bf16_prefill128_attention.cuh"
#include "qwen38_bf16_kv_low_batch_overlay.cuh"
#include "qwen38_bf16_kv_overlay.cuh"
namespace Garnet {
namespace { int unexpected=0;
bool legacy(Qwen35AttentionCoreConfig c){return (c.batch==4||c.batch==8)&&(c.tokens==4||c.tokens==32)&&c.heads==24&&c.kv==4&&c.capacity==4096&&c.dim==256;}
}
size_t Qwen38BF16KVOverlayWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    return legacy(c)?size_t(c.batch)*c.tokens*c.heads*c.capacity*sizeof(float):0;
}
size_t Qwen38BF16KVLowBatchCandidateWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    return BF16KVLowBatchCandidateProfile(c)?size_t(c.batch)*c.tokens*c.heads*c.capacity*sizeof(float):0;
}
size_t Qwen38BF16KVPrefill128AttentionWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    return BF16KVPrefill128AttentionProfile(c)?size_t(c.batch)*c.tokens*c.heads*c.capacity*sizeof(float):0;
}
int Prefill128PluginCPUUnexpectedEnqueues() noexcept{return unexpected;}
int RunQwen38BF16KVOverlay(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int64_t,int64_t,cudaStream_t) noexcept{++unexpected;return 1;}
int RunQwen38BF16KVLowBatchCandidate(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int64_t,int64_t,cudaStream_t) noexcept{++unexpected;return 1;}
int RunQwen38BF16KVPrefill128Attention(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int64_t,int64_t,int,cudaStream_t) noexcept{++unexpected;return 1;}
}
