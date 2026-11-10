// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "../backends/metal/compiled_graph.h"
#include "../backends/metal/token_prefix_profile.h"
#include "../backends/metal/decoder_session.h"
#ifdef GARNET_METAL_CONTINUOUS_ROWS
#include "../backends/metal/continuous_scheduler.h"
#endif
#include "model_tensor_io.h"
#include "tokenizer_api.h"
#ifdef GARNET_APPLE_COREML
#include "../backends/coreml/coreml_api.h"
#endif
#include "../backends/common/tensor_views.h"
#include "../backends/common/tensor_pixels.h"
#include "../backends/common/generation_noise.h"
#include "../media/media_api.h"
#include "../model/generation_api.h"
#include "../image/native_image.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>
#include <dlfcn.h>
namespace Garnet {
class GarnetMetalAPI {
    static void LibraryAnchor() {}
    X::Module Serving(X3PackageHost* host) {
        const std::string root = GARNET_METAL_SOURCE_ROOT;
        if (x3_runtime_add_import_root(host->runtime, root.c_str()) != X3_STATUS_OK ||
                x3_runtime_add_import_root(host->runtime, (root + "/runtime").c_str()) != X3_STATUS_OK)
            throw X::Error(x3_runtime_last_error(host->runtime));
        return X::Module(host, "metal_serving");
    }
public:
    std::string NativeLibraryPath() const {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<const void*>(&LibraryAnchor), &info) || !info.dli_fname)
            throw X::Error("Cannot identify the loaded native Metal library");
        return info.dli_fname;
    }
    std::string PrimaryStatsJson() {
        const auto p=MetalContext::PrimaryStats();
        return nlohmann::json{{"active_weights",p.activeWeights},{"original_bytes",p.originalBytes},{"payload_bytes",p.payloadBytes},{"metadata_bytes",p.metadataBytes},{"allocated_bytes",p.allocatedBytes},{"uploads",p.uploads},{"decode_encoded",p.decodeEncoded},{"prefill_encoded",p.prefillEncoded},{"head_encoded",p.headEncoded},{"vector_decode_encoded",p.vectorDecodeEncoded},{"vector_head_encoded",p.vectorHeadEncoded}}.dump();
    }
    std::string MemoryStatsJson() {
        const auto m=MetalContext::MemoryStats();
        nlohmann::json result{{"metal_allocated_bytes",m.allocated},{"metal_recommended_working_set_bytes",m.recommended},
            {"metal_registry_id",m.registry},{"task_info_status",m.taskStatus},{"task_physical_footprint_bytes",m.physical},
            {"task_resident_bytes",m.resident},{"task_compressed_bytes",m.compressed},{"task_internal_bytes",m.internal},
            {"task_virtual_bytes",m.virtualBytes},{"scope","process Metal device resources and current task; snapshots, not peak or system free RAM"}};
#ifdef GARNET_METAL_AFFINE_PROFILE
        const auto owners=MetalContext::ProfileOwnerStats();
        result["garnet_live_matrix_owners"]=owners.matrices;result["garnet_live_affine_weight_owners"]=owners.affineWeights;
        result["garnet_live_bf16_weight_owners"]=owners.bf16Weights;result["garnet_live_pipeline_sets"]=owners.pipelineSets;
#endif
        return result.dump();
    }
    std::string MemoryInventoryJson(X::Value graphs,X::Value session) {
        if(!graphs.IsList())throw X::Error("Inventory graph descriptors must be a list");
        nlohmann::json references=nlohmann::json::array(),buffers=nlohmann::json::array();std::unordered_map<uintptr_t,size_t> ids;
        auto append=[&](const std::string& tag,const std::vector<MetalResourceReference>& refs){for(const auto& r:refs){
            auto found=ids.find(r.buffer.identity);size_t id;
            if(found==ids.end()){id=buffers.size();ids.emplace(r.buffer.identity,id);buffers.push_back({{"id",id},{"allocated_bytes",r.buffer.allocatedBytes},{"length_bytes",r.buffer.lengthBytes}});}
            else {id=found->second;if(buffers[id]["allocated_bytes"]!=r.buffer.allocatedBytes||buffers[id]["length_bytes"]!=r.buffer.lengthBytes)throw X::Error("Inventory resource identity changed");}
            references.push_back({{"buffer_id",id},{"tag",tag},{"slot",r.slot==SIZE_MAX?nlohmann::json(nullptr):nlohmann::json(r.slot)},{"role",r.role},{"name",r.name},{"shape",r.shape},{"representation",r.buffer.representation},{"resident_mask_operand",r.residentOperand},{"any_op_writer",r.anyWriter},{"resident_mask_writer",r.residentWriter},{"input",r.input},{"output_part",r.outputPart},{"legacy_output",r.legacyOutput},{"weight",r.weight}});
        }};
        for(uint64_t i=0;i<graphs.Size();++i){auto descriptor=graphs.Get(i);if(!descriptor.IsDict())throw X::Error("Invalid inventory graph descriptor");auto value=descriptor.Get("object");if(value.runtime()!=graphs.runtime())throw X::Error("Inventory runtime mismatch");auto graph=value.NativeData<MetalCompiledGraph>();if(!graph)throw X::Error("Inventory requires graph objects");append(descriptor.Get("tag").ToString(),graph->MemoryReferences());}
        if(session.IsValid()&&!session.IsNone()){if(session.runtime()!=graphs.runtime())throw X::Error("Inventory session runtime mismatch");auto native=session.NativeData<MetalDecoderSession>();if(!native)throw X::Error("Inventory requires a native session");append("session",native->MemoryReferences());}
        uint64_t bytes=0;for(const auto& b:buffers)bytes+=b["allocated_bytes"].get<uint64_t>();
        return nlohmann::json{{"buffers",buffers},{"references",references},{"unique_allocated_bytes",bytes},{"device_snapshot",nlohmann::json::parse(MemoryStatsJson())},{"scope","Collection-local opaque ids, no addresses. Declared graph/held-weight/session owners only; context scratch/driver resources may be absent. Resident mask references are conservative, not a safe-free proof."}}.dump();
    }
    X::Value PrepareImage(X::Value path,X::Value options) {
        auto params=nlohmann::json::parse(options.ToString());
        Image::NativeImageConfig c;
        c.patch=params.at("patch_size").get<int>();c.merge=params.at("merge_size").get<int>();
        c.temporal=params.at("temporal_patch_size").get<int>();c.pixels=params.at("pixels").get<int>();
        auto mean=params.at("image_mean"),stddev=params.at("image_std");
        if(mean.size()!=3||stddev.size()!=3)throw X::Error("Native image normalization requires RGB");
        for(int i=0;i<3;++i){c.mean[i]=mean.at(i).get<float>();c.stddev[i]=stddev.at(i).get<float>();}
        auto image=Image::PrepareNativeImage(path.ToString(),c);
        std::vector<int64_t> shape{1,int64_t(image.gridH)*image.gridW,image.features};
        auto result=X::Value::List(path.host());result.Append(X::Tensor::Create(path.host(),X3_TENSOR_FLOAT32,shape,image.patches.data(),image.patches.size()*4));
        result.Append(X::Value(image.gridH));result.Append(X::Value(image.gridW));
        result.Append(X::Value(image.decodeMs));result.Append(X::Value(image.resizeMs));result.Append(X::Value(image.packMs));return result;
    }
    std::string BenchmarkPrefillPolicy(X::Value enabled) {
        auto module=Serving(enabled.host());
        return module["set_benchmark_prefill_policy"](enabled).ToString();
    }
    std::string ServeModel(X::Value root, X::Value xmodel, X::Value cache,
            X::Value profile, X::Value model) {
        auto module = Serving(root.host());
        return module["serve_model"](root, xmodel, cache, profile, model).ToString();
    }
    std::string ServeStatus(X::Value model) {
        auto module = Serving(model.host());
        return module["serve_status_json"](model).ToString();
    }
    X::Value StopServing(X::Value model) {
        auto module = Serving(model.host());
        return module["stop_serving"](model);
    }
    std::string InferJson(X::Value prompt, X::Value image, X::Value limit, X::Value model) {
        auto module = Serving(prompt.host());
        std::filesystem::path temporary;
        std::string imagePath;
        if (image.IsBin()) {
            uint64_t size = 0;
            const auto* data = static_cast<const char*>(image.BytesData(&size));
            if (data && size) {
                static std::atomic<uint64_t> sequence{0};
                const auto folder = std::filesystem::temp_directory_path() / "garnet-metal-images";
                std::filesystem::create_directories(folder);
                temporary = folder / (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                    + "-" + std::to_string(sequence++) + ".jpg");
                std::ofstream out(temporary, std::ios::binary);
                out.write(data, static_cast<std::streamsize>(size));
                if (!out) throw X::Error("cannot stage Garnet Metal image input");
                imagePath = temporary.string();
            }
        } else if (image.IsString()) imagePath = image.ToString();
        try {
            const auto result = module["infer_json"](prompt, imagePath, limit, model).ToString();
            if (!temporary.empty()) std::filesystem::remove(temporary);
            return result;
        } catch (...) {
            if (!temporary.empty()) std::filesystem::remove(temporary);
            throw;
        }
    }
    std::string BackendInfo() const {
        auto info="Garnet Metal/MPS "+MetalContext::AvailableDeviceName();
#ifdef GARNET_METAL_SILU_PROFILE32
        info+="; silu_profile32_optional; silu_profile32_max_rows=512; silu_profile32_threads=128";
#endif
#ifdef GARNET_METAL_PAIRED_PACKED_PREFILL
        info+="; paired_packed_prefill_optional";
#endif
#ifdef GARNET_METAL_VALUE_PROFILE32
        info+="; value_profile32_optional; value_profile32_threads=128";
#endif
#ifdef GARNET_METAL_SPLIT_DECODE_SCORES
        info+="; split_decode_scores_optional; split_decode_scores_chunk=128; split_decode_scores_min_visible=512";
#ifdef GARNET_METAL_SPLIT_SCORE_PROFILE32
        info+="; split_score_profile32_optional";
#endif
#endif
#ifdef GARNET_METAL_AFFINE_WEIGHT_RESIDENCY
        info+="; native_bounded_affine_residency";
#endif
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
        info+="; dense_key_split_k";
#endif
#ifdef GARNET_METAL_QWEN35_EXTENDED_PREFILL
        info+="; qwen35_extended_prefill";
#endif
#ifdef GARNET_METAL_QWEN35_WIDE_BATCH_PREFILL
        info+="; qwen35_wide_batch_prefill";
#endif
#ifdef GARNET_METAL_INTERLEAVED_BF16_BODY
        info+="; interleaved_bf16_body";
#endif
#ifdef GARNET_METAL_COALESCED_BF16_BODY
        info+="; coalesced_bf16_body";
#endif
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
        info+="; mlp_prefill_graph";
#endif
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
        info+="; resident_kv_inplace";
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
        info+="; joined_mps_prefill";
#endif
#ifdef GARNET_METAL_ATTENTION_PREFIX
        info+="; attention_prefix";
#endif
#ifdef GARNET_METAL_DIRECT_STATE_ZERO
        info+="; direct_state_zero";
#endif
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
        info+="; deferred_state_zero";
#endif
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        info+="; reused_zero_source";
#endif
#ifdef GARNET_METAL_QWEN35_4B_WIDE_PREFILL
        info+="; qwen35_4b_wide_prefill";
#endif
#if defined(GARNET_METAL_QWEN38_WIDE_PREFILL)||defined(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS)
#ifdef GARNET_METAL_DENSE_PREFILL_MAX_TOKENS
        info+="; prepared_dense_prefill; bounded_prefill_workspace; dense_prefill_max_tokens="+std::to_string(GARNET_METAL_DENSE_PREFILL_MAX_TOKENS);
        if(kMetalAttentionIndexBytes==2)info+="; attention_index_i16";
#else
        info+="; qwen38_wide_prefill; bounded_prefill_workspace";
#ifdef GARNET_METAL_QWEN38_LARGE_PREFILL
        info+="; qwen38_large_prefill; attention_index_i16";
#endif
#endif
#ifdef GARNET_METAL_AFFINE_TALL_TILES
        info+="; affine_q4_tall_tiles";
#endif
#ifdef GARNET_METAL_SHARED_PREFILL_SCRATCH
        info+="; shared_prefill_scratch";
#ifdef GARNET_METAL_PREFILL_LIFETIME_REUSE
        info+="; prefill_lifetime_reuse";
#ifdef GARNET_METAL_PREFILL_ALIAS_GROUPS
        info+="; prefill_alias_groups";
#ifdef GARNET_METAL_PREFILL_ALIAS_TEST_FAULTS
        info+="; prefill_alias_test_faults";
#endif
#endif
#endif
#endif
#ifdef GARNET_METAL_RETIRE_INPLACE_DEFAULTS
        info+="; retired_inplace_defaults";
#endif
#endif
#ifdef GARNET_METAL_INDEXED_AFFINE
    info+="; indexed_affine_primitive";
#endif
#ifdef GARNET_METAL_INDEXED_TOKEN_PROJECTIONS
    info+="; indexed_token_projections";
#endif
#ifdef GARNET_METAL_PREDICTION_CONFIDENCE
    info+="; prediction_confidence_top2_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_PACKED_PREFILL
    info+="; mistral_packed_prefill_optional; packed_prefill_min_rows=128; packed_prefill_scratch_bytes=14680064";
#endif
#ifdef GARNET_METAL_MISTRAL_STREAMED_PREFILL
    info+="; mistral_streamed_prefill_optional; streamed_prefill_min_rows=512; streamed_prefill_scratch_bytes=161480704";
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE64
    info+="; mpp_affine_reuse64_optional; mpp_affine_reuse64_min_outputs="+std::to_string(int(GARNET_METAL_REUSE64_MIN_OUTPUTS));
    info+="; mpp_affine_reuse64_relaxed="+std::to_string(int(GARNET_METAL_REUSE64_RELAXED_VALUE));
#endif
#ifdef GARNET_METAL_MPP_AFFINE_REUSE128
    info+="; mpp_affine_reuse128_optional; mpp_affine_reuse128_min_rows=128; mpp_affine_reuse128_threads=256; mpp_affine_reuse128_even64_tiles=1";
#ifdef GARNET_METAL_VERIFY_ATTENTION
    info+="; verification_affine_route_control";
#endif
#endif
#ifdef GARNET_METAL_INDEXED_DECODE
    info+="; indexed_fp32_decode_primitive";
#ifdef GARNET_METAL_INDEXED_DECODE_FUSED_CLEAR
    info+="; native_indexed_decode_fused_clear_optional_v1";
#endif
#endif
#ifdef GARNET_METAL_ACTIVE_DECODE_PROJECTIONS
    info+="; active_decode_projections";
#endif
#ifdef GARNET_METAL_TARGET_JACOBI
    info+="; native_target_shifted_jacobi_optional_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_DECODE_REUSE
    info+="; mistral_decode_reuse_four_rows";
#ifdef GARNET_METAL_MISTRAL_GROUPED_VALUE_CONTEXT
    info+="; mistral_grouped_value_context_two_heads";
#endif
#ifdef GARNET_METAL_MISTRAL_THREE_ROW_REUSE
    info+="; mistral_decode_reuse_three_rows";
#endif
#endif
#ifdef GARNET_METAL_AFFINE_PROFILE
    info+="; bounded_affine_kernel_profiles; affine_profile_index_bits="+std::to_string(MetalContext::AffineProfileIndexBits());
#ifdef GARNET_METAL_AFFINE_PROFILE_TGP128
    info+="; affine_profile_max_threads=128";
#else
    info+="; affine_profile_max_threads=0";
#endif
#endif
#ifdef GARNET_METAL_CHUNK_PROBABILITY_REUSE
    info+="; chunk_probability_reuse";
#endif
#ifdef GARNET_METAL_CHUNK_ATTENTION_INDEX32
    info+="; chunk_attention_index32";
#endif
#ifdef GARNET_METAL_CHUNK_HALF_ATTENTION
    info+="; fp16_prefill_all_p16_gpupack_v1";
#ifdef GARNET_METAL_HALF_ONLY_PREPARATION
    info+="; native_half_resident_only_preparation_optional_v1";
#endif
#ifdef GARNET_METAL_CHUNK_HALF_WIDE
        info+="; half_attention_maximum_chunk=512";
#endif
#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
    info+="; fused_mistral_mlp_prefill_fast_v1";
#endif
#endif
#ifdef GARNET_METAL_COMMITTED_KV_MPS
    info+="; committed_kv_mps_fp32_v1";
#endif
#ifdef GARNET_METAL_CHUNK_KV_COMMIT_HOIST
    info+="; chunk_kv_commit_hoist";
#endif
#ifdef GARNET_METAL_CHUNK_ATTENTION_MPS
    info+="; chunk_attention_mps";
#ifdef GARNET_METAL_CHUNK_ATTENTION_SCRATCH
    info+="; chunk_attention_full_write_scratch";
#endif
#ifdef GARNET_METAL_CHUNK_FLASH_ATTENTION
    info+="; chunk_flash_attention_fp32_operands";
#endif
#endif
#ifdef GARNET_METAL_CHUNK_ATTENTION_BUCKETS
    info+="; chunk_attention_buckets";
#endif
#ifdef GARNET_METAL_CHUNK_ATTENTION_ACTIVE_ROWS
    info+="; chunk_attention_active_rows";
#endif
#ifdef GARNET_METAL_RETIRE_PREFILL_INPUTS
    info+="; retired_prefill_hidden_inputs";
#endif
#ifdef GARNET_METAL_RETIRE_NONRESIDENT_PACKS
    info+="; retired_nonresident_packs";
#endif
#ifdef GARNET_METAL_CACHED_MATRIX_METADATA
    info += " cached_matrix_metadata";
#endif
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        info+="; primary_bf16_blocks; primary_vector_decode; primary_kmajor_decode; primary_u32_metadata";
#endif
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
        info+="; lazy_resident_defaults";
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
        info+="; dense_prefix_cache";
#endif
#ifdef GARNET_METAL_MLX_AFFINE_INT4
#ifdef GARNET_METAL_AFFINE_FP16_COEFFICIENTS
        info+="; mlx_mistral_affine_int4_g64_fp16; affine_coefficient_dtype=fp16";
#else
        info+="; mlx_qwen35_affine_int4_g64_bf16";
#endif
#ifdef GARNET_METAL_MPP_AFFINE
        info+="; mpp_affine_tile_k="+std::to_string(GARNET_METAL_MPP_AFFINE_TILE_K)+"; mpp_affine_relaxed="+std::to_string(GARNET_METAL_MPP_AFFINE_RELAXED_VALUE)+"; mpp_affine_n_major="+std::to_string(GARNET_METAL_MPP_AFFINE_N_MAJOR_VALUE)+"; mpp_affine_registers="+std::to_string(GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE)+"; mpp_affine_tensor_k="+std::to_string(GARNET_METAL_MPP_AFFINE_REGISTERS_VALUE?16:GARNET_METAL_MPP_AFFINE_TILE_K);
#endif
#ifdef GARNET_METAL_MPP_AFFINE_FP32
        info+="; mpp_affine_fp32_optional";
#endif
#ifdef GARNET_METAL_MPP_AFFINE_BF16
        info+="; mpp_affine_bf16_optional";
#endif
#ifdef GARNET_METAL_MPP_AFFINE_FP16
        info+="; mpp_affine_fp16_optional";
#endif
#endif
#ifdef GARNET_METAL_REQUEST_ROWS
        info+="; request_row_reset";
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        info+="; native_request_row_leases; native_continuous_scheduler; chunked_prefill_scheduler; packed_token_plan; native_prefill_completion_priority_optional";
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
        info+="; row_final_hidden";
#ifdef GARNET_METAL_TOKEN_PREFIX_CACHE
        info+="; native_hybrid_token_prefix; native_cached_prefill_scheduler";
#ifdef GARNET_METAL_PREFIX_PREFILL_ONLY
        info+="; native_prefix_prefill_only_v1";
#endif
#ifdef GARNET_METAL_DIAGNOSTIC_LAYER_BULK
        info+="; native_leased_bulk_prefill_scheduler; native_tracking_only_rows";
#endif
        info+="; native_prefix_host_limit_bytes="+std::to_string(kTokenPrefixHostLimitBytes);
#if GARNET_METAL_TOKEN_PREFIX_HOST_MIB > 512
        info+="; native_prefix_completion_recency";
#ifdef GARNET_METAL_PREFIX_SHARED_BLOCKS
        info+="; native_shared_prefix_blocks; native_prefix_block_tokens=64; native_prefix_chunk_checkpoints";
#ifdef GARNET_METAL_PREFIX_DIRECT_CAPTURE
        info+="; native_prefix_direct_capture";
#endif
#else
        info+="; native_prefix_checkpoint_min_tail=64";
#endif
#endif
#endif
#endif
#endif
#ifdef GARNET_METAL_CHUNK_KV_INPLACE
        info+="; chunk_kv_inplace";
#endif
#ifdef GARNET_METAL_CHUNK_KV_SCATTER
        info+="; chunk_kv_scatter_contiguous";
#endif
#ifdef GARNET_METAL_COMPACT_OWNER_PREFILL
        info+="; compact_owner_prefill";
#ifdef GARNET_METAL_WIDE_OWNER_PREFILL
    info+="; wide_owner_prefill; native_prefill_completion_owner_limit_v1";
#endif
#endif
#ifdef GARNET_METAL_LAYOUT_VECTOR4
        info+="; chunk_layout_vector4";
#endif
#endif
#ifdef GARNET_METAL_COMPACT_ATTENTION_INDEX
        info+="; compact_attention_index";
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_CONV
        info+="; linear_chunk_conv";
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_DELTA
        info+="; linear_chunk_delta";
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_ACTIVE_ONLY
        info+="; linear_chunk_active_only";
#endif
#ifdef GARNET_METAL_MTP
        info+="; raw_bf16_mtp_graph";
        info+="; raw_bf16_mtp_prefill_profiles";
#endif
#ifdef GARNET_METAL_MTP_BRIDGE
        info+="; native_mtp_target_bridge";
        info+="; native_mtp_generation";
        info+="; native_mtp_profile_family";
        info+="; native_mtp_continuous_scheduler";
#endif
#ifdef GARNET_METAL_MTP_KV_INPLACE
        info+="; native_mtp_kv_inplace";
#endif
#ifdef GARNET_METAL_MTP_DEFER_PACKS
        info+="; native_mtp_deferred_packs";
#endif
#ifdef GARNET_METAL_PREFILL_HIDDEN_POOL
        info+="; native_prefill_hidden_pool";
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_STATE_INPLACE
        info+="; linear_chunk_state_inplace";
#endif
#ifdef GARNET_METAL_VERIFY_ATTENTION
        info+="; verification_attention_fp32";
#ifdef GARNET_METAL_MISTRAL_VERIFY_VECTOR4
        info+="; mistral_verification_value_vector4; verification_value_route_control";
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
        info+="; mistral_verification_layout_b1t4_c4096_h32_kv8_d128; verification_layout_route_control";
#endif
        info+="; verification_recurrent_checkpoint";
        info+="; verification_kv_span_checkpoint";
#if defined(GARNET_METAL_MLX_AFFINE_INT4)&&defined(GARNET_METAL_INDEXED_TOKEN_PROJECTIONS)&&defined(GARNET_METAL_ACTIVE_DECODE_PROJECTIONS)&&defined(GARNET_METAL_INDEXED_DECODE)&&defined(GARNET_METAL_AFFINE_PROFILE)
        info+="; compiled_verification_fp32";
#ifdef GARNET_METAL_ROW_FINAL_HIDDEN
        info+="; native_greedy_verification";
        info+="; native_greedy_dense_verification";
#if defined(GARNET_METAL_LINEAR_CHUNK_CONV)&&defined(GARNET_METAL_LINEAR_CHUNK_DELTA)
        info+="; native_greedy_recurrent_verification";
#endif
#endif
#endif
#endif
        info+="; medusa_residual_head_bf16";
#ifdef GARNET_METAL_BODY_SHAPE
        info+="; body_shape";
#endif
#ifdef GARNET_METAL_PREFILL_STREAM
        info+="; prefill_stream";
#endif
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        info+="; dirty_prefix_reset";
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        info+="; constant_bf16_head";
#endif
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        info+="; single_coder_head";
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
        info+="; fused_head_argmax";
#endif
#ifdef GARNET_METAL_FUSED_HEAD_DECODE_ONLY
        info+="; fused_head_decode_only";
#endif
#ifdef GARNET_METAL_RESIDUAL_NORM
        info+="; residual_norm";
#endif
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
        info+="; packed_prefill_norm";
#endif
#ifdef GARNET_METAL_GLOBAL_HALF_ALPHA
        info+="; global_half_alpha";
#endif
#ifdef GARNET_METAL_EXACT_HALF_PREFILL
        info+="; exact_half_prefill_max_mib="+std::to_string(GARNET_METAL_EXACT_HALF_MAX_MIB);
#endif
#ifdef GARNET_METAL_MLX_STEEL_ATTENTION
        info+="; mlx_steel_attention";
#endif
#ifdef GARNET_METAL_MLX_AFFINE_QMM
        info+="; mlx_affine_qmm";
#endif
#ifdef GARNET_METAL_SCREENED_HEAD
        info+="; screened_head";
#endif
#ifdef GARNET_METAL_SMALL_SCREEN_SEED
        info+="; screen_seed_1024";
#endif
#ifdef GARNET_METAL_INTERLEAVED_SCREEN_INDEX
        info+="; interleaved_screen_index";
#endif
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
        info+="; serial_vision_prefetch; vector_vision_context4";
#endif
#ifdef GARNET_METAL_VISION_GLOBAL_ALPHA
        info+="; vision_global_alpha";
#endif
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
        info+="; parallel_vision_context";
#endif
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
        info+="; joined_global_mlp";
#endif
#ifdef GARNET_APPLE_COREML
        info+="; coreml_stateful_optional";
#endif
#ifdef GARNET_METAL_WIDE_DENSE_VERIFICATION
    info+="; native_dense_verify_b4t4_optional_v1; native_wide_verify_preparation_lifetime_v1";
#endif
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
    info+="; native_compact_verify_b4t1_optional_v1";
#endif
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    info+="; native_scheduled_dense_verify_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_SPLIT_SCORES
    info+="; native_mistral_verification_split_scores128_default_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES
    info+="; native_mistral_verification_gqa16_values64_32_optional_v1";
#endif
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    info+="; native_mistral_verification_gqa16_scores32_optional_v1";
#endif
    #ifdef GARNET_METAL_FUSED_Q4_DECODE
    info+="; native_fused_q4_decode_rows4_default_v1";
#ifdef GARNET_METAL_FUSED_Q4_WIDE_SELECTION
    info+="; native_fused_q4_wide_selection_optional_v1";
#endif
#endif
#ifdef GARNET_METAL_LOOKAHEAD_RUNTIME
    info+="; native_AR_DAG_lookahead_session_optional_v1";
#endif
    return info;
    }
    // Graphs own synchronous queues and release them through close(). There
    // is no package-global executor to drain, unlike asynchronous backends.
    bool Shutdown() const {return true;}
    X::Value Generate(const X::ARGS& args,const X::KWARGS& options){return GenerateMedia(Host(),args,options);}
    X::Value GaussianNoise(const X::ARGS& args,const X::KWARGS& options){return GenerationGaussianNoise(Host(),args,options);}
    BEGIN_PACKAGE(GarnetMetalAPI)
#ifdef GARNET_APPLE_COREML
        APISET().AddClass<0,CoreMLAPI>("coreml_session");
#endif
        APISET().AddVarFunc("generate",&GarnetMetalAPI::Generate);
        APISET().AddVarFunc("gaussian_noise",&GarnetMetalAPI::GaussianNoise);
        APISET().AddClass<0,GarnetTensor>("tensor");
        APISET().AddClass<0,MetalCompiledGraph>("compiled_graph");
        APISET().AddClass<0,MetalDecoderSession>("decoder_session");
#ifdef GARNET_METAL_CONTINUOUS_ROWS
        APISET().AddClass<0,MetalContinuousScheduler>("continuous_scheduler");
#endif
        APISET().AddClass<0,ModelTensorIO>("tensor_io");
        APISET().AddClass<0,CPUTokenizer>("tokenizer");
        APISET().AddClass<0,TensorViews>("tensor_views");
        APISET().AddClass<0,TensorPixels>("tensor_pixels");
        APISET().AddClass<0,NativeMediaAPI>("media");
        APISET().AddClass<1,NativeVideoWriter>("video_writer");
        APISET().AddClass<1,NativeVideoAssembler>("video_assembler");
        APISET().AddFunc<2>("prepare_image",&GarnetMetalAPI::PrepareImage);
        APISET().AddFunc<2>("memory_inventory_json",&GarnetMetalAPI::MemoryInventoryJson);
        APISET().AddFunc<0>("primary_storage_stats_json",&GarnetMetalAPI::PrimaryStatsJson);
        APISET().AddFunc<0>("memory_stats_json",&GarnetMetalAPI::MemoryStatsJson);
        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);
        APISET().AddFunc<0>("native_library_path",&GarnetMetalAPI::NativeLibraryPath);
        APISET().AddFunc<5>("serve_model",&GarnetMetalAPI::ServeModel);
        APISET().AddFunc<1>("serve_status_json",&GarnetMetalAPI::ServeStatus);
        APISET().AddFunc<1>("stop_serving",&GarnetMetalAPI::StopServing);
        APISET().AddFunc<4>("infer_json",&GarnetMetalAPI::InferJson);
        APISET().AddFunc<1>("set_benchmark_prefill_policy",&GarnetMetalAPI::BenchmarkPrefillPolicy);
        APISET().AddFunc<0>("shutdown",&GarnetMetalAPI::Shutdown);
    END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current) {
    auto* host=static_cast<X3PackageHost*>(rawHost);
    if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;
    try {Garnet::GarnetMetalAPI::BuildAPI();return Garnet::GarnetMetalAPI::APISET().Create(host,"garnet",current);}
    catch(...){return X3_STATUS_ERROR;}
}
