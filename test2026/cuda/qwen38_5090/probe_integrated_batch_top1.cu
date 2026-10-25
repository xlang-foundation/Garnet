// SPDX-License-Identifier: Apache-2.0
// Production batch entry-point controls; both opt-in and fallback processes required.
#include "../../../src/cuda/cuda_lib.h"
#include "../../../src/cuda/logits_top1_batch_dispatch.h"
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" cudaError_t runReferenceLogitsTop1BatchFP32(const float*,long long*,float*,int,int,cudaStream_t);
extern "C" cudaError_t runReferenceLogitsTop1BatchBF16(const bfloat16*,long long*,float*,int,int,cudaStream_t);
template<class T> cudaError_t integrated(const T*,long long*,float*,int,int,cudaStream_t);
template<> cudaError_t integrated(const float* x,long long* ids,float* scores,int rows,int vocab,cudaStream_t stream) {
    return runLogitsTop1BatchFP32(x,ids,scores,rows,vocab,stream);
}
template<> cudaError_t integrated(const __nv_bfloat16* x,long long* ids,float* scores,int rows,int vocab,cudaStream_t stream) {
    return runLogitsTop1BatchBF16(x,ids,scores,rows,vocab,stream);
}
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
uint32_t bits(float v) { uint32_t result; std::memcpy(&result, &v, 4); return result; }
float fromBits(uint32_t v) { float result; std::memcpy(&result, &v, 4); return result; }
template<class T> float value(T v) { return v; }
template<> float value(__nv_bfloat16 v) { return __bfloat162float(v); }
template<class T> T scalar(float v) { return v; }
template<> __nv_bfloat16 scalar(float v) { return __float2bfloat16(v); }

float fixture(int row, int col, int vocab, int pattern) {
    const auto seed = (uint32_t(col) * 1664525u + 1013904223u) ^ (uint32_t(row) * 2246822519u);
    switch (pattern) {
    case 0: return float(int(seed % 2047u) - 1023) / 64.0f;
    case 1: return std::numeric_limits<float>::quiet_NaN();
    case 2: return -std::numeric_limits<float>::infinity();
    case 3: return -std::numeric_limits<float>::max();
    case 4: return fromBits((row + col) % 2 ? 0x80000000u : 0u);
    case 5: return col == std::min(vocab - 1, 1024) || col == vocab - 1 ? std::numeric_limits<float>::infinity() : -1.0f;
    case 6: return col == vocab - 1 ? -1.0f : std::numeric_limits<float>::quiet_NaN();
    case 7: return col == std::min(vocab - 1, 1023) || col == std::min(vocab - 1, 1024) || col == vocab - 1 ? 4096.0f : -3.0f;
    case 8: return fromBits(col == vocab - 1 ? 1u : 0x80000001u);
    case 9: return col == vocab - 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::quiet_NaN();
    case 10: return col == (row * 257) % vocab || col == vocab - 1 ? -1000.0f : -2000.0f;
    }
    throw std::runtime_error("unknown fixture pattern");
}

template<class T> void legacy(const T* x, long long* ids, float* scores, int rows, int vocab, cudaStream_t s);
template<> void legacy(const float* x, long long* ids, float* scores, int rows, int vocab, cudaStream_t s) {
    check(runReferenceLogitsTop1BatchFP32(x, ids, scores, rows, vocab, s));
}
template<> void legacy(const __nv_bfloat16* x, long long* ids, float* scores, int rows, int vocab, cudaStream_t s) {
    check(runReferenceLogitsTop1BatchBF16(x, ids, scores, rows, vocab, s));
}

template<class T> void runType(std::ofstream& log, const char* type, int& cases, int& graphs, int& rejected) {
    const std::vector<std::pair<int,int>> shapes{{1,1},{1,7},{4,255},{4,256},{4,1023},{4,1024},{4,1025},
        {4,8191},{4,8192},{1,248320},{4,248320},{8,248320},{16,248320},{128,248320},{4,248321},{129,7}};
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    for (const auto shape : shapes) {
        const int rows = shape.first, vocab = shape.second;
        std::vector<T> host(size_t(rows) * vocab), after(host.size());
        std::vector<long long> expectedIds(rows), ids(rows + 4), baselineIds(rows + 4);
        std::vector<float> expectedScores(rows), scores(rows + 4), baselineScores(rows + 4);
        T* x; long long* gpuIds; long long* gpuBaselineIds; float* gpuScores; float* gpuBaselineScores;
        check(cudaMalloc(&x, host.size() * sizeof(T)));
        check(cudaMalloc(&gpuIds, ids.size() * sizeof(long long)));
        check(cudaMalloc(&gpuBaselineIds, ids.size() * sizeof(long long)));
        check(cudaMalloc(&gpuScores, scores.size() * sizeof(float)));
        check(cudaMalloc(&gpuBaselineScores, scores.size() * sizeof(float)));
        auto clear = [&] {
            check(cudaMemsetAsync(gpuIds, 0x5a, ids.size()*8, stream));
            check(cudaMemsetAsync(gpuBaselineIds, 0x5a, ids.size()*8, stream));
            check(cudaMemsetAsync(gpuScores, 0x5a, scores.size()*4, stream));
            check(cudaMemsetAsync(gpuBaselineScores, 0x5a, scores.size()*4, stream));
        };
        auto readOutputs = [&] {
            check(cudaMemcpyAsync(ids.data(), gpuIds, ids.size()*8, cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(scores.data(), gpuScores, scores.size()*4, cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(baselineIds.data(), gpuBaselineIds, ids.size()*8, cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(baselineScores.data(), gpuBaselineScores, scores.size()*4, cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
        };
        auto observe = [&](const char* phase, int pattern, int repeat) {
            log << "{\"kind\":\"observation\",\"dtype\":\"" << type << "\",\"rows\":" << rows << ",\"vocab\":" << vocab
                << ",\"pattern\":" << pattern << ",\"phase\":\"" << phase << "\",\"repeat\":" << repeat;
            auto integers = [&](const char* name, const auto& data) {
                log << ",\"" << name << "\":[";
                for (size_t i=0;i<data.size();++i) log << (i ? "," : "") << data[i];
                log << "]";
            };
            auto floats = [&](const char* name, const auto& data) {
                log << ",\"" << name << "\":[";
                for (size_t i=0;i<data.size();++i) log << (i ? "," : "") << bits(data[i]);
                log << "]";
            };
            integers("candidate_ids",ids);floats("candidate_score_bits",scores);
            integers("legacy_ids",baselineIds);floats("legacy_score_bits",baselineScores);
            integers("cpu_ids",expectedIds);floats("cpu_score_bits",expectedScores);
            log << "}\n";log.flush();
        };
        auto verify = [&] {
            require(ids == baselineIds, "candidate/legacy ID or canary difference");
            require(std::memcmp(scores.data(), baselineScores.data(), scores.size()*4) == 0, "candidate/legacy value or canary difference");
            for (int row = 0; row < rows; ++row) {
                require(ids[row+2] == expectedIds[row] && bits(scores[row+2]) == bits(expectedScores[row]), "independent serial CPU oracle mismatch");
            }
            for (int row : {0, 1, rows+2, rows+3}) {
                require(ids[row] == 0x5a5a5a5a5a5a5a5aLL && bits(scores[row]) == 0x5a5a5a5au, "output boundary canary changed");
            }
        };
        for (int pattern = 0; pattern < 11; ++pattern) {
            log << "{\"kind\":\"case_started\",\"dtype\":\"" << type << "\",\"rows\":" << rows
                << ",\"vocab\":" << vocab << ",\"pattern\":" << pattern << "}\n";log.flush();
            for (int row = 0; row < rows; ++row) {
                float best = -std::numeric_limits<float>::max(); long long index = 0;
                for (int col = 0; col < vocab; ++col) {
                    const T element = scalar<T>(fixture(row,col,vocab,pattern));
                    host[size_t(row)*vocab+col] = element;
                    // Independent serial scan, with no packed keys, chunks,
                    // parallel reduction, GPU helper or candidate implementation.
                    const float v = value(element);
                    if (v > best || (v == best && col < index)) { best=v; index=col; }
                }
                expectedIds[row]=index; expectedScores[row]=best;
            }
            check(cudaMemcpyAsync(x, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice, stream));
            clear(); legacy(x,gpuBaselineIds+2,gpuBaselineScores+2,rows,vocab,stream);
            for (int repeat = 0; repeat < 3; ++repeat) {
                check(integrated(x,gpuIds+2,gpuScores+2,rows,vocab,stream));
                readOutputs();observe("normal",pattern,repeat);verify();
            }
            check(cudaMemcpyAsync(after.data(),x,host.size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            require(std::memcmp(host.data(),after.data(),host.size()*sizeof(T))==0,"readonly input changed");
            log << "{\"kind\":\"case\",\"dtype\":\"" << type << "\",\"rows\":" << rows << ",\"vocab\":" << vocab
                << ",\"pattern\":" << pattern << ",\"repeats\":3,\"cpu_legacy_candidate_exact\":true,\"readonly\":true,\"canaries\":true,\"ids\":[";
            for (int row=0; row<rows; ++row) log << (row ? "," : "") << ids[row+2];
            log << "],\"score_bits\":[";
            for (int row=0; row<rows; ++row) log << (row ? "," : "") << bits(scores[row+2]);
            log << "]}\n";log.flush();++cases;
            // Also exercise nullable score output and captured replay on the
            // actual current B4 vocabulary. No performance numbers are taken.
            if (rows==4 && vocab==248320 && pattern==0) {
                check(integrated(x,gpuIds+2,static_cast<float*>(nullptr),rows,vocab,stream));
                readOutputs();observe("nullable_score",pattern,0);verify();
                cudaGraph_t graph;cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                check(integrated(x,gpuIds+2,gpuScores+2,rows,vocab,stream));
                check(cudaStreamEndCapture(stream,&graph));
                check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                for (int repeat=0;repeat<3;++repeat) {check(cudaGraphLaunch(executable,stream));readOutputs();observe("graph",pattern,repeat);verify();}
                check(cudaMemcpyAsync(after.data(),x,host.size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                require(std::memcmp(host.data(),after.data(),host.size()*sizeof(T))==0,"captured readonly input changed");
                check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));++graphs;
                log << "{\"kind\":\"graph\",\"dtype\":\"" << type << "\",\"rows\":4,\"vocab\":248320,\"replays\":3,\"exact\":true,\"nullable_score_exact\":true}\n";log.flush();
            }
        }
        if (rows==4 && vocab==248320) {
            auto reject = [&](const char* label, cudaError_t status) {
                log << "{\"kind\":\"rejection\",\"dtype\":\"" << type << "\",\"label\":\"" << label
                    << "\",\"status\":" << int(status) << ",\"expected\":" << int(cudaErrorInvalidValue) << "}\n";log.flush();
                require(status==cudaErrorInvalidValue,"invalid candidate call accepted");++rejected;
            };
            for (auto args : std::vector<std::pair<int,int>>{{0,vocab},{-1,vocab},{rows,0},{rows,-1}}) {
                const std::string label="dimensions-"+std::to_string(args.first)+"-"+std::to_string(args.second);
                reject(label.c_str(),integrated(x,gpuIds+2,gpuScores+2,args.first,args.second,stream));
            }
            reject("null-input",integrated(static_cast<const T*>(nullptr),gpuIds+2,gpuScores+2,rows,vocab,stream));
            reject("null-ids",integrated(x,static_cast<long long*>(nullptr),gpuScores+2,rows,vocab,stream));
            readOutputs();observe("after_rejections",10,0);verify();
        }
        check(cudaFree(x));check(cudaFree(gpuIds));check(cudaFree(gpuBaselineIds));check(cudaFree(gpuScores));check(cudaFree(gpuBaselineScores));
    }
    check(cudaStreamDestroy(stream));
}

int main(int argc,char** argv) {
    if(argc!=3 || (std::string(argv[2])!="0" && std::string(argv[2])!="1")) return 2;
    std::ofstream log(argv[1]);if(!log) return 2;
    try {
        const bool enabled=std::string(argv[2])=="1";
        auto before=Garnet::GetBatchTop1DispatchSnapshot();
        require(before.enabled==enabled && !before.legacyHostCalls && !before.parallelHostCalls,"unexpected initial dispatch");
        // Freeze the process choice even if the environment changes later.
#ifdef _WIN32
        require(_putenv_s("GARNET_PARALLEL_BATCH_TOP1",enabled?"0":"1")==0,"environment control failed");
#else
        require(setenv("GARNET_PARALLEL_BATCH_TOP1",enabled?"0":"1",1)==0,"environment control failed");
#endif
        require(Garnet::GetBatchTop1DispatchSnapshot().enabled==enabled,"dispatch mode changed after initialization");
        log << "{\"kind\":\"dispatch_start\",\"parallel_enabled\":" << (enabled?"true":"false")
            << ",\"legacy_host_calls\":0,\"parallel_host_calls\":0,\"environment_changed_after_snapshot\":true}\n";log.flush();
        int cases=0,graphs=0,rejected=0;
        runType<float>(log,"fp32",cases,graphs,rejected);
        runType<__nv_bfloat16>(log,"bf16",cases,graphs,rejected);
        require(cases==352 && graphs==2 && rejected==12,"incomplete control inventory");
        auto after=Garnet::GetBatchTop1DispatchSnapshot();
        // Seven large-vocabulary shapes x eleven patterns x three calls x two dtypes,
        // plus nullable-score and capture construction for each dtype. Graph replays
        // and rejected calls do not invoke a successful host dispatch.
        const uint64_t expectedParallel=enabled?466:0;
        require(after.enabled==enabled && after.parallelHostCalls==expectedParallel
            && after.legacyHostCalls==1060-expectedParallel,"production route counts differ");
        log << "{\"kind\":\"dispatch_end\",\"parallel_enabled\":" << (enabled?"true":"false")
            << ",\"legacy_host_calls\":" << after.legacyHostCalls << ",\"parallel_host_calls\":" << after.parallelHostCalls
            << ",\"counter_scope\":\"host dispatch attempts including capture construction; excludes graph replays\"}\n";log.flush();
        log << "{\"kind\":\"summary\",\"status\":\"integrated_batch_top1_correctness_passed\",\"cases\":" << cases
            << ",\"graphs\":" << graphs << ",\"rejected_calls\":" << rejected << ",\"performance_admitted\":false}\n";
        std::cout << "INTEGRATED_BATCH_TOP1_CORRECTNESS_PASS cases=" << cases << " graphs=" << graphs << " rejected=" << rejected << '\n';
        return 0;
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
