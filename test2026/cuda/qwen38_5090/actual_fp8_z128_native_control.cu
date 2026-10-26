// SPDX-License-Identifier: Apache-2.0
// Untimed actual non-repeated128 Z projection control, no application DLL.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"
#include "modelopt_fp8_z128_canonical.h"

static bool shared = false, savedValid = false, nativeZ = false;
static int observedCalls = 0;
static cublasLtMatmulAlgo_t saved{};
static void* interventionWorkspace = nullptr;
static constexpr size_t interventionBytes = 8ULL << 20;

static void Algorithm(const char* label, const cublasLtMatmulAlgo_t* algo) {
    int id = -1; size_t written = 0;
    const auto status = cublasLtMatmulAlgoConfigGetAttribute(algo,
        CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written);
    printf("%s id_query=%d written=%zu id=%d opaque=", label, int(status), written, id);
    const auto* bytes = reinterpret_cast<const unsigned char*>(algo);
    for (size_t i = 0; i < sizeof(*algo); ++i) printf("%02x", bytes[i]);
    puts("");
}

// Default mode observes unchanged actual helper calls. Shared mode replaces
// only actualM128 algorithm/workspace after AlgoCheck of those descriptors.
cublasStatus_t ObserveMatmul(cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t ad, const void* b, cublasLtMatrixLayout_t bd,
    const void* beta, const void* c, cublasLtMatrixLayout_t cd,
    void* d, cublasLtMatrixLayout_t dd, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceBytes, cudaStream_t stream) {
    uint64_t m = 0; size_t written = 0;
    if (cublasLtMatrixLayoutGetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_COLS,
        &m, sizeof(m), &written) != CUBLAS_STATUS_SUCCESS ||
        (m != 32 && m != 128) || workspaceBytes != (4ULL << 20))
        return CUBLAS_STATUS_INVALID_VALUE;
    ++observedCalls;
    if(nativeZ && m==128 && (!savedValid || !algo || std::memcmp(algo,&saved,sizeof(saved)) || workspaceBytes!=(4ULL<<20)))return CUBLAS_STATUS_INVALID_VALUE;
    printf("OBSERVE m=%llu original_workspace=%zu\n", (unsigned long long)m, workspaceBytes);
    Algorithm("DEFAULT_SELECTION", algo);
    const cublasLtMatmulAlgo_t* selected = algo;
    if (shared && m == 128) {
        if (!savedValid || !interventionWorkspace) return CUBLAS_STATUS_INVALID_VALUE;
        cublasLtMatmulHeuristicResult_t check{};
        const auto status = cublasLtMatmulAlgoCheck(handle, operation, ad, bd, cd, dd, &saved, &check);
        printf("M32_ALGORITHM_CHECK_M128 status=%d state=%d required_workspace=%zu supplied_workspace=%zu\n",
            int(status), int(check.state), check.workspaceSize, interventionBytes);
        if (status != CUBLAS_STATUS_SUCCESS || check.state != CUBLAS_STATUS_SUCCESS ||
            check.workspaceSize > interventionBytes) return CUBLAS_STATUS_INVALID_VALUE;
        selected = &saved; workspace = interventionWorkspace; workspaceBytes = interventionBytes;
    }
    Algorithm("ACTUAL_SELECTION", selected);
    const auto status = cublasLtMatmul(handle, operation, alpha, a, ad, b, bd,
        beta, c, cd, d, dd, selected, workspace, workspaceBytes, stream);
    printf("ACTUAL_MATMUL m=%llu status=%d workspace=%zu\n",
        (unsigned long long)m, int(status), workspaceBytes);
    if (m == 32 && status == CUBLAS_STATUS_SUCCESS) {
        if (savedValid && std::memcmp(&saved, selected, sizeof(saved)))
            return CUBLAS_STATUS_INVALID_VALUE;
        saved = *selected; savedValid = true;
    }
    // Invalid, rather than NOT_SUPPORTED, forbids a fallback on failed policy.
    if (shared && m == 128 && status != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_INVALID_VALUE;
    return status;
}
#define cublasLtMatmul ObserveMatmul
#include "modelopt_cublaslt.cpp"
#undef cublasLtMatmul

static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static std::vector<unsigned char> Read(const std::string& name, size_t size) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(size))
        throw std::runtime_error("Input missing or wrong size: " + name);
    f.seekg(0); std::vector<unsigned char> data(size);
    if (!f.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("Input read failed: " + name);
    return data;
}
static void Write(const std::string& name, const std::vector<unsigned char>& data) {
    std::ofstream f(name, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(data.data()), data.size()))
        throw std::runtime_error("Output write failed: " + name);
}
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { Check(cudaMalloc(&p, bytes)); }
    ~Device() { if (p) cudaFree(p); }
};
__global__ void Cast(const float* input, __nv_bfloat16* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
}
int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4)
            throw std::runtime_error("Usage: control FIXTURE_LAYER OUTPUT_EXISTING_DIR [--native-z128]");
        if (argc == 4) {
            if (std::strcmp(argv[3], "--native-z128")) throw std::runtime_error("Unknown mode");
            nativeZ = true;
        }
        const std::string fixture = std::string(argv[1]) + "/", output = std::string(argv[2]) + "/";
        constexpr int m = 128, n = 6144, k = 5120;
        const auto weights = Read(fixture + "weight-n6144-k5120.e4m3", n * k);
        const auto input = Read(fixture + "input-m128-k5120.f32", m * k * 4);
        const auto expected32 = Read(fixture + "expected-four-m32-n6144.bf16", m * n * 2);
        const auto expected128 = Read(fixture + "expected-m128-n6144.bf16", m * n * 2);
        const auto ws = Read(fixture + "weight-scale.f32", 4), xs = Read(fixture + "input-scale.f32", 4);
        float weightScale, inputScale;
        std::memcpy(&weightScale, ws.data(), 4); std::memcpy(&inputScale, xs.data(), 4);
        if (!std::isfinite(weightScale) || weightScale <= 0 || !std::isfinite(inputScale) || inputScale <= 0)
            throw std::runtime_error("Invalid calibration");
        int device = -1; cudaDeviceProp properties{};
        Check(cudaGetDevice(&device)); Check(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 12 || properties.minor != 0 || cublasLtGetVersion() != 130400)
            throw std::runtime_error("Control requires pinnedSM120/cuBLASLt13.4");
        printf("DEVICE %s SM%d%d CUBLASLT_VERSION %zu\n", properties.name, properties.major, properties.minor, cublasLtGetVersion());
        char loaded[MAX_PATH]{};
        const auto length = GetModuleFileNameA(GetModuleHandleA("cublasLt64_13.dll"), loaded, MAX_PATH);
        if (!length || length >= MAX_PATH) throw std::runtime_error("Cannot identify actual cuBLASLt DLL");
        for (DWORD i = 0; i < length; ++i) if (loaded[i] == '\\') loaded[i] = '/';
        printf("CUBLASLT_DLL %s\n", loaded);
        cublasLtHandle_t handle = nullptr;
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt creation failed");
        Device w(weights.size()), x(input.size()), bf16(m * k * 2), packed(m * k), result(m * n * 2), workspace(4ULL << 20), wide(interventionBytes);
        interventionWorkspace = wide.p;
        Check(cudaMemcpy(w.p, weights.data(), weights.size(), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(x.p, input.data(), input.size(), cudaMemcpyHostToDevice));
        Cast<<<(m * k + 255) / 256, 256>>>(static_cast<const float*>(x.p), static_cast<__nv_bfloat16*>(bf16.p), m * k);
        Check(cudaGetLastError());
        Check(Garnet::QuantizeModelOptFP8Activation(bf16.p, static_cast<unsigned char*>(packed.p), inputScale, m, k, nullptr));
        std::vector<unsigned char> cast(m * k * 2), quantized(m * k);
        Check(cudaMemcpy(cast.data(), bf16.p, cast.size(), cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(quantized.data(), packed.p, quantized.size(), cudaMemcpyDeviceToHost));
        Write(output + "cast-m128.bf16", cast); Write(output + "quantized-m128.e4m3", quantized);
        size_t totalChanged = 0;
        for (int tokens : {32, 128}) {
            const std::string profile = tokens == 32 ? "four-m32" : "m128";
            std::shared_ptr<Garnet::ModelOptFP8Z128Context> context;
            if(nativeZ && tokens==128){
                context=Garnet::CreateModelOptFP8Z128Context();Garnet::ModelOptFP8Z128Info info;
                if(!savedValid || !Garnet::GetModelOptFP8Z128Info(context,info) || info.algorithm!=67 || info.device!=0 || info.requiredWorkspace!=0 || std::memcmp(&info.opaque,&saved,sizeof(saved)))
                    throw std::runtime_error("Typed Z128 context does not match qualified actualM32 configuration");
                printf("NATIVE_Z128_CONTEXT algorithm=%d device=%d required_workspace=%zu blas_workspace=%zu opaque=",info.algorithm,info.device,info.requiredWorkspace,Garnet::ModelOptFP8Z128BlasWorkspace);
                const auto* bytes=reinterpret_cast<const unsigned char*>(&info.opaque);for(size_t i=0;i<sizeof(info.opaque);++i)printf("%02x",bytes[i]);puts("");
                const int callsBefore=observedCalls;int rejected=0;
                auto reject=[&](const std::shared_ptr<Garnet::ModelOptFP8Z128Context>& owner,const void* weight,const void* input,void* output,int batch,int rows,int columns,float scale,void* scratch,size_t size,cublasStatus_t expected){
                    if(Garnet::RunModelOptFP8Z128Matmul(owner,weight,input,output,batch,rows,columns,scale,scratch,size,nullptr)!=expected)throw std::runtime_error("Native Z invalid argument accepted");++rejected;
                };
                const float scale=weightScale*inputScale;const size_t size=Garnet::ModelOptFP8Z128BlasWorkspace;
                reject({},w.p,packed.p,result.p,128,n,k,scale,workspace.p,size,CUBLAS_STATUS_NOT_INITIALIZED);
                reject(context,nullptr,packed.p,result.p,128,n,k,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,nullptr,result.p,128,n,k,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,packed.p,nullptr,128,n,k,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,packed.p,result.p,128,n,k,scale,nullptr,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,packed.p,result.p,32,n,k,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,packed.p,result.p,128,5120,k,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                reject(context,w.p,packed.p,result.p,128,n,6144,scale,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                for(size_t wrong:{size_t(0),size-1,size_t(8ULL<<20)})reject(context,w.p,packed.p,result.p,128,n,k,scale,workspace.p,wrong,CUBLAS_STATUS_INVALID_VALUE);
                for(float wrong:{0.f,-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})reject(context,w.p,packed.p,result.p,128,n,k,wrong,workspace.p,size,CUBLAS_STATUS_INVALID_VALUE);
                if(rejected!=15 || observedCalls!=callsBefore)throw std::runtime_error("Invalid native Z control accounting/GPU call");
                printf("NATIVE_Z128_REJECTIONS count=%d unexpected_matmul_calls=%d\n",rejected,observedCalls-callsBefore);
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::vector<unsigned char> actual(m * n * 2);
                for (int start = 0; start < m; start += tokens) {
                    printf("CONTROL profile=%s repeat=%d start=%d\n", profile.c_str(), repeat, start);
                    const auto status = nativeZ && tokens==128
                        ? Garnet::RunModelOptFP8Z128Matmul(context,w.p,packed.p,result.p,128,n,k,weightScale*inputScale,workspace.p,Garnet::ModelOptFP8Z128BlasWorkspace,nullptr)
                        : Garnet::RunModelOptFP8Matmul(handle, w.p,
                        static_cast<unsigned char*>(packed.p) + start * k, result.p, tokens, n, k,
                        weightScale * inputScale, workspace.p, 4ULL << 20, nullptr, false);
                    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Actual native FP8 matmul failed");
                    Check(cudaDeviceSynchronize());
                    Check(cudaMemcpy(actual.data() + start * n * 2, result.p, tokens * n * 2, cudaMemcpyDeviceToHost));
                }
                Write(output + profile + "-r" + std::to_string(repeat) + ".bf16", actual);
                const auto& expected = tokens == 32 || shared || nativeZ ? expected32 : expected128;
                size_t changed = 0;
                for (size_t i = 0; i < actual.size(); i += 2)
                    if (actual[i] != expected[i] || actual[i + 1] != expected[i + 1]) ++changed;
                totalChanged += changed;
                printf("COMPLETE_OUTPUT profile=%s repeat=%d changed_words=%zu total_words=%d\n",
                    profile.c_str(), repeat, changed, m * n);
            }
        }
        cublasLtDestroy(handle);
        printf("REAL_GDN_Z_%s_%s\n", nativeZ ? "TYPED_NATIVE_Z128" : (shared ? "SHARED_M32" : "BOTH_DEFAULT_PROFILES"), totalChanged ? "FAILED" : "PASSED");
        return totalChanged ? 1 : 0;
    } catch (const std::exception& e) { fprintf(stderr, "ERROR %s\n", e.what()); return 2; }
}
