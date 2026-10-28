// Diagnostic CPU-only NVTX3 bridge. No CUDA/Garnet/model linkage.
#include <Windows.h>
#include <nvtx3/nvToolsExt.h>

extern "C" __declspec(dllexport) int qwen35NvtxPush(const char* name) {
    return nvtxRangePushA(name);
}
extern "C" __declspec(dllexport) int qwen35NvtxPop() {
    return nvtxRangePop();
}
extern "C" __declspec(dllexport) unsigned long qwen35NvtxThreadId() {
    return GetCurrentThreadId();
}
extern "C" __declspec(dllexport) int qwen35NvtxVersion() {
    return NVTX_VERSION;
}
