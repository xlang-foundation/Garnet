// Use the existing XLang package ABI; no libffi, CUDA calls during Load or SDK edits.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "xlang3/xlang3.h"
#include <string>
#include <mutex>
#include <cstring>
namespace {
class ContextNative {
public:
    int Abi(){return int(X3_ABI_VERSION);}
    int Observe(std::string target){
        const std::string allowed="D:\\CantorAI\\5090-study-20261023\\results\\fp8-z128-native-package-audit-attempt1\\";
        char absolute[32768]{};const DWORD count=GetFullPathNameA(target.c_str(),DWORD(sizeof(absolute)),absolute,nullptr);
        if(!count||count>=sizeof(absolute)||_strnicmp(absolute,allowed.c_str(),allowed.size()))return 20;
        static std::mutex lock;std::lock_guard<std::mutex> guard(lock);
        // Fixed, hash-checked observer from the immutable failed libffi attempt.
        static HMODULE library=LoadLibraryExA("D:\\CantorAI\\5090-study-20261023\\results\\fp8-z128-context-audit-attempt1\\build\\context-audit.dll",nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!library)return 21;
        using Call=int(__cdecl*)(const char*);const auto call=reinterpret_cast<Call>(GetProcAddress(library,"AuditZ128Context"));
        return call?call(absolute):22;
    }
    BEGIN_PACKAGE(ContextNative)
        APISET().AddFunc<0>("abi",&ContextNative::Abi);
        APISET().AddFunc<1>("observe",&ContextNative::Observe);
    END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current){
    auto* host=static_cast<X3PackageHost*>(rawHost);if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;
    try{ContextNative::BuildAPI();return ContextNative::APISET().Create(host,"fp8_z128_context_native",current);}catch(...){return X3_STATUS_ERROR;}
}
