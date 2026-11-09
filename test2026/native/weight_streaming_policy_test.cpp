// SPDX-License-Identifier: Apache-2.0
#include "weight_streaming_policy.h"
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char* description) {
    if (!condition) throw std::runtime_error(description);
}
void SetEnvironment(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
void SetBudget(const char* value) {SetEnvironment("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB",value);}
struct RestoreEnvironment {
    bool existed;
    std::string name,value;
    explicit RestoreEnvironment(const char* key):name(key) {
        const char* old = std::getenv(key);
        existed = old != nullptr;
        if (old) value = old;
    }
    ~RestoreEnvironment() { SetEnvironment(name.c_str(),existed ? value.c_str() : nullptr); }
};
}

int main() {
    try {
        RestoreEnvironment restore("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB");
        RestoreEnvironment restoreLimit("GARNET_TRT_FULL_RESIDENT_WEIGHT_LIMIT_MIB");
        RestoreEnvironment restoreRuntime("GARNET_TRT_RUNTIME_WEIGHT_STREAMING_BUDGET_MIB");
        RestoreEnvironment restoreMinimum("GARNET_TRT_RUNTIME_FULL_RESIDENT_MIN_WEIGHT_MIB");
        SetEnvironment("GARNET_TRT_FULL_RESIDENT_WEIGHT_LIMIT_MIB",nullptr);
        for (const char* value : {static_cast<const char*>(nullptr), ""}) {
            const auto policy = Garnet::ParseTRTWeightStreamingPolicy(value);
            Require(policy.valid && !policy.Enabled(), "Unset policy must retain existing behavior");
        }
        const auto zero = Garnet::ParseTRTWeightStreamingPolicy("0");
        Require(zero.valid && zero.Enabled() && zero.budgetBytes == 0, "Zero must enable maximum streaming");
        const auto resident = Garnet::ParseTRTWeightStreamingPolicy("512");
        Require(resident.valid && resident.budgetBytes == 536870912, "MiB conversion failed");
        const auto max = Garnet::ParseTRTWeightStreamingPolicy("8796093022207");
        Require(max.valid && max.budgetBytes == 9223372036853727232LL, "Maximum safe MiB value rejected");
        for (const char* value : {"-1", "+1", "1.5", "1M", " 1", "1 ",
                "8796093022208", "18446744073709551615"}) {
            Require(!Garnet::ParseTRTWeightStreamingPolicy(value).valid, "Malformed or overflowing budget accepted");
        }
        const auto tiered=Garnet::ParseTRTWeightStreamingPolicy("256","1024");
        Require(tiered.valid && tiered.BudgetForEngine(1024LL<<20)==(1024LL<<20),"Small engine must retain all streamable weights");
        Require(tiered.BudgetForEngine(1025LL<<20)==(256LL<<20),"Large engine must keep the bounded streaming budget");
        Require(tiered.BudgetForEngine(0)==(256LL<<20),"Weightless engine changed budget policy");
        Require(Garnet::ParseTRTWeightStreamingPolicy("256","0").BudgetForEngine(512LL<<20)==(256LL<<20),"Zero limit must preserve default policy");
        Require(!Garnet::ParseTRTWeightStreamingPolicy(nullptr,"1024").valid,"Full residency requires explicit streaming opt-in");
        for(const char* value:{"-1","1.5","8796093022208"})
            Require(!Garnet::ParseTRTWeightStreamingPolicy("256",value).valid,"Invalid residency limit accepted");
        SetBudget(nullptr);
        Require(Garnet::TRTWeightStreamingFingerprint().empty(), "Default cache fingerprint changed");
        SetBudget("0");
        const auto zeroFingerprint = Garnet::TRTWeightStreamingFingerprint();
        Require(!zeroFingerprint.empty(), "Streaming engine must use a distinct cache");
        SetBudget("00");
        Require(Garnet::TRTWeightStreamingFingerprint() == zeroFingerprint, "Equivalent budgets must share cache identity");
        SetBudget("512");
        Require(Garnet::TRTWeightStreamingFingerprint() != zeroFingerprint, "Different budgets reused cache identity");
        const auto boundedFingerprint=Garnet::TRTWeightStreamingFingerprint();
        SetEnvironment("GARNET_TRT_RUNTIME_WEIGHT_STREAMING_BUDGET_MIB", "4096");
        Require(Garnet::ReadTRTRuntimeWeightStreamingPolicy().budgetBytes == (4096LL << 20),
            "Runtime budget must change engine residency");
        SetEnvironment("GARNET_TRT_RUNTIME_FULL_RESIDENT_MIN_WEIGHT_MIB", "7000");
        const auto large = Garnet::ReadTRTRuntimeWeightStreamingPolicy();
        Require(large.valid && large.BudgetForEngine(7392LL << 20) == (7392LL << 20) &&
            large.BudgetForEngine(5940LL << 20) == (4096LL << 20),
            "Large engine residency must preserve bounded streaming for smaller engines");
        Require(Garnet::TRTWeightStreamingFingerprint() == boundedFingerprint,
            "Runtime budget must preserve the existing serialized engine cache");
        SetBudget(nullptr);
        Require(!Garnet::ReadTRTRuntimeWeightStreamingPolicy().valid,
            "Runtime override requires an engine compiled with streaming enabled");
        SetBudget("512");
        SetEnvironment("GARNET_TRT_RUNTIME_WEIGHT_STREAMING_BUDGET_MIB", nullptr);
        SetEnvironment("GARNET_TRT_RUNTIME_FULL_RESIDENT_MIN_WEIGHT_MIB", nullptr);
        SetEnvironment("GARNET_TRT_FULL_RESIDENT_WEIGHT_LIMIT_MIB","1024");
        Require(Garnet::TRTWeightStreamingFingerprint()!=boundedFingerprint,"Full-resident limit reused bounded-only cache identity");
        SetBudget("invalid");
        Require(!Garnet::ReadTRTWeightStreamingPolicy().valid, "Invalid environment silently disabled streaming");
        std::cout << "TRT weight streaming policy tests PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
