// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace Garnet {
// Explicitly opt in. The budget covers each engine's resident streamable
// weights, not tensors, KV pages or execution-context scratch memory.
struct TRTWeightStreamingPolicy {
    bool valid = true;
    int64_t budgetBytes = -1;
    int64_t fullResidentLimitBytes = 0;
    int64_t fullResidentMinimumBytes = 0;
    bool Enabled() const noexcept { return budgetBytes >= 0; }
    int64_t BudgetForEngine(int64_t weights) const noexcept {
        const bool resident = weights <= fullResidentLimitBytes ||
            (fullResidentMinimumBytes > 0 && weights >= fullResidentMinimumBytes);
        return Enabled() && weights > 0 && resident ? weights : budgetBytes;
    }
};

inline TRTWeightStreamingPolicy ParseTRTWeightStreamingPolicy(const char* value) {
    TRTWeightStreamingPolicy result;
    if (!value || !*value) return result;
    constexpr int64_t mib = 1024 * 1024;
    constexpr int64_t limit = std::numeric_limits<int64_t>::max() / mib;
    int64_t amount = 0;
    for (const char* cursor = value; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9' ||
            amount > (limit - (*cursor - '0')) / 10) {
            result.valid = false;
            return result;
        }
        amount = amount * 10 + (*cursor - '0');
    }
    result.budgetBytes = amount * mib;
    return result;
}

inline TRTWeightStreamingPolicy ParseTRTWeightStreamingPolicy(const char* budget, const char* fullResidentLimit) {
    auto result=ParseTRTWeightStreamingPolicy(budget);
    if(fullResidentLimit && *fullResidentLimit){
        const auto limit=ParseTRTWeightStreamingPolicy(fullResidentLimit);
        result.valid=result.valid && limit.valid && (result.Enabled() || limit.budgetBytes==0);
        if(limit.valid)result.fullResidentLimitBytes=limit.budgetBytes;
    }
    return result;
}

inline TRTWeightStreamingPolicy ReadTRTWeightStreamingPolicy() {
    return ParseTRTWeightStreamingPolicy(std::getenv("GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB"),
        std::getenv("GARNET_TRT_FULL_RESIDENT_WEIGHT_LIMIT_MIB"));
}

inline TRTWeightStreamingPolicy ReadTRTRuntimeWeightStreamingPolicy() {
    auto policy = ReadTRTWeightStreamingPolicy();
    const char* budget = std::getenv("GARNET_TRT_RUNTIME_WEIGHT_STREAMING_BUDGET_MIB");
    if (budget && *budget) {
        const auto runtime = ParseTRTWeightStreamingPolicy(budget);
        policy.valid = policy.valid && policy.Enabled() && runtime.valid;
        if (policy.valid) policy.budgetBytes = runtime.budgetBytes;
    }
    const char* minimum = std::getenv("GARNET_TRT_RUNTIME_FULL_RESIDENT_MIN_WEIGHT_MIB");
    if (minimum && *minimum) {
        const auto threshold = ParseTRTWeightStreamingPolicy(minimum);
        policy.valid = policy.valid && policy.Enabled() && threshold.valid && threshold.budgetBytes > 0;
        if (policy.valid) policy.fullResidentMinimumBytes = threshold.budgetBytes;
    }
    return policy;
}

inline std::string TRTWeightStreamingFingerprint() {
    const auto policy = ReadTRTWeightStreamingPolicy();
    if (!policy.valid) return "weight_streaming_v1:invalid\n";
    if (!policy.Enabled()) return {};
    return "weight_streaming_v1:" + std::to_string(policy.budgetBytes) +
        (policy.fullResidentLimitBytes ? ":full_resident_limit:"+std::to_string(policy.fullResidentLimitBytes) : "") + "\n";
}
}
