// SPDX-License-Identifier: Apache-2.0
#include "../../src/cuda/logits_top1_batch_dispatch.h"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace Garnet;
    if (!ParseParallelBatchTop1("1")) throw std::runtime_error("explicit opt-in rejected");
    for (const char* value : {static_cast<const char*>(nullptr), "", "0", "true", "01", "1 ", "11", "-1"})
        if (ParseParallelBatchTop1(value)) throw std::runtime_error("implicit opt-in accepted");
    for (int rows : {-1, 0, 1, 4, 8, 16, 128, 129})
        for (int vocab : {-1, 0, 1, 8191, 8192, 248320, 2147483647}) {
            if (SelectParallelBatchTop1(false, rows, vocab)) throw std::runtime_error("default route changed");
            bool expected = rows > 0 && rows < 129 && vocab > 8191;
            if (SelectParallelBatchTop1(true, rows, vocab) != expected) throw std::runtime_error("fallback boundary changed");
        }
    std::cout << "BATCH_TOP1_POLICY_PASS parse=9 boundaries=112\n";
}
