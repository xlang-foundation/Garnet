// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {class TRTGraphReservation;}
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include <cuda_runtime_api.h>
#include <utility>
#include <vector>
namespace Garnet {
// A captured context and its pointers remain reserved until the containing
// execution bundle retires. Per-submit tensor fences belong to the outer run.
class TRTGraphReservation {
public:
    virtual ~TRTGraphReservation()=default;
    virtual cudaGraph_t Graph() const=0;
    virtual cudaGraphExec_t Executable() const=0;
    virtual int Device() const=0;
    virtual cudaStream_t Stream() const=0;
    virtual X::Value Output() const=0;
    virtual const std::vector<std::pair<X::Tensor,X3TensorAccess>>& Tensors() const=0;
};
}
#endif
