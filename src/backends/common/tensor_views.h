// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include <climits>
#include <cstdint>
#include <vector>

namespace Garnet {
// Metadata-only views retain the SDK storage owner and its access fences. No
// backend-specific memory API, synchronization, or payload access is involved.
class TensorViews {
    static int64_t ItemBytes(X3TensorDType dtype) {
        switch(dtype){
        case X3_TENSOR_FLOAT64:case X3_TENSOR_INT64:return 8;
        case X3_TENSOR_FLOAT32:case X3_TENSOR_INT32:return 4;
        case X3_TENSOR_FLOAT16:case X3_TENSOR_BFLOAT16:case X3_TENSOR_UINT16:return 2;
        case X3_TENSOR_FLOAT8_E4M3FN:case X3_TENSOR_FLOAT8_E4M3FNUZ:
        case X3_TENSOR_FLOAT8_E5M2:case X3_TENSOR_FLOAT8_E5M2FNUZ:case X3_TENSOR_UINT8:return 1;
        default:throw X::Error("unsupported tensor view dtype");
        }
    }
    X::Value View(X::Value source,X::Value dimensions,int64_t start,bool reshape) {
        if(!X::Tensor::IsTensor(source) || !dimensions.IsList() || dimensions.Size()>64)
            throw X::Error("tensor view requires a tensor and a shape list of rank at most 64");
        X::Tensor tensor(source);const auto info=tensor.Info();
        if(info.symbolic || info.rank==UINT32_MAX || !info.data)
            throw X::Error("reshape view requires materialized, nonempty storage");
        int64_t stride=ItemBytes(info.dtype);uint64_t sourceCount=1;
        for(uint32_t axis=info.rank;axis-->0;){
            const auto extent=info.shape[axis];
            if(extent<1 || sourceCount>UINT64_MAX/uint64_t(extent) || stride>INT64_MAX/extent ||
                (extent>1 && info.strides[axis]!=stride))throw X::Error("reshape view requires bounded contiguous storage");
            sourceCount*=uint64_t(extent);stride*=extent;
        }
        std::vector<int64_t> shape;uint64_t targetCount=1;
        for(uint64_t axis=0;axis<dimensions.Size();++axis){
            const auto value=dimensions.Get(axis);
            if(!value.IsInt64())throw X::Error("reshape dimensions must be integers");
            const auto extent=value.ToLongLong();
            if(extent<1 || targetCount>UINT64_MAX/uint64_t(extent))throw X::Error("reshape dimensions out of range");
            targetCount*=uint64_t(extent);shape.push_back(extent);
        }
        if(reshape && sourceCount!=targetCount)throw X::Error("reshape view must preserve the element count");
        if(start<0 || uint64_t(start)>sourceCount || targetCount>sourceCount-uint64_t(start))
            throw X::Error("flat tensor view exceeds the source's logical elements");
        std::vector<int64_t> strides(shape.size());stride=ItemBytes(info.dtype);
        for(size_t axis=shape.size();axis-->0;){
            strides[axis]=stride;
            if(stride>INT64_MAX/shape[axis])throw X::Error("reshape strides overflow");
            stride*=shape[axis];
        }
        return tensor.View(shape,strides,uint64_t(start)*uint64_t(ItemBytes(info.dtype)));
    }
public:
    X::Value Reshape(X::Value source,X::Value dimensions) {
        return View(source,dimensions,0,true);
    }
    // A contiguous logical window, in elements, with retained storage ownership.
    // This also exposes packed outputs without copying multi-gigabyte caches.
    X::Value Slice(X::Value source,X::Value start,X::Value dimensions) {
        if(!start.IsInt64())throw X::Error("flat tensor view start must be an integer");
        return View(source,dimensions,start.ToLongLong(),false);
    }
    BEGIN_PACKAGE(TensorViews)
        APISET().AddFunc<2>("reshape",&TensorViews::Reshape);
        APISET().AddFunc<3>("slice",&TensorViews::Slice);
    END_PACKAGE
};
}
