// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Garnet::TensorContracts {
using Shape = std::vector<int64_t>;

inline int Axis(int axis, size_t rank) {
    if (rank > size_t(std::numeric_limits<int>::max())) throw std::invalid_argument("tensor rank overflows");
    if (axis < 0) axis += int(rank);
    if (axis < 0 || size_t(axis) >= rank) throw std::invalid_argument("tensor axis out of range");
    return axis;
}
inline int Slice(const Shape& shape, int axis, int64_t start, int64_t count) {
    axis = Axis(axis, shape.size());
    if (start < 0 || count < 1 || start > shape[axis] || count > shape[axis] - start)
        throw std::invalid_argument("tensor_slice start/count out of range");
    return axis;
}
inline int Concat(const Shape& left, const Shape& right, int axis) {
    if (left.size() != right.size()) throw std::invalid_argument("tensor_concat ranks differ");
    axis = Axis(axis, left.size());
    for (size_t index = 0; index < left.size(); ++index)
        if (int(index) != axis && left[index] != right[index]) throw std::invalid_argument("tensor_concat extents differ");
    if (left[axis] < 1 || right[axis] < 1 || left[axis] > std::numeric_limits<int64_t>::max() - right[axis])
        throw std::invalid_argument("tensor_concat extent overflows");
    return axis;
}
inline void PositiveEpsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0) throw std::invalid_argument("normalization epsilon must be positive and finite");
}
inline void Attention(const Shape& query, const Shape& key, const Shape& value) {
    if (query.size()!=4 || key.size()!=4 || value.size()!=4 || query[0]!=key[0] || key[0]!=value[0] ||
        key[2]<1 || query[2]%key[2] || key[2]!=value[2] || query[3]!=key[3] || key[1]!=value[1])
        throw std::invalid_argument("scaled_dot_product_attention requires compatible [B,S,H,D] tensors");
}
inline void RotaryInterleaved(const Shape& source, const Shape& cosine, const Shape& sine) {
    if (source.size()!=4 || source[3]<2 || source[3]%2)
        throw std::invalid_argument("rotary_interleaved requires [B,S,H,even D]");
    if (cosine.size()!=4 || sine!=cosine || (cosine[0]!=1&&cosine[0]!=source[0]) || cosine[1]!=source[1] ||
        (cosine[2]!=1&&cosine[2]!=source[2]) || cosine[3]!=source[3])
        throw std::invalid_argument("rotary tables require broadcastable [B,S,H,D]");
}
inline void Convolution(const Shape& source,const Shape& filter,int spatial,int groups,
    const Shape& stride,const Shape& before,const Shape& after,const Shape& dilation) {
    constexpr int64_t limit=std::numeric_limits<int>::max();
    if(source.size()!=size_t(spatial+2) || filter.size()!=source.size() || groups<1 ||
        filter[1]>limit/groups || source[1]!=filter[1]*groups || filter[0]%groups ||
        stride.size()!=size_t(spatial) || before.size()!=stride.size() ||
        after.size()!=stride.size() || dilation.size()!=stride.size())
        throw std::invalid_argument("convolution ranks, channel groups, or spatial attributes differ");
    for(int axis=0;axis<spatial;++axis){
        if(stride[axis]<1 || stride[axis]>limit || dilation[axis]<1 || dilation[axis]>limit ||
            before[axis]<0 || before[axis]>limit || after[axis]<0 || after[axis]>limit ||
            filter[axis+2]<1 || filter[axis+2]>limit || source[axis+2]<1 || source[axis+2]>limit)
            throw std::invalid_argument("convolution extent/stride/padding/dilation out of range");
        const int64_t effective=(filter[axis+2]-1)*dilation[axis]+1;
        if(source[axis+2]+before[axis]+after[axis]<effective)
            throw std::invalid_argument("convolution kernel exceeds padded input");
    }
}
}
