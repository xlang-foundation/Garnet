// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Garnet {
// MLX Conv3d stores O,T,H,W,C; canonical Garnet patches and weights use
// O,C,T,H,W. Convert only this small patch projection, never text matrices.
inline std::vector<float> MlxVisionPatchToCanonical(const std::vector<float>& source,
    size_t outputs,size_t temporal,size_t height,size_t width,size_t channels) {
    size_t count=1;
    for(auto dimension:{outputs,temporal,height,width,channels}) {
        if(!dimension||count>SIZE_MAX/dimension)throw std::invalid_argument("Vision patch shape overflow");
        count*=dimension;
    }
    if(source.size()!=count)throw std::invalid_argument("Vision patch payload shape mismatch");
    std::vector<float> result(count);
    for(size_t o=0;o<outputs;++o)for(size_t c=0;c<channels;++c)
      for(size_t t=0;t<temporal;++t)for(size_t h=0;h<height;++h)for(size_t w=0;w<width;++w)
        result[((((o*channels+c)*temporal+t)*height+h)*width+w)]=
            source[((((o*temporal+t)*height+h)*width+w)*channels+c)];
    return result;
}
}
