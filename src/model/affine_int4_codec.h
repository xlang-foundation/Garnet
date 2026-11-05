// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "float16_codec.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace Garnet {
// MLX affine Q4: eight low-nibble-first values per little-endian uint32;
// Typed BF16/FP16 scale and additive bias per64 input columns; unsigned INT4.
enum class AffineInt4CoefficientType {BF16,FP16};
struct AffineInt4Layout {
    size_t rows=0,columns=0,groupsPerRow=0,weightBytes=0,coefficientBytes=0;
    AffineInt4CoefficientType coefficientType=AffineInt4CoefficientType::BF16;

    static AffineInt4Layout Create(size_t rows,size_t columns,
                                  AffineInt4CoefficientType type=AffineInt4CoefficientType::BF16) {
        if(!rows||!columns||columns%64||rows>SIZE_MAX/columns)
            throw std::invalid_argument("Affine INT4 requires nonempty rows and group64 columns");
        if(type!=AffineInt4CoefficientType::BF16&&type!=AffineInt4CoefficientType::FP16)
            throw std::invalid_argument("Unsupported affine INT4 coefficient type");
        const size_t elements=rows*columns;
        AffineInt4Layout result;
        result.rows=rows;result.columns=columns;result.groupsPerRow=columns/64;
        result.weightBytes=elements/2;result.coefficientBytes=elements/32;
        result.coefficientType=type;
        return result;
    }
};

inline float AffineInt4BF16(const unsigned char* bytes) {
    const uint32_t bits=(uint32_t(bytes[0])|uint32_t(bytes[1])<<8)<<16;
    float result;std::memcpy(&result,&bits,sizeof(result));return result;
}

inline float AffineInt4Coefficient(const unsigned char* bytes,AffineInt4CoefficientType type) {
    if(type==AffineInt4CoefficientType::BF16)return AffineInt4BF16(bytes);
    if(type!=AffineInt4CoefficientType::FP16)throw std::invalid_argument("Unknown affine coefficient type");
    return Float16ToFloat32(uint16_t(bytes[0])|(uint16_t(bytes[1])<<8));
}

inline float AffineInt4RoundBF16(float value) {
    if(!std::isfinite(value))throw std::invalid_argument("Cannot round nonfinite affine embedding");
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
    bits=(bits+0x7fffu+((bits>>16)&1u))&0xffff0000u;
    float result;std::memcpy(&result,&bits,sizeof(result));
    if(!std::isfinite(result))throw std::invalid_argument("Affine embedding exceeds BF16 range");
    return result;
}

class AffineInt4View {
    AffineInt4Layout layout_;
    const unsigned char *weights_=nullptr,*scales_=nullptr,*biases_=nullptr;
public:
    AffineInt4View(AffineInt4Layout layout,const void* weights,size_t weightBytes,
                  const void* scales,size_t scaleBytes,const void* biases,size_t biasBytes)
        :layout_(AffineInt4Layout::Create(layout.rows,layout.columns,layout.coefficientType)),
         weights_(static_cast<const unsigned char*>(weights)),
         scales_(static_cast<const unsigned char*>(scales)),
         biases_(static_cast<const unsigned char*>(biases)) {
        if(!weights_||!scales_||!biases_||weightBytes!=layout_.weightBytes||
           scaleBytes!=layout_.coefficientBytes||biasBytes!=layout_.coefficientBytes)
            throw std::invalid_argument("Affine INT4 payload length mismatch");
    }
    const AffineInt4Layout& Layout() const {return layout_;}
    void DecodeRow(size_t row,float* output,size_t count) const {
        if(row>=layout_.rows||!output||count!=layout_.columns)
            throw std::invalid_argument("Affine INT4 embedding row/width out of bounds");
        for(size_t group=0;group<layout_.groupsPerRow;++group) {
            const size_t coefficient=row*layout_.groupsPerRow+group;
            const float scale=AffineInt4Coefficient(scales_+2*coefficient,layout_.coefficientType);
            const float bias=AffineInt4Coefficient(biases_+2*coefficient,layout_.coefficientType);
            if(!std::isfinite(scale)||!std::isfinite(bias))
                throw std::invalid_argument("Nonfinite affine INT4 coefficient");
            const size_t begin=group*64;
            for(size_t i=0;i<64;++i) {
                const size_t element=row*layout_.columns+begin+i;
                const unsigned value=(weights_[element/2]>>((element%2)*4))&15;
                const float decoded=std::fma(float(value),scale,bias);
                if(!std::isfinite(decoded))throw std::invalid_argument("Affine INT4 decoding overflow");
                output[begin+i]=decoded;
            }
        }
    }
    void DecodeEmbeddingRow(size_t row,float* output,size_t count) const {
        DecodeRow(row,output,count);
        // MLX QuantizedEmbedding invokes dequantize without a dtype override:
        // Output storage follows coefficient dtype, before host FP32 exposure.
        for(size_t i=0;i<count;++i) {
            if(layout_.coefficientType==AffineInt4CoefficientType::BF16)output[i]=AffineInt4RoundBF16(output[i]);
            else {
                output[i]=Float16ToFloat32(Float32ToFloat16RNE(output[i]));
                if(!std::isfinite(output[i]))throw std::invalid_argument("Affine embedding exceeds FP16 range");
            }
        }
    }
};
}
