// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Garnet {
// Host-side RGB observation shared by image and video frontends. Callers make
// the tensor CPU-resident explicitly; this class has no hardware dependency.
class TensorPixels {
    static X3TensorInfo HostFloat(const X::Tensor& tensor) {
        const auto info=tensor.Info();
        if(info.symbolic || info.device_type!=0 || info.dtype!=X3_TENSOR_FLOAT32 ||
            info.rank<1 || info.rank>5 || !info.data)
            throw X::Error("pixel operations require materialized CPU FP32 tensors");
        int64_t stride=4;uint64_t count=1;
        for(uint32_t axis=info.rank;axis-->0;){
            const auto extent=info.shape[axis];
            if(extent<1 || count>UINT64_MAX/uint64_t(extent) || stride>INT64_MAX/extent ||
                (extent>1 && info.strides[axis]!=stride))
                throw X::Error("RGB8 requires bounded contiguous storage");
            count*=uint64_t(extent);stride*=extent;
        }
        if(count>SIZE_MAX/4 || info.byte_size<count*4)
            throw X::Error("RGB8 source storage is too small");
        return info;
    }
    static X3TensorInfo RGBLayout(const X::Tensor& tensor) {
        const auto info=HostFloat(tensor);
        if((info.rank!=4 && info.rank!=5) || info.shape[0]!=1 || info.shape[1]!=3)
            throw X::Error("RGB pixels require [1,3,H,W] or [1,3,F,H,W]");
        return info;
    }
    static uint64_t Frames(const X3TensorInfo& info){return info.rank==5?uint64_t(info.shape[2]):1;}
    static uint64_t Height(const X3TensorInfo& info){return uint64_t(info.shape[info.rank-2]);}
    static uint64_t Width(const X3TensorInfo& info){return uint64_t(info.shape[info.rank-1]);}
    static bool Overlap(const X3TensorInfo& a,const X3TensorInfo& b) {
        uint64_t an=4,bn=4;
        for(uint32_t i=0;i<a.rank;++i)an*=uint64_t(a.shape[i]);
        for(uint32_t i=0;i<b.rank;++i)bn*=uint64_t(b.shape[i]);
        const auto pa=reinterpret_cast<uintptr_t>(a.data),pb=reinterpret_cast<uintptr_t>(b.data);
        return pa>=pb?pa-pb<bn:pb-pa<an;
    }
public:
    X::Value RGB8(X::Value source,X::Value frameValue) {
        if(!X::Tensor::IsTensor(source) || !frameValue.IsInt64())
            throw X::Error("RGB8 requires a tensor and an integer frame index");
        X::Tensor tensor(source);const auto info=RGBLayout(tensor);
        const uint64_t frames=info.rank==5?uint64_t(info.shape[2]):1;
        const auto frame=frameValue.ToLongLong();
        if(frame<0 || uint64_t(frame)>=frames)throw X::Error("RGB8 frame index is out of range");
        const uint64_t plane=uint64_t(info.shape[info.rank-2])*uint64_t(info.shape[info.rank-1]);
        std::vector<unsigned char> pixels(size_t(plane*3));
        auto use=tensor.Acquire();const auto* values=static_cast<const float*>(info.data);
        for(uint64_t position=0;position<plane;++position)for(uint64_t channel=0;channel<3;++channel){
            const double value=values[(channel*frames+uint64_t(frame))*plane+position];
            if(!std::isfinite(value))throw X::Error("RGB8 tensor contains nonfinite pixels");
            const double scaled=std::clamp((value+1.0)*127.5,0.0,255.0);
            const double lower=std::floor(scaled),fraction=scaled-lower;
            auto rounded=static_cast<unsigned int>(lower);
            if(fraction>0.5 || (fraction==0.5 && (rounded&1)))++rounded;
            pixels[size_t(position*3+channel)]=static_cast<unsigned char>(rounded);
        }
        return X::Value::Bytes(tensor.host(),pixels.data(),pixels.size());
    }
    // Accumulate a normalized float tile into a CPU canvas. Feather widths are
    // [top,left,bottom,right], in pixels. Weights have shape [F,H,W].
    void BlendInto(X::Value source,X::Value destination,X::Value weightsValue,
        X::Value frameValue,X::Value topValue,X::Value leftValue,X::Value featherValue) {
        X::Tensor tile(source),canvas(destination),weights(weightsValue);
        const auto a=RGBLayout(tile),b=RGBLayout(canvas),w=HostFloat(weights);
        if(!frameValue.IsInt64() || !topValue.IsInt64() || !leftValue.IsInt64() || !featherValue.IsList() || featherValue.Size()!=4)
            throw X::Error("tile position and four feather widths must be integers");
        const auto top=topValue.ToLongLong(),left=leftValue.ToLongLong();
        const auto first=frameValue.ToLongLong();
        const uint64_t h=Height(a),width=Width(a),ch=Height(b),cw=Width(b),frames=Frames(a);
        const uint64_t canvasFrames=Frames(b);
        if(b.readonly || w.readonly || first<0 || uint64_t(first)>canvasFrames || frames>canvasFrames-uint64_t(first) || w.rank!=3 ||
            uint64_t(w.shape[0])!=canvasFrames || uint64_t(w.shape[1])!=ch || uint64_t(w.shape[2])!=cw || top<0 || left<0 ||
            uint64_t(top)>ch || h>ch-uint64_t(top) || uint64_t(left)>cw || width>cw-uint64_t(left) ||
            Overlap(a,b) || Overlap(a,w) || Overlap(b,w))
            throw X::Error("tile/canvas dimensions, mutable storage or ownership overlap are invalid");
        uint64_t feather[4]{};
        for(uint64_t i=0;i<4;++i){
            auto value=featherValue.Get(i);
            if(!value.IsInt64() || value.ToLongLong()<0 || uint64_t(value.ToLongLong())>(i%2?width:h))
                throw X::Error("tile feather width exceeds its image dimension");
            feather[i]=uint64_t(value.ToLongLong());
        }
        auto use=X::Tensor::AcquireMany({{tile,X3_TENSOR_READ},{canvas,X3_TENSOR_WRITE},{weights,X3_TENSOR_WRITE}});
        const auto* input=static_cast<const float*>(a.data);
        auto* output=static_cast<float*>(b.data);auto* weight=static_cast<float*>(w.data);
        for(uint64_t y=0;y<h;++y)for(uint64_t x=0;x<width;++x){
            float wy=1,wx=1;
            if(feather[0] && y<feather[0])wy=std::min(wy,float(y+1)/float(feather[0]+1));
            if(feather[2] && y>=h-feather[2])wy=std::min(wy,float(h-y)/float(feather[2]+1));
            if(feather[1] && x<feather[1])wx=std::min(wx,float(x+1)/float(feather[1]+1));
            if(feather[3] && x>=width-feather[3])wx=std::min(wx,float(width-x)/float(feather[3]+1));
            const float alpha=wx*wy;const uint64_t position=(uint64_t(top)+y)*cw+uint64_t(left)+x;
            for(uint64_t frame=0;frame<frames;++frame){
                const uint64_t targetFrame=uint64_t(first)+frame;
                weight[targetFrame*ch*cw+position]+=alpha;
                for(uint64_t channel=0;channel<3;++channel){
                    const float value=input[(channel*frames+frame)*h*width+y*width+x];
                    if(!std::isfinite(value))throw X::Error("tile contains nonfinite pixels");
                    output[(channel*canvasFrames+targetFrame)*ch*cw+position]+=value*alpha;
                }
            }
        }
    }
    void Normalize(X::Value destination,X::Value weightsValue) {
        X::Tensor canvas(destination),weights(weightsValue);const auto a=RGBLayout(canvas),w=HostFloat(weights);
        const auto height=Height(a),width=Width(a),frames=Frames(a),plane=height*width;
        if(a.readonly || w.rank!=3 || uint64_t(w.shape[0])!=frames || uint64_t(w.shape[1])!=height || uint64_t(w.shape[2])!=width || Overlap(a,w))
            throw X::Error("normalization requires a mutable canvas and matching nonoverlapping weights");
        auto use=X::Tensor::AcquireMany({{canvas,X3_TENSOR_WRITE},{weights,X3_TENSOR_READ}});
        auto* data=static_cast<float*>(a.data);const auto* weight=static_cast<const float*>(w.data);
        for(uint64_t frame=0;frame<frames;++frame)for(uint64_t position=0;position<plane;++position){
            const auto weightValue=weight[frame*plane+position];
            if(!std::isfinite(weightValue) || weightValue<=0)throw X::Error("tile canvas has uncovered or invalid weights");
            for(uint64_t channel=0;channel<3;++channel){
                auto& value=data[(channel*frames+frame)*plane+position];value/=weightValue;
                if(!std::isfinite(value))throw X::Error("tile canvas contains nonfinite pixels");
            }
        }
    }
    BEGIN_PACKAGE(TensorPixels)
        APISET().AddFunc<2>("rgb8",&TensorPixels::RGB8);
        APISET().AddFunc<7>("blend_into",&TensorPixels::BlendInto);
        APISET().AddFunc<2>("normalize",&TensorPixels::Normalize);
    END_PACKAGE
};
}
