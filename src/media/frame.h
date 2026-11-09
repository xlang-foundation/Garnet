// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "media_plugin_api.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace Garnet::Media {
inline bool IsVideoStreamTarget(std::string_view target) {
    constexpr std::string_view scheme="tcp://", loopback="tcp://127.0.0.1:";
    if(target.substr(0,scheme.size())!=scheme)return false;
    if(target.substr(0,loopback.size())!=loopback)throw std::runtime_error("video stream requires an IPv4 loopback target");
    const auto port=target.substr(loopback.size());
    if(port.empty()||port.size()>5||port.front()=='0')throw std::runtime_error("invalid video stream port");
    unsigned number=0;
    for(char digit:port){
        if(digit<'0'||digit>'9')throw std::runtime_error("invalid video stream port");
        number=number*10+unsigned(digit-'0');
    }
    if(number>65535)throw std::runtime_error("invalid video stream port");
    return true;
}
inline uint8_t Quantize(float value) {
    if(!std::isfinite(value))throw std::runtime_error("media frame contains nonfinite pixels");
    const double v=std::clamp((double(value)+1.0)*127.5,0.0,255.0),lower=std::floor(v);
    unsigned n=unsigned(lower);if(v-lower>0.5 || (v-lower==0.5 && (n&1)))++n;
    return uint8_t(n);
}
inline void Dimensions(uint32_t w,uint32_t h) {
    if(!w || !h || w>16384 || h>16384 || uint64_t(w)*h>67108864)
        throw std::runtime_error("media dimensions exceed admission limits");
}
inline GarnetMediaFrame RGBFrame(const uint8_t* data,uint32_t w,uint32_t h,int64_t pts=0) {
    GarnetMediaFrame f{};f.size=sizeof(f);f.format=GM_RGB8;f.device=GM_CPU;
    f.width=w;f.height=h;f.planes[0]=data;f.strides[0]=uint64_t(w)*3;f.pts=pts;return f;
}
inline std::vector<uint8_t> PlanarRGB(const float* data,uint32_t w,uint32_t h,uint32_t frames,uint32_t frame) {
    Dimensions(w,h);const size_t plane=size_t(w)*h;std::vector<uint8_t> out(plane*3);
    for(size_t i=0;i<plane;++i)for(size_t c=0;c<3;++c)out[i*3+c]=Quantize(data[(c*frames+frame)*plane+i]);
    return out;
}
}
