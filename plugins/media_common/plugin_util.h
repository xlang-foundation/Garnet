// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "frame.h"
#include "nlohmann/json.hpp"
#include <memory>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <cstring>
namespace GMPlugin {
using Json=nlohmann::json;
inline int64_t Integer(const Json& options,const char* key,int64_t fallback,int64_t minimum,int64_t maximum){
    const auto value=options.value(key,Json(fallback));
    if(!value.is_number_integer()||value<minimum||value>maximum)throw std::runtime_error(std::string("invalid integer ")+key);
    return value.get<int64_t>();
}
inline void Error(char* out,size_t n,const char* text){if(out&&n)std::snprintf(out,n,"%s",text);}
template<class F> int Guard(char* out,size_t n,F action) noexcept {try{action();return 0;}catch(const std::exception& e){Error(out,n,e.what());return 1;}catch(...){Error(out,n,"unknown media error");return 1;}}
inline void Write(const char* path,const void* data,size_t size){auto p=std::filesystem::u8path(path);if(!p.parent_path().empty())std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p,std::ios::binary);out.write(static_cast<const char*>(data),size);if(!out)throw std::runtime_error("image output write failed");}
inline std::vector<uint8_t> PixelsRGB(const GarnetMediaFrame* f){
    if(!f||f->size<sizeof(*f)||f->device!=GM_CPU||!f->planes[0])throw std::runtime_error("CPU image frame required");
    Garnet::Media::Dimensions(f->width,f->height);const size_t channels=f->format==GM_RGBA8?4:3;
    if((f->format!=GM_RGB8&&f->format!=GM_BGR8&&f->format!=GM_RGBA8)||f->strides[0]<f->width*channels)throw std::runtime_error("invalid RGB image layout");
    std::vector<uint8_t> out(size_t(f->width)*f->height*3);for(uint32_t y=0;y<f->height;++y){auto* row=static_cast<const uint8_t*>(f->planes[0])+y*f->strides[0];
        for(uint32_t x=0;x<f->width;++x)for(size_t c=0;c<3;++c){unsigned value=row[x*channels+(f->format==GM_BGR8?2-c:c)];
            if(channels==4){unsigned alpha=row[x*4+3];value=(value*alpha+255*(255-alpha)+127)/255;}out[(size_t(y)*f->width+x)*3+c]=uint8_t(value);}}
    return out;
}
}
